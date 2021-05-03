from PIL import Image
import requests
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50, resnet101
import torchvision.transforms as T
torch.set_grad_enabled(False);
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, models, transforms
#import json
import math


class resnet101_backbone(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-101 backbone
        self.backbone = resnet101()
        del self.backbone.layer4
        del self.backbone.avgpool
        del self.backbone.fc
        # create conversion layer
        self.conv = nn.Conv2d(1024, hidden_dim, 1)

        self.encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)


        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        #self.backbone.layer4.register_forward_hook(conv_hook)
        #self.transformer.encoder.layers[-1].self_attn.register_forward_hook(attn_hook)
        #self.transformer.encoder.norm.register_forward_hook(enc_hook)

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)

        # convert from 1024 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the encoder
        h = self.encoder(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        return h



class ClassifierHead(nn.Module):

    def __init__(self,model, query_pos, col_embed, row_embed, size, num_classes):
        super().__init__()
        self.backbone = model.backbone
        self.conv = model.conv
        self.encoder = model.encoder
        self.size = math.ceil(size/32)
        #self.size = 13

        self.conv_features, self.attn_weights, self.enc_output = [], [], []
        #def enc_hook(module, input, output):
        #    self.enc_output.clear()
        #    self.enc_output.append(output)

        def attn_hook(module, input, output):
            self.attn_weights.clear()
            self.attn_weights.append(output)

        def conv_hook(module, input, output):
            self.conv_features.clear()
            self.conv_features.append(output)

        self.query_pos = query_pos
        self.col_embed = col_embed
        self.row_embed = row_embed

        self.backbone.layer3.register_forward_hook(conv_hook)
        self.encoder.layers[-1].self_attn.register_forward_hook(attn_hook)
        #self.encoder.norm.register_forward_hook(enc_hook)

        self.classification_head = nn.Sequential()
        self.classification_head.fc1 = nn.Linear(201617, 2048)
        self.classification_head.dropout1 = nn.Dropout(0.5)
        self.classification_head.fc2 = nn.Linear(2048, 1024)
        self.classification_head.dropout2 = nn.Dropout(0.5)
        self.classification_head.fc3 = nn.Linear(1024, num_classes)


    def forward(self, input):
        x = self.backbone.conv1(input)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        #x = self.backbone.layer4(x) 
        #print(x.shape)

        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
            
            #print(pos.shape)

            # propagate through the transformer
        h = self.encoder(pos + 0.1 * h.flatten(2).permute(2, 0, 1)).transpose(0, 1)#,
                                #self.query_pos.unsqueeze(1)).transpose(0, 1)
            #print(h.shape)
            #preparing the outputs
        f_map = self.conv_features[0]
        print(f_map.shape)
        shape = f_map.shape[-2:]
        print(shape)
        sattn = self.attn_weights[0][1][0].reshape(shape + shape)

        shape2 = f_map.shape[-1:]
        #print(shape2[0])
        atn_by_pixel = self.attn_weights[0][1].reshape(f_map.shape[0],shape2[0],shape2[0],shape2[0]*shape2[0])
        atn_by_pixel = atn_by_pixel.permute(0, 3, 1, 2)
        #print(atn_by_pixel.shape)
        f_map = torch.cat((f_map, atn_by_pixel), dim=1)

        #output, attn, f_map = self.encoder(x)
        #print(f_map.shape)
        #print(h.shape)
        #implementar if dependendo da architecture
        #x = h.reshape(-1, 169*256)
        print(self.size)
        print(f_map.shape)
        x = f_map.view(-1, 201617)#(self.size*self.size+2048)*self.size*self.size)
        x = F.relu(self.classification_head.fc1(x))
        x = F.relu(self.classification_head.dropout1(x))
        x = F.relu(self.classification_head.fc2(x))
        x = F.relu(self.classification_head.dropout2(x))
        x = self.classification_head.fc3(x)
        return x, sattn, f_map