import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.encoder import EncoderModule
from models.backbone import Backbone, NoBackbone
from models.utils.losses import Attention_penalty_factor

from models.positional_encoding import PositionalEncodingSin
from models.backbone import Backbone
from models.encoder import EncoderModule
from models.unet import UNet

# from fastai.vision.all import *

__all__ = ['Joiner', 'create_mask', 'penalty_mask', 'pos_emb', 'GAN']


class GAN(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False, num_encoder_layers = 6, nhead=1, backbone = False, num_classes = 10, bypass=False, mask=None, pos_enc = "sin", batch_size=10, hidden_dim=256, image_h=32, image_w=32, grid_l=4, penalty_factor="1", alpha=1):
        super().__init__()
        self.generator = UNet(n_channels,n_classes,bilinear)
        self.model = Joiner(num_encoder_layers, nhead, backbone, num_classes, bypass, mask, pos_enc, batch_size, hidden_dim, image_h, image_w, grid_l, penalty_factor, alpha)
        self.noise_mode = True
        
    def switcher():
        if self.noise_mode == True:
            self.noise_mode = False
        else:
            self.noise_mode = True
        
    def forward(self, inputs):
        
        x0 = inputs
        if self.noise_mode == True:
            inputs = self.generator(inputs)
            
        outputs, sattn, pattn = self.model(inputs)
        
        return [outputs, sattn, pattn, x0, inputs]
        
        
class Joiner(nn.Module):
    def __init__(self, num_encoder_layers = 6, nhead=1, backbone = False, num_classes = 10, bypass=False, mask=None, pos_enc = "sin", batch_size=10, hidden_dim=256, image_h=32, image_w=32, grid_l=4, penalty_factor="1", alpha=1):
        super().__init__()
        
        self.bypass = bypass
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pos_enc = pos_enc
        if backbone == True:
            self.f_map_h = image_h//16 
            self.f_map_w = image_w//16
            self.backbone = Backbone(hidden_dim=hidden_dim)
        else:
            self.f_map_h = image_h 
            self.f_map_w = image_w
            self.backbone = nn.Conv2d(3, hidden_dim, 1)
    
        self.encoder = EncoderModule(d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers)

        if self.bypass == True:

            self.fc1 = nn.Linear(2*self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)

        else:

            self.fc1 = nn.Linear(self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)

            
        self.pos = nn.Parameter(pos_emb(batch_size, hidden_dim, self.f_map_h, self.f_map_w),requires_grad=False)
        self.mask = nn.Parameter(create_mask(batch_size, self.f_map_h, self.f_map_w),requires_grad=False)
        self.penalty_mask = nn.Parameter(penalty_mask(batch_size, self.f_map_w, self.f_map_h, grid_l, penalty_factor, alpha),requires_grad=False)
            
        # spatial positional encodings
        if self.pos_enc == "learned":
            self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
            self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
            
    def forward(self, inputs, mask=Optional[Tensor], pos=Optional[Tensor], penalty_mask=Optional[Tensor]):
        #print("Passing through the model")
        
        penalty_mask = self.penalty_mask
        mask = self.mask
        #print(inputs.shape)
        h = self.backbone(inputs)
        
        if self.pos_enc == "learned":
            # construct positional encodings
            H, W = h.shape[-2:]
            pos = torch.cat([
                self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ], dim=-1).flatten(0, 1).unsqueeze(1)
            
            att, sattn = self.encoder(src= 0.3*h, mask=mask, pos_embed=pos)

        else:
            pos = self.pos
            if pos.shape[0] != h.shape[0]:
                pos = pos[0:h.shape[0],...]
                mask = mask[0:h.shape[0],...]
                #print(pos.shape)
                #print(mask.shape)
                
            #print("Shape h:",h.shape)
            #print("Shape pos:",pos.shape)
            att, sattn = self.encoder(src= 0.4*h, mask=mask, pos_embed=pos)
            #xattn = sattn.reshape(sattn.shape[:-1] + h.shape[-2:])

        sattn = sattn.reshape(sattn.shape[:-2] + h.shape[-2:] + h.shape[-2:])

        sattn = sattn.permute(0,3,4,1,2)

        if penalty_mask.shape[0] != sattn.shape[0]:
            penalty_mask = penalty_mask[0:sattn.shape[0],...]
            #print(penalty_mask.shape)
        
        pattn = sattn*penalty_mask#.to(device)

        if self.bypass == True:

            x = torch.cat([h,att],1)

        else:

            x = att


        x = x.flatten(1)

        x = self.fc1(x)
        #x = self.fc2(x)


        return [x, sattn, pattn]


def create_mask(batch_size, f_map_h, f_map_w):
    mask = torch.zeros((batch_size, f_map_h, f_map_w), dtype=torch.bool)
    return mask

def penalty_mask(batch_size, f_map_w, f_map_h, grid_l=4, penalty_factor="1", alpha=1):
    dist_matrix = Attention_penalty_factor.distance_matrix(batch_size, f_map_w, f_map_h, grid_l)
    grids_matrix = Attention_penalty_factor.grids_matrix(batch_size, f_map_w, f_map_h, grid_l)
    pf_matrix = Attention_penalty_factor.penalty_factor(dist_matrix, penalty_factor, alpha)

    penalty_mask = Attention_penalty_factor.penalty_matrix(batch_size, f_map_w, f_map_h, grids_matrix, pf_matrix, grid_l)
    
    return penalty_mask

def pos_emb(batch_size, hidden_dim, f_map_h, f_map_w):
    pos = PositionalEncodingSin.positionalencoding2d(batch_size,hidden_dim, f_map_h, f_map_w)
    return pos

def irregular_batch_size(testMask,testBatch):  
    mask_one = testMask[0]
    bs = testBatch.shape[0]
    bMask = []
    for b in range(bs):
        bMask.append(mask_one)
    d = torch.Tensor(testBatch.shape)
    newMask = torch.cat(bMask, out=d)
    newMask = newMask.view(testBatch.shape)
    return newMask
    
    

