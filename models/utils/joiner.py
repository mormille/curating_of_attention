import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.encoder import EncoderModule
from models.backbone import Backbone, NoBackbone
from models.utils.losses import Attention_penalty_factor

from models.positional_encoding import PositionalEncodingSin

# from fastai.vision.all import *

__all__ = ['Joiner', 'create_mask', 'penalty_mask', 'pos_emb']

class Joiner(nn.Module):
    def __init__(self, encoder, backbone = False, num_classes = 10, bypass=False, mask=None, pos_enc = "sin", batch_size=10, hidden_dim=256, image_h=32, image_w=32, grid_l=4, penalty_factor="1", alpha=1):
        super(Joiner,self).__init__()
        
        self.bypass = bypass
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pos_enc = pos_enc
        if backbone != False:
            self.f_map_h = image_h//15 
            self.f_map_w = image_w//15
            self.backbone = backbone
        else:
            self.f_map_h = image_h 
            self.f_map_w = image_w
            self.backbone = nn.Conv2d(3, hidden_dim, 1)
    
        self.encoder = encoder

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
        
        penalty_mask = self.penalty_mask
        mask = self.mask
        h = self.backbone(inputs)
        
        if self.pos_enc == "learned":
            # construct positional encodings
            H, W = h.shape[-2:]
            pos = torch.cat([
                self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ], dim=-1).flatten(0, 1).unsqueeze(1)
            
            att, sattn = self.encoder(src= 0.4*h, mask=mask, pos_embed=pos)

        else:
            pos = self.pos
            att, sattn = self.encoder(src= 0.4*h, mask=mask, pos_embed=pos)
            #xattn = sattn.reshape(sattn.shape[:-1] + h.shape[-2:])

        sattn = sattn.reshape(sattn.shape[:-2] + h.shape[-2:] + h.shape[-2:])

        sattn = sattn.permute(0,3,4,1,2)

        pattn = sattn*penalty_mask#.to(device)

        if self.bypass == True:

            x = torch.cat([h,att],1)

        else:

            x = att


        x = x.flatten(1)

        x = self.fc1(x)
        #x = self.fc2(x)


        return x#, sattn, pattn


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


