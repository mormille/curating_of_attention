
import copy
from typing import Optional, List

from fastai.distributed import *
from fastai.vision.all import *

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

from models.encoder import EncoderModule
from models.backbone import Backbone, NoBackbone

__all__ = ['Attention_penalty_factor', 'Curating_of_attention_loss', 'Generator_loss', 'CriticLoss','Generator_loss3','CriticLoss2','GeneratorLossWrapper','CriticLossWrapper']

class Attention_penalty_factor(nn.Module):
    def __init__(self, ):
        super().__init__()

    def distance_matrix(batch_size, width, height, grid_l=3):
        
        w = width
        h = height
        qt_hor_grids = w//grid_l
        qt_ver_grids = h//grid_l
        qtd_grids = qt_hor_grids*qt_ver_grids
        c = 0
        grids = []
        for i in range(qtd_grids):
            hor_pos = i//qt_hor_grids
            ver_pos = c
            c = c+1
            grid = [hor_pos,ver_pos]
            grids.append(grid)
            if c == qt_ver_grids:
                c=0

        dist_grid = []
        for g in range(len(grids)):
            dist_pair_list = []
            for n in range(len(grids)):
                dist_pair_list.append(distance.cityblock(grids[g], grids[n]))
            dist_grid.append(dist_pair_list)
            
        dist_matrix = torch.tensor(np.array(dist_grid))

        return dist_matrix

    def grids_matrix(batch_size, width, height, grid_l=3):
        
        w = width
        h = height
        len_input_seq = h*w
        qt_hor_grids = w//grid_l
        qt_ver_grids = h//grid_l


        grid_list = []
        for i in range(h):
            row_grid_list = []
            preliminar_ver_grid = i//grid_l
            if preliminar_ver_grid != 0:
                preliminar_ver_grid = preliminar_ver_grid*qt_hor_grids
                
            for h in range(w):
                preliminar_grid = h//grid_l+preliminar_ver_grid
                row_grid_list.append(preliminar_grid)
                
            grid_list.append(row_grid_list)
        grid_matrix = torch.tensor(np.array(grid_list))
        
        return grid_matrix
    
    def penalty_factor(dist_matrix, penalty_factor="1", alpha=1):
        if penalty_factor == "1" or penalty_factor =="distraction":
            pf_matrix = (1/(dist_matrix+1))**alpha
            return pf_matrix
        if penalty_factor == "2" or penalty_factor =="misdirection":
            pf_matrix = alpha*((torch.max(dist_matrix)//2)-dist_matrix+0.1)**3
            return pf_matrix


    def penalty_matrix(batch_size, width, height, grid_matrix, pf_matrix, grid_l=3):
        
        w = width
        h = height

        qt_hor_grids = w//grid_l
        qt_ver_grids = h//grid_l
        qtd_grids = qt_hor_grids*qt_ver_grids

        penalty_mask = []
        for i in range(qtd_grids):
            ref_column = pf_matrix[i]
            p_matrix = grid_matrix.type(torch.FloatTensor)
            for j in range(1,len(ref_column)):
                #print(float(j))
                p_matrix[p_matrix==j]=float(ref_column[j])
            p_matrix[p_matrix==0]=float(ref_column[0])
            penalty_mask.append(p_matrix)

        penalty_enc = []
        for i in range(h):
            penalty_row = []
            for j in range(w):
                #print(grid_matrix[i,j])
                #print(penalty_mask[grid_matrix[i,j]].shape)
                penalty_row.append(penalty_mask[grid_matrix[i,j]])
                #print(len(penalty_row))
            generic_tensor = Tensor(h,w)
            penalty_row_tensor = torch.cat(penalty_row, out=generic_tensor)
            penalty_enc.append(penalty_row_tensor)
            #print(penalty_row_tensor.shape)
            #break

        b = torch.Tensor(h, w, h, w)
        c=torch.cat(penalty_enc, out=b)
        c = c.view(h, w, h, w)

        pep = []
        for b in range(batch_size):
            pep.append(c)

        d = torch.Tensor(batch_size, h, w, h, w)
        penalty_encoding_pattern = torch.cat(pep, out=d)
        penalty_encoding_pattern = penalty_encoding_pattern.view(batch_size, h, w, h, w)

        return penalty_encoding_pattern

    def penalty_mask(batch_size, width, height, grid_l=3, penalty_factor="1", alpha=1):
        dist_matrix = distance_matrix(batch_size, width, height, grid_l)
        grids_matrix = grids_matrix(batch_size, width, height, grid_l)
        pf_matrix = penalty_factor(dist_matrix, penalty_factor, alpha)

        mask = penalty_matrix(batch_size, width, height,grids_matrix,pf_matrix)
        #UNFINISHED
        return mask

class Curating_of_attention_loss(nn.Module):
    def __init__(self, ):
        super().__init__()
 
    def forward(self, pattn):
        #Computing the Attention Loss
        Latt = torch.sum(pattn)

        return Latt

class Generator_loss(nn.Module):
    def __init__(self, beta=0.00005, gamma=0.005,sigma=1):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma

    def forward(self, pattn, noised, original_image, model_loss):
        #Computing the Attention Loss
        LCA = Curating_of_attention_loss()
        Latt = LCA(pattn)

        MSE = nn.MSELoss()
        Lrec = MSE(noised,original_image)

        Lg = self.beta*Latt - self.gamma*model_loss + self.sigma*Lrec

        return Lg
    
class Generator_loss2(nn.Module):
    def __init__(self, beta=0.00005, gamma=0.005,sigma=1):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma

    def forward(self, noised, original_image):
        #Computing the Attention Loss
        #LCA = Curating_of_attention_loss()
        #Latt = LCA(pattn)

        MSE = nn.MSELoss()
        Lrec = MSE(noised,original_image)

        #Lg = self.beta*Latt - self.gamma*model_loss + self.sigma*Lrec

        return Lrec

class CriticLoss(nn.Module):
    def __init__(self, beta=1e-7, sigma=1):
        super(CriticLoss, self).__init__()
        self.beta = beta
        self.sigma = sigma

    def forward(self, preds, label):

        crossEntropy = nn.CrossEntropyLoss()
        classificationLoss = crossEntropy(preds[0], label)

        LCA = Curating_of_attention_loss()
        Latt = LCA(preds[2])
        
        Lc = self.sigma*classificationLoss - self.beta*Latt
        
        return Lc
    
    
    
class Generator_loss3(nn.Module):
    def __init__(self, beta=0.00005, gamma=0.005,sigma=1):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma

    def forward(self, output, target):

        LCA = Curating_of_attention_loss()
        Latt = LCA(output[2])

        crossEntropy = nn.CrossEntropyLoss()
        model_loss = crossEntropy(output[0],target)
        
        MSE = nn.MSELoss()
        Lrec = MSE(output[4],output[3])

        Lg = TensorCategory(self.beta*Latt) - TensorCategory(self.gamma*model_loss) + TensorCategory(self.sigma*Lrec)

        return Lg
    
    
class CriticLoss2(nn.Module):
    def __init__(self, beta=1e-7, sigma=1):
        super().__init__()
        self.beta = beta
        self.sigma = sigma

    def forward(self, realPreds, fakePreds, target):

        crossEntropy = nn.CrossEntropyLoss()
        realClassLoss = crossEntropy(realPreds[0], target)
        fakeClassLoss = crossEntropy(fakePreds[0], target)
        classificationLoss = realClassLoss + fakeClassLoss

        LCA = Curating_of_attention_loss()
        Latt = LCA(realPreds[2]) + LCA(fakePreds[2])
        
        Lc = self.sigma*classificationLoss - self.beta*Latt
        #print("Model Loss:", Lc)
        
        return Lc
    
    
class GeneratorLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(GeneratorLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, output, target):

        LCA = Curating_of_attention_loss()
        crossEntropy = nn.CrossEntropyLoss()
        MSE = nn.MSELoss()
        
        loss0 = TensorCategory(LCA(output[2]))
        loss1 = TensorCategory(crossEntropy(output[0],target))
        loss2 = TensorCategory(MSE(output[4],output[3]))

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2*loss2 + self.log_vars[2]
        
        return loss0+loss1+loss2

    
class CriticLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(CriticLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, output, target):
        
        LCA = Curating_of_attention_loss()
        crossEntropy = nn.CrossEntropyLoss()
        
        loss0 = LCA(output[2])
        loss1 = crossEntropy(output[0],target)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]
        
        return loss0+loss1