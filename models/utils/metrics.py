
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

from models.encoder import EncoderModule
from models.backbone import Backbone, NoBackbone
from models.utils.losses import *

from fastai.vision.all import *
from fastai.distributed import *
from fastai.metrics import *
from fastai.callback.tracker import SaveModelCallback

__all__ = ['Accuracy','Generator_Attention_loss','Critic_Attention_loss','Adversarial_loss','Reconstruction_Loss']

def Accuracy(preds,target): 

    _, pred = torch.max(preds[0], 1)

    return (pred == target).float().mean()


def Generator_Attention_loss(preds,target,beta=1): 

    LCA = Curating_of_attention_loss()
    Latt = beta*LCA(preds[2])

    return Latt

def Critic_Attention_loss(preds,target,beta=1): 

    LCA = Curating_of_attention_loss()
    Latt = -1*(beta*LCA(preds[2]))

    return Latt

def Adversarial_loss(preds,target,gamma=1): 

    crossEntropy = nn.CrossEntropyLoss()
    classLoss = -1*(gamma*crossEntropy(preds[0], target))

    return classLoss

def Reconstruction_Loss(preds,target,sigma=1):
    
    MSE = nn.MSELoss()
    Lrec = sigma*MSE(preds[4],preds[3])
    
    return Lrec


