
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

from models.encoder import EncoderModule
from models.backbone import Backbone, NoBackbone
from models.unet import UNet

from fastai.vision.all import *
from fastai.distributed import *
from fastai.metrics import *
from fastai.callback.tracker import SaveModelCallback

__all__ = ['_adapted_set_freeze_model','_adapted_set_trainable','alt_critic','alt_generator','alt_before_batch','alt_do_one_batch','alt_accumulate']

def _adapted_set_freeze_model(m, rg):
    fb = ["mask","penalty_mask","pos"]
    for name, p in m.named_parameters(): 
        if name not in fb:
            p.requires_grad_(rg)
            
def _adapted_set_trainable(self):
    train_model = self.generator if     self.gen_mode else self.critic
    loss_model  = self.generator if not self.gen_mode else self.critic
    _adapted_set_freeze_model(train_model, True)
    _adapted_set_freeze_model(loss_model, False)
    if self.switch_eval:
        train_model.train()
        loss_model.eval()
        
        
def alt_critic(self, real_pred,input,target):
    "Create some `fake_pred` with the generator from `input` and compare them to `real_pred` in `self.crit_loss_func`."
    fake = self.gan_model.generator(input)#.requires_grad_(False)
    fake_pred = self.gan_model.critic(fake)
    self.crit_loss = self.crit_loss_func(real_pred, fake_pred,target)
    return self.crit_loss
#GANLoss.critic = alt_critic



def alt_generator(self, output, target, labels):
    "Evaluate the `output` with the critic then uses `self.gen_loss_func`"
    fake_pred = self.gan_model.critic(output)
    self.gen_loss = self.gen_loss_func(fake_pred, output, target)
    return self.gen_loss
#GANLoss.generator = alt_generator


def alt_before_batch(self):
    "Clamp the weights with `self.clip` if it's not None, set the correct input/target."
    if self.training and self.clip is not None:
        for p in self.critic.parameters(): p.data.clamp_(-self.clip, self.clip)
    if not self.gen_mode:
        (self.learn.xb,self.learn.yb) = (self.xb,self.yb)
#GANTrainer.before_batch = alt_before_batch


def alt_do_one_batch(self):
    self.pred = self.model(*self.xb)
    self('after_pred')
    if len(self.yb):
        self.loss_grad = self.loss_func(self.pred, *self.xb, *self.yb)
        self.loss = self.loss_grad.clone()
    self('after_loss')
    if not self.training or not len(self.yb): return
    self('before_backward')
    self.loss_grad.backward()
    self._with_events(self.opt.step, 'step', CancelStepException)
    self.opt.zero_grad()
#Learner._do_one_batch = alt_do_one_batch


def alt_accumulate(self, learn):
    self.count += 1
    #print(learn.loss.mean())
    #print(self.val)
    self.val = TensorCategory(torch.lerp(to_detach(TensorCategory(learn.loss.mean()), gather=False), self.val, self.beta))
#AvgSmoothLoss.accumulate = alt_accumulate
        
        
        






