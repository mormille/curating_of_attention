from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import *

from collections import defaultdict, deque
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torch

from classy_vision.dataset import ClassyDataset
from classy_vision.tasks import ClassificationTask

from classy_vision.dataset import ClassyDataset, register_dataset
from classy_vision.dataset.transforms import ClassyTransform, build_transforms
from torchvision.datasets.imagenet import ImageNet

import random

__all__ = ['ImagenetRotation','ImagenetOverlapedRotation','ImagenetColorization']

class ImagenetRotation(datasets.ImageNet):
    def __init__(self, path, transforms, size, train=True):
        super().__init__(path)
        self.transforms = transforms
        self.size = size
        self.degrees = [0,90,180,270]
        self.labelDict = {0:0,90:1,180:2,270:3}
        self.indexes = self.create_idx()

    def create_idx(self):
        indexes = {}
        i = 0
        while len(indexes) < (self.size):
            r=random.randint(0,1281166)
            if r not in indexes.values() : 
                indexes[i] = r
                i +=1

        return indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        index = self.indexes[index]
        im, _ = super().__getitem__(index)
        dgr = random.choice(self.degrees)
        self.category = self.labelDict[dgr] 
        return rotate(self.transforms(im),dgr), self.category

class ImagenetOverlapedRotation(datasets.ImageNet):
    def __init__(self, path, transforms, size, train=True):
        super().__init__(path)
        self.transforms = transforms
        self.size = size
        self.degrees = [0,90,180,270]
        self.labelDict = {0:0,90:1,180:2,270:3}
        self.indexes = self.create_idx()

    def create_idx(self):
        indexes = {}
        idx_list = []
        i = 0
        while i < (self.size/4):
            r=random.randint(0,1281166)
            if r not in idx_list: 
                idx_list.append(r)
                i +=1
        d = 0    
        k = 0
        while d < 4:
            for v in idx_list:
                indexes[k]=v
                k+=1
            d+=1

        return indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        key = index
        index = self.indexes[index]
        im, _ = super().__getitem__(index)

        if key < self.size/4:
            dgr = 0
        elif key >= self.size/4 and key < self.size/2:
            dgr = 90
        elif key >= self.size/2 and key < 3*self.size/4:
            dgr = 180
        else:
            dgr = 270

        self.category = self.labelDict[dgr]
        
        return rotate(self.transforms(im),dgr), self.category

class ImagenetColorization(datasets.ImageNet):
    def __init__(self, path, transforms, size, train=True):
        super().__init__(path)
        self.transforms = transforms
        self.size = size
        self.indexes = self.create_idx()

    def create_idx(self):
        indexes = {}
        i = 0
        while len(indexes) < (self.size):
            r=random.randint(0,1281166)
            if r not in indexes.values(): 
                indexes[i] = r
                i +=1

        return indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        index = self.indexes[index]
        im, _ = super().__getitem__(index)
        grayscale = T.Compose([transforms.Grayscale(num_output_channels=3)])
        grayImg = grayscale(im)
        return self.transforms(grayImg), self.transforms(im)