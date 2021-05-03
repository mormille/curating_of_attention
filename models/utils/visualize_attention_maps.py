from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50, resnet101
import torchvision.transforms as T
torch.set_grad_enabled(False);
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, models, transforms


transform = T.Compose([
T.Resize((400,400)),
T.ToTensor(),
T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def model_pass(im, model):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs, sattn, f_map = model(img)
    print("Encoder attention:      ", sattn.shape)
    print("Feature map:            ", f_map.shape)
    print("Output shape:           ", outputs.shape)
    return outputs, sattn, f_map

def visualize_attention_maps(im,div,sattn):
    # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
    img = transform(im).unsqueeze(0)
    fact = 32
    # let's select 4 reference points for visualization
    idxs = [(400//div, 600//div), (600//div, 440//div), (600//div, 1200//div), (440//div, 900//div),]

    # here we create the canvas
    fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    # and we add one plot per reference point
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, -1]),
        fig.add_subplot(gs[1, -1]),
    ]

    # for each one of the reference points, let's plot the self-attention
    # for that point
    for idx_o, ax in zip(idxs, axs):
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'self-attention{idx_o}')

    # and now let's add the central image, with the reference points as red circles
    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(im)
    for (y, x) in idxs:
        scale = im.height / img.shape[-1]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
        fcenter_ax.axis('off')