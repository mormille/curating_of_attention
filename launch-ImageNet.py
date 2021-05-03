#python -m torch.distributed.launch --nproc_per_node={num_gpus} launch-ImageNet.py

from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.tracker import SaveModelCallback
from fastprogress import fastprogress
import torch
import argparse
from models.utils.joiner2 import Joiner
from models.utils.losses import Attention_penalty_factor, Generator_loss, CriticLoss
from models.utils.metrics import Accuracy

from torchvision import datasets, transforms, models
import torchvision.transforms as T


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

H = 32
W= 32
bs = 5

path = Path.home()/'Luiz/gan_attention/data'

transforms = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(),
                 get_y=parent_label,
                 item_tfms=Resize(H,W),
                 batch_tfms=transforms)

dld = data.dataloaders(path/'train',bs=bs) 

model = Joiner(num_encoder_layers = 5, nhead=4, backbone=False, bypass=False, hidden_dim=256, batch_size=bs, image_h=H, image_w=W,grid_l=4,penalty_factor="1")


learn = Learner(dld, model, loss_func=CriticLoss(beta=0, sigma=1), metrics=[Accuracy]).to_distributed(args.local_rank)
learn.fit_one_cycle(1, 1e-6)

model_dir = Path.home()/'Luiz/saved_models'
learn.export(model_dir/'ImageNet01_model.pkl')