#python -m torch.distributed.launch --nproc_per_node={num_gpus} launch.py

from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.gan import *
from fastai.callback.tracker import SaveModelCallback
from fastprogress import fastprogress
import torch
import argparse
from models.utils.joiner2 import GAN
from models.utils.losses import *
from models.utils.metrics import Accuracy
from models.utils.misc import *
from models.unet import UNet

from torchvision import datasets, transforms, models
import torchvision.transforms as T


beta = -1e-6
gamma = 3e-3
sigma = 1.2

def Generator_Attention_loss(preds,target,beta=beta): 

    LCA = Curating_of_attention_loss()
    Latt = beta*LCA(preds[2])

    return Latt

def Critic_Attention_loss(preds,target,beta=beta): 

    LCA = Curating_of_attention_loss()
    Latt = -1*(beta*LCA(preds[2]))

    return Latt

def Adversarial_loss(preds,target,gamma=gamma): 

    crossEntropy = nn.CrossEntropyLoss()
    classLoss = -1*(gamma*crossEntropy(preds[0], target))

    return classLoss

def Reconstruction_Loss(preds,target,sigma=sigma):
    
    MSE = nn.MSELoss()
    Lrec = sigma*MSE(preds[4],preds[3])
    
    return Lrec

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

H = 320
W= 320
bs = 5
epochs = 10

seed = 1234

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

path = untar_data(URLs.IMAGENETTE_320)

transforms = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(),
                 get_y=parent_label,
                 item_tfms=Resize(H,W),
                 batch_tfms=transforms)

dloader = data.dataloaders(path,bs=bs,device='cuda') 

generator_loss = GeneratorLossWrapper(3)
critic_loss = CriticLossWrapper(2)

gan = GAN(num_encoder_layers = 4, nhead=4, backbone = True, num_classes = 10, bypass=False, hidden_dim=256, batch_size=bs, image_h=H, image_w=W,grid_l=4,penalty_factor="2")

#print("Number of Training Images:", len(dld.train)*bs)
#print("Number of Validation Images:", len(dld.valid)*bs)
#print("Batch Size:", bs)
critic_learn = Learner(dloader, gan, loss_func=critic_loss, metrics=[Accuracy, Critic_Attention_loss]).to_distributed(args.local_rank)
generator_learn = Learner(dloader, gan, loss_func=generator_loss, metrics=[Generator_Attention_loss, Adversarial_loss, Reconstruction_Loss]).to_distributed(args.local_rank)

for e in range(epochs):
    print("Epoch", e+1)
    #print("Generator training")
    #Generator Training
    for param in gan.generator.parameters():
        param.requires_grad = True
    for param in gan.model.parameters():
        param.requires_grad = False
    gan.noise_mode = True
    
    generator_learn.fit_one_cycle(1,0.0007)
    
    #print("Critit training without noised images")
    #Critit training without noised images
    for param in gan.generator.parameters():
        param.requires_grad = False
    fb = ["mask","penalty_mask","pos"]
    for name, p in gan.model.named_parameters(): 
        if name not in fb:
            p.requires_grad_(True)
    gan.noise_mode = False
    critic_learn.fit(1,2e-6)
    #print("Critit training with noised images")
    #Critit training with noised images
    gan.noise_mode = True
    critic_learn.fit(1,2e-6)

model_dir = Path.home()/'Luiz/saved_models'
#name pattern
#{dataset}_{loss}_{loss_coeficients}_{epochs}_{backbone}_{model}.pkl
critic_learn.export(model_dir/'IMAGENETTE_2_-1e6_3e3_1p2_2_Resnet101_GAN.pkl')