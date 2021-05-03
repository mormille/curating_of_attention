from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import torchvision.transforms as T
import torch

from PIL import Image
import requests

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn

#from .models.backbone import Backbone
#from .models.encoder import EncoderModule
#from .models.joiner import Joiner
#from models.losses import Attention_penalty_factor, Generator_loss
#from .models.unet import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def data_transform(H,W, normalize:bool=True):
    if normalize == True:
        transform = T.Compose([
        T.Resize((H,W)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform
    else:
        transform = T.Compose([
        T.Resize((H,W)),
        T.ToTensor()
        ])
        return transform



def train_model(model, generator, inputs, labels, opt_d, model_criterion):
    
    
    # Clear discriminator gradients
    opt_d.zero_grad()
    
    real_inputs = inputs.to(device) # allow gpu use
    labels = labels.to(device) # allow gpu use
    # Pass real images through model
    real_preds, sattn, pattn = model(real_inputs)
    real_loss = model_criterion(real_preds, labels)
    real_score = torch.mean(real_preds).item()
    
    _, preds = torch.max(real_preds, 1) # return the index of the maximum value predicted for that image (used to generate the accuracy)
     # the sum of the loss of all itens
    running_corrects = torch.sum(preds == labels.data) # the sum of correct prediction on an epochs
    #print("TRAIN MODEL FUNCTION - RUNNING CORRECTS",running_corrects)
    
    # Generate fake images
    noised_inputs = generator(real_inputs)

    # Pass fake images through discriminator
    noised_preds, noised_sattn, noised_pattn = model(noised_inputs)
    noised_loss = model_criterion(noised_preds, labels)
    noised_score = torch.mean(noised_preds).item()
    
    _, preds_noised = torch.max(noised_preds, 1)
    running_noised_corrects = torch.sum(preds_noised == labels.data)
    #print("TRAIN MODEL FUNCTION - RUNNING NOISED CORRECTS",running_noised_corrects)

    # Update discriminator weights
    loss = real_loss + noised_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, noised_score, running_corrects, running_noised_corrects, sattn, noised_sattn

def train_generator(generator, model, inputs, labels, opt_g, gen_criterion, model_criterion):
    # Clear generator gradients
    opt_g.zero_grad()
    
    real_inputs = inputs.to(device) # allow gpu use
    labels = labels.to(device) # allow gpu use
    
    # Generate fake images
    noised_inputs = generator(real_inputs)
    
    # Try to fool the discriminator
    preds, sattn, pattn = model(noised_inputs)
    model_loss = model_criterion(preds, labels)
    loss = gen_criterion(pattn, noised_inputs, real_inputs, model_loss)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()

def fit_adversarial(training_loader, validation_loader, model, generator, model_criterion, gen_criterion, model_optimizer, 
    gen_optimizer, model_lr_scheduler, generator_lr_scheduler, len_train, len_val, 
    path='test/TOP_PERFORMANCE_epoch_', epochs=25, start_idx=1):
    
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    noise_scores = []
    real_corrects = []
    noised_corrects = []
    val_real_corrects = []
    val_noised_corrects = []
    train_real_attention_maps = []
    train_adversarial_attention_maps = []
    val_real_attention_maps = []
    val_adversarial_attention_maps = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        
        running_loss = 0.0
        running_real_corrects = 0.0
        running_noised_corrects = 0.0
        val_running_loss = 0.0
        val_running_real_corrects = 0.0
        val_running_noised_corrects = 0.0

        
        for inputs, labels in training_loader:
            # Train discriminator
            loss_d, real_score, noise_score, real_pred, noised_pred, sattn, noised_sattn = train_model(model, generator, inputs, labels, model_optimizer, model_criterion)
            running_real_corrects +=real_pred
            running_noised_corrects +=noised_pred
            #print("FIT FUNCTION - RUNNING REAL CORRECTS",running_real_corrects.item())
            #print("FIT FUNCTION - RUNNING NOISED CORRECTS",running_noised_corrects.item())
            # Train generator
            loss_g = train_generator(generator, model, inputs, labels, gen_optimizer, gen_criterion, model_criterion)
        

        else:
        #VALIDATION
            with torch.no_grad(): # to save memory (temporalely set all the requires grad to be false)
                for val_inputs, val_labels in validation_loader:
                    val_inputs = val_inputs.to(device) # allow gpu use
                    val_labels = val_labels.to(device) # allow gpu use
                    val_noised_input = generator(val_inputs) #passes the image through the network and get the output
                    val_preds_real, val_real_sattn, val_real_pattn = model(val_inputs)
                    val_preds_noised, val_noised_sattn, val_noised_pattn = model(val_noised_input)
                    val_real_loss = model_criterion(val_preds_real, val_labels) #compare output and labels to get the loss 
                    val_noised_loss = model_criterion(val_preds_noised, val_labels)
                    
                    _, val_real_preds = torch.max(val_preds_real, 1) #same as for training
                    _, val_noised_preds = torch.max(val_preds_noised, 1) #same as for training
                    
                    val_loss = val_real_loss + val_noised_loss
                    val_running_real_corrects += torch.sum(val_real_preds == val_labels.data) #same as for training
                    val_running_noised_corrects += torch.sum(val_noised_preds == val_labels.data) #same as for training

                    val_generator_loss = gen_criterion(val_noised_pattn,val_noised_input,val_inputs,val_noised_loss)
        
        
        model_lr_scheduler.step()
        generator_lr_scheduler.step()

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        noise_scores.append(noise_score)
        real_corrects.append(running_real_corrects)
        noised_corrects.append(running_noised_corrects)
        val_real_corrects.append(val_running_real_corrects)
        val_noised_corrects.append(val_running_noised_corrects)
        train_real_attention_maps.append(sattn)
        train_adversarial_attention_maps.append(noised_sattn)
        val_real_attention_maps.append(val_real_sattn)
        val_adversarial_attention_maps.append(val_noised_sattn)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, noised_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, noise_score))
    
    
        # Model accuracy       
        print("EPOCH RUNNING REAL CORRECT PREDICTIONS",running_real_corrects.item())
        print("EPOCH RUNNING NOISED CORRECTS PREDICTIONS",running_noised_corrects.item())
        epoch_real_acc = running_real_corrects.item()/ len_train
        epoch_noised_acc = running_noised_corrects.item()/ len_train
        print("Epoch [{}/{}], Training - real acc: {:.4f}, noised acc: {:.4f}".format(
            epoch+1, epochs, epoch_real_acc, epoch_noised_acc))
        
        val_epoch_real_acc = val_running_real_corrects.float()/ len_val
        val_epoch_noised_acc = val_running_noised_corrects.float()/ len_val
        print("Epoch [{}/{}], Validation - real acc: {:.4f}, noised acc: {:.4f}".format(
            epoch+1, epochs, val_epoch_real_acc, val_epoch_noised_acc))
              
        # Save generated images
        #save_samples(epoch+start_idx, fixed_latent, show=False)
        
        
        epoch_time_elapsed = time.time() - start_time
        print('Epoch training complete in {:.0f}m {:.0f}s'.format(
            epoch_time_elapsed // 60, epoch_time_elapsed % 60))

        if val_epoch_real_acc > best_acc:
                best_acc = val_epoch_real_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("TOP PERFORMANCE UPDATED")
                            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),path + str(epoch) + '.pth')
    
    return losses_d, losses_g, real_corrects, noised_corrects, val_real_corrects, val_noised_corrects,train_real_attention_maps, train_adversarial_attention_maps, val_real_attention_maps, val_adversarial_attention_maps

