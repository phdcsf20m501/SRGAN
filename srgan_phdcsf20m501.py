# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:15:51 2021

@author: E42
"""

import argparse
import itertools
import glob
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from srgan_datasets import ImageDataset
from srgan_models import Generator, Discriminator
from srgan_utils import LambdaLR
from srgan_utils import weights_init_normal

#%%
# =============================================================================
# import os
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))
# os.chdir("C:\\Users\\E42\\AnacondaPython\\DeepLearningAnaconda\\SRGAN")
# #os.chdir("C:\\Users\\E42\\AnacondaPython\\MachineLearningAnaconda\\SentimentAnalysis\\CNN-sentimentAnalysis")
# 
# =============================================================================
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--size_low', type=int, default=64, help='size of the data crop for low res images (squared assumed)')
parser.add_argument('--size_high', type=int, default=256, help='size of the data crop for high res images (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot_low', type=str, default='DIV2K_train_LR_bicubic/X16T/*.*', help='root directory of the dataset')
parser.add_argument('--dataroot_high', type=str, default='DIV2K_train_LR_bicubic/X4T/*.*', help='root directory of the dataset')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
opt = parser.parse_args()
print(opt)

#%% Dataset loader
transforms_low = [transforms.CenterCrop(opt.size_low),
                  transforms.ToTensor()]
transforms_high = [transforms.CenterCrop(opt.size_high),
                   transforms.ToTensor()]
dataloader = DataLoader(ImageDataset(opt.dataroot_low,opt.dataroot_high, transforms_low, transforms_high), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

#%% defining genrator and discriminator
gen = Generator(opt.input_nc, opt.output_nc)
disc = Discriminator(opt.input_nc)

gen.apply(weights_init_normal)
disc.apply(weights_init_normal)

#%% Defining optimizers and LR Schedulers
optimizer_G = torch.optim.Adam(gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(disc.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

#%%
#defining losses

target_one = Variable(torch.Tensor(opt.batchSize).fill_(1.0),requires_grad=False)
target_zero =Variable(torch.Tensor(opt.batchSize).fill_(0.0),requires_grad=False)
print(target_one)
print(target_zero)
# Lossess
criterion_GAN = torch.nn.MSELoss(reduction='sum')
criterion_Disc = torch.nn.BCEWithLogitsLoss()

#%%
input_low = torch.Tensor(opt.batchSize, opt.input_nc, opt.size_low, opt.size_low) # [1, 3, 64, 64]
input_high = torch.Tensor(opt.batchSize, opt.output_nc, opt.size_high, opt.size_high) # [1, 3, 256, 256]

#Loss plot
losses = []

for epoch in range(opt.epoch, opt.n_epochs):
    print("epoch %s:"%epoch)
    for i, batch in enumerate(dataloader):
        
        real_low_resImg = Variable(input_low.copy_(batch['low_res']))
        real_high_resImg = Variable(input_high.copy_(batch['high_res']))
               
                
        #Generator's loss computed
        #high resulution image generated from low resolution image
        gen_high_resImg = gen(real_low_resImg)
        gen_high_resImg_prob = disc(gen_high_resImg)
        
        optimizer_G.zero_grad()     
        loss_content = criterion_GAN(gen_high_resImg, real_high_resImg)/gen_high_resImg.size(0)
          
        loss_adversarial = criterion_Disc(gen_high_resImg_prob,target_one)
        loss_G = loss_content + loss_adversarial
        loss_G.backward()
        optimizer_G.step()

        
        #Discriminator's loss computed
        
        real_high_resImg_prob = disc(real_high_resImg)
        gen_high_resImg_prob = disc(gen_high_resImg.detach())
        optimizer_D.zero_grad()
        loss_real = criterion_Disc(real_high_resImg_prob,target_one)
        loss_fake = criterion_Disc(gen_high_resImg_prob,target_zero)
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()
        
        #TEST: storing high res  image generated on each epoch
        #test_gen_img = transforms.ToPILImage()(gen_high_resImg.squeeze(0))
        #test_gen_img.save("DIV2K_train_LR_bicubic/XT/sample%s.png"%epoch)
    
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    torch.save(gen.state_dict(), 'output/gen.pth')
    torch.save(disc.state_dict(), 'output/disc.pth')
    losses.append((loss_D.item(), loss_G.item()))
        

print("training is over")
    #imsave(args, kwds)
#%%Plot loss
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
plt.savefig('loss_plot.png')
