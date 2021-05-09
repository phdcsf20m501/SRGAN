import os
import argparse
import itertools
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from srgan_datasets import ImageDataset
from srgan_models import Generator, Discriminator



parser = argparse.ArgumentParser()
parser.add_argument('--size_low', type=int, default=64, help='size of the data crop for low res images (squared assumed)')
parser.add_argument('--size_high', type=int, default=256, help='size of the data crop for low res images (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot_low', type=str, default='DIV2K_test_LR_bicubic/X16/*.*', help='root directory of the dataset')
parser.add_argument('--dataroot_high', type=str, default='DIV2K_test_LR_bicubic/X4T/*.*', help='root directory of the dataset')
parser.add_argument('--generator_stored', type=str, default='output/gen.pth', help='generator checkpoint file')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

#%% Dataset loader
transforms_low = [transforms.ToTensor()]
transforms_high = [transforms.ToTensor()]
dataloader = DataLoader(ImageDataset(opt.dataroot_low, opt.dataroot_high, transforms_low = transforms_low,transforms_high = transforms_high, mode='test'), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

#%% defining genrator and discriminator
gen = Generator(opt.input_nc, opt.output_nc)
gen.load_state_dict(torch.load(opt.generator_stored))
# Set model's test mode
gen.eval()

# Inputs & targets memory allocation


#%% Create output dirs if they don't exist
if not os.path.exists('output/results'):
    os.makedirs('output/results')

MeanPSNR = 0
for i, batch in enumerate(dataloader):
    # Set model input
    tempA = batch['low_res']
    #tempB = batch['high_res_real']
    input_low = torch.Tensor(opt.batchSize, opt.input_nc, tempA.size(2), tempA.size(3)) # [1, 3, 64, 64]
    #input_high = torch.Tensor(opt.batchSize, opt.input_nc, tempB.size(2), tempB.size(3))
    real_low_resImg = Variable(input_low.copy_(batch['low_res']))
    #real_high_resImg = Variable(input_high.copy_(batch['high_res_real']))
    gen_high_resImg = gen(real_low_resImg)
    
    #converting tensor into image and storing
    gen_high_resImgPIL = transforms.ToPILImage()(gen_high_resImg.squeeze(0))
    #gen_high_resImgPIL.save('output/results/1.png')

    gen_high_resImgPIL.save('output/results/%s.png'%str(i))

    #PSNR
    
# =============================================================================
#     diff = (real_high_resImg - gen_high_resImg)**2
#     sum = np.sum(diff)
#     mse = sum/(real_high_resImg.shape[0]*real_high_resImg.shape[1]*real_high_resImg.shape[2])  
# 
#     
#     maxval = torch.max(real_high_resImg)
# =============================================================================


