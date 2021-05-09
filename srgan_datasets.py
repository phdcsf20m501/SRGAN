import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root_low,root_high=None, transforms_low=None, transforms_high=None, mode='train'):
        self.mode = mode
        if self.mode=='train':
            self.transforms_low = transforms.Compose(transforms_low)
            self.transforms_high = transforms.Compose(transforms_high)
            self.files_low = sorted(glob.glob(root_low))
            self.files_high = sorted(glob.glob(root_high))
        elif self.mode=='test':
            self.transforms_low = transforms.Compose(transforms_low)
            self.transforms_high = transforms.Compose(transforms_high)
            self.files_low = sorted(glob.glob(root_low))
            self.files_high_real = sorted(glob.glob(root_high))
    def __getitem__(self, index):
        if self.mode=='train':
            item_low =self.transforms_low(Image.open(self.files_low[index % len(self.files_low)]))
    
            item_high = self.transforms_high(Image.open(self.files_high[index % len(self.files_high)]))
    
            return {'low_res': item_low, 'high_res': item_high}
        elif self.mode=='test':
            item_low =self.transforms_low(Image.open(self.files_low[index % len(self.files_low)]))
            item_high = self.transforms_high(Image.open(self.files_high_real[index % len(self.files_high_real)]))
    
            return {'low_res': item_low, 'high_res_real': item_high}
    def __len__(self):
        if self.mode=='train':
            return max(len(self.files_low), len(self.files_high))
        elif self.mode=='test':
            return len(self.files_low)