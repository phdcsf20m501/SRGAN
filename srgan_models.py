import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.Conv2d(in_features, in_features, 3, padding = 1),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_features, in_features, 3, padding =1),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
    


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=10):
        super(Generator, self).__init__()

        # pre-residual block       
        modelA = [   nn.Conv2d(input_nc, 64, 9, padding = 4),
                     nn.InstanceNorm2d(64),
                     nn.ReLU(inplace=True) ]
        
        in_features = 64 #64
        out_features = 64  #64
        
        #residual block
        for _ in range(n_residual_blocks):
            modelB = [ResidualBlock(in_features)]
        
        
        #post-residual block
        for _ in range(1):
            modelC = [ nn.Conv2d(in_features, out_features, 3, padding =1),
                       nn.InstanceNorm2d(out_features),
                       nn.ReLU(inplace=True) ]
    
        
        in_features = 64 #64
        out_features = 256  #64
        
        #upsamploing block
        modelD = [nn.Upsample(scale_factor=2), #128x128x64
                  nn.Conv2d(in_features, out_features, 3, padding =1),#128x128x256
                  nn.ReLU(inplace=True),
                  nn.Upsample(scale_factor=2),#256x256x256
                  nn.Conv2d(out_features, 3, 9, padding = 4), #256x256x3
                  nn.Tanh()]
        
        self.modelA = nn.Sequential(*modelA)
        self.modelB = nn.Sequential(*modelB)
        self.modelC = nn.Sequential(*modelC)
        self.modelD = nn.Sequential(*modelD)
        
    def forward(self, x):
        gen1 = self.modelA(x)
        res = self.modelB(gen1)
        gen2 = self.modelC(res)
        gen3 = gen1 + gen2
        gen4 = self.modelD(gen3)
        return gen4

        
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 3, stride=2, padding = 1), #input[1x3x256x256]
                 nn.InstanceNorm2d(64),
                 nn.LeakyReLU(0.2, inplace=True) ]

        model += [nn.Conv2d(64, 128, 3, stride=2, padding=1), #input[1x64x128x128]
                  nn.InstanceNorm2d(128), 
                  nn.LeakyReLU(0.2, inplace=True) ]

        model += [nn.Conv2d(128, 256, 3, stride=2, padding=1), #input[1x128x64x64]
                  nn.InstanceNorm2d(256), 
                  nn.LeakyReLU(0.2, inplace=True) ]

        model += [nn.Conv2d(256, 512, 3,stride = 2, padding=1),  #input[1x256x32x32],output[1x512x16x16]
                  nn.InstanceNorm2d(512), 
                  nn.LeakyReLU(0.2, inplace=True) ]
        
        #add average pooling here
        fc = [ nn.Linear(512*16*16, 1024),
                   nn.LeakyReLU(0.2),
                   nn.Linear(1024,1),
                   nn.Sigmoid()]

        
        self.model = nn.Sequential(*model)
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x =  self.model(x)
        #x = x.view(-1,512*16*16)
        x = x.view(512*16*16)
        x = self.fc(x)
        return x
        
