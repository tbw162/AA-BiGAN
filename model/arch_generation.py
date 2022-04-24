# -*- coding: utf-8 -*-
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)    
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
       
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
       
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        

    def forward(self, z):
        z = z.reshape(-1,100,1,1)
        out = self.layer1(z)  
        
        out = self.layer5(out)       
     
               
        out = self.layer6(out)
        
        img = self.layer7(out)        
        
        
        
        return img
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)    
        )
        self.layer6 = nn.Linear(512*1*1, 100)
        
    def forward(self,x):

        x = x.view(x.size(0),3,32,32)
       
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
       
        x = self.layer5(x)
       
        x = x.view(x.size(0),-1)
        z = self.layer6(x)
        return z

class Discriminatorxz(nn.Module):
    def __init__(self):
        super(Discriminatorxz, self).__init__()

        self.img_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.img_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
 
        self.img_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.img_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.rnd_layer1 = nn.Linear(100, 512)


        self.layer1 = nn.Sequential(
            nn.Linear(256 + 512, 1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 1),
            
            )
        
    def forward(self, img, z):
        
        img_out = self.img_layer1(img)
        img_out = self.img_layer2(img_out)
        img_out = self.img_layer3(img_out)
        img_out = self.img_layer4(img_out)
       
        z = z.view(z.shape[0], -1)
        z_out = self.rnd_layer1(z)
        
        img_out = img_out.view(img.shape[0], -1)
        
        out = torch.cat([img_out, z_out], dim=1)
        
        out = self.layer1(out)

        feature = out
        feature = feature.view(feature.size()[0], -1)

        d_out = self.layer2(out)
        
        return d_out,feature
    
    

class Discriminatorxx(nn.Module):
    def __init__(self):
        super(Discriminatorxx, self).__init__()

        self.img_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.img_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
 
        self.img_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.img_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )


        self.img_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.img_layer6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
 
        self.img_layer7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.img_layer8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )




        self.layer1 = nn.Sequential(
            nn.Linear(64 * 2 * 2 * 2, 1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 1),
            
            )

    def forward(self, img1, img2):
        img_out1 = self.img_layer1(img1)
        img_out1 = self.img_layer2(img_out1)
        img_out1 = self.img_layer3(img_out1)
        img_out1 = self.img_layer4(img_out1)
     
        img_out1 = img_out1.view(-1, 64 * 2 * 2)
        
        #out = self.ca2(img2)*img2
        
        #out = self.sa2(out)*out
        
        img_out2 = self.img_layer5(img2)
        img_out2 = self.img_layer6(img_out2)
        img_out2 = self.img_layer7(img_out2)
        img_out2 = self.img_layer8(img_out2)
        img_out2 = img_out2.view(-1, 64 * 2 * 2)
        
        
        out = torch.cat([img_out1, img_out2], dim=1)
        out = self.layer1(out)
        
        feature = out
        feature = feature.view(feature.size()[0], -1)

        out = self.layer2(out)

        return out,feature


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        
        
        self.D2  = Discriminatorxz()
        self.D1  = Discriminatorxx()
        
    def forward(self, z1, z2, f):
        
        if(f=='xz'):
            return self.D2(z1,z2)

        elif(f=='xx'):
            return self.D1(z1,z2)