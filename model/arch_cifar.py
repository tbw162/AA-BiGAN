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
    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x

class Discriminatorxz(nn.Module):
    def __init__(self,rep_dim=128):
        super(Discriminatorxz, self).__init__()
        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.rnd_layer1 = nn.Linear(self.rep_dim, 512)
        self.layer1 = nn.Sequential(
            nn.Linear(128*4*4 + 512, 1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 1),
            
            )
    def forward(self, img, z):
        
        x = img.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        img_out = x.view(int(x.size(0)), -1)
        
        
       
        z = z.view(z.shape[0], -1)
        z_out = self.rnd_layer1(z)
        
        
        
        out = torch.cat([img_out, z_out], dim=1)
        
        out = self.layer1(out)

        feature = out
        feature = feature.view(feature.size()[0], -1)

        d_out = self.layer2(out)
        
        return d_out,feature
    
    

class Discriminatorxx(nn.Module):
    def __init__(self,rep_dim=128):
        super(Discriminatorxx, self).__init__()
        self.rep_dim = rep_dim
        
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        
        self.conv4 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d6 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        

        self.layer1 = nn.Sequential(
            nn.Linear(128 * 4 * 4 * 2, 1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 1),
            
        )

        
    def forward(self, img1, img2):
        x = img1.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        img_out1 = x.view(int(x.size(0)), -1)
        
        x = img2.view(-1, 3, 32, 32)
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))
        x = self.conv6(x)
        x = self.pool(F.leaky_relu(self.bn2d6(x)))
        img_out2 = x.view(int(x.size(0)), -1)
        
        
        
        out = torch.cat([img_out1, img_out2], dim=1)
        out = self.layer1(out)
        
        feature = out
        feature = feature.view(feature.size()[0], -1)

        d_out = self.layer2(out)
        
        return d_out,feature
    

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
