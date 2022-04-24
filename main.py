# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import tqdm
import copy
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch
import inspect
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from itertools import cycle
import warnings



a = 1
b = 0 
c = 0.75
bn = (a+b)/2
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--normal_digit", type=int, default=0, help="noraml class")
parser.add_argument("--auxiliary_digit", type=int, default=1, help="abnormal aviliable during training process")
parser.add_argument("--gpu", type=str, default='3', help="gpu_num")
parser.add_argument("--dataset", type=str, default='MNIST', help="choice of dataset(CIFAR,F-MNIST,MNIST)")
parser.add_argument("--dir", type=str, default='/summary//', help="save dir")
parser.add_argument("--name", type=str, default='result', help="file name")
parser.add_argument("--gamma_l", type=float, default=0.2, help="ratio of auxiliary data")
parser.add_argument("--gamma_p", type=float, default=0, help="ratio of pollution data")
parser.add_argument("--k", type=float, default=1, help="the number of categories of the anomalous data")

opt = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': False} if torch.cuda.is_available() else {}
if(opt.k<=1):
    seed = 12
else:
    seed = opt.auxiliary_digit
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
cuda = True if torch.cuda.is_available() else False

if(opt.dataset=='CIFAR'):
    import model.arch_cifar as arch
else:
    import model.arch_mnist as arch


adversarial_loss = torch.nn.MSELoss()


generator = arch.Generator()
discriminator = arch.Discriminator()
encoder = arch.Encoder()

if(opt.dataset == 'CIFAR'):
    from dataset.cifarset import create_loader 
    train_pos, train_neg, val_loader, test_loader = create_loader(opt,kwargs)
elif(opt.dataset == 'F-MNIST'):
    from dataset.fmnistset import create_loader
    train_pos, train_neg, val_loader, test_loader = create_loader(opt,kwargs)
else:
    from dataset.mnistset import create_loader
    train_pos, train_neg, val_loader, test_loader = create_loader(opt,kwargs)

if cuda:
    generator = generator.cuda('cuda')
    encoder = encoder.cuda('cuda')
    discriminator = discriminator.cuda('cuda')
    adversarial_loss = adversarial_loss.cuda('cuda')
    
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.000025, betas=(0.5,0.9))
optimizer_E = torch.optim.Adam(encoder.parameters(),lr=0.0001,betas=(0.5,0.9))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
StepLR_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.98)
StepLR_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.98)
StepLR_E = torch.optim.lr_scheduler.StepLR(optimizer_E, step_size=100, gamma=0.98)

from testing import test_eva
PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
auc_re = pd.DataFrame()
best_val_recon = 0
best_test_recon = 0
best_val_zs = 0
best_test_zs = 0
import time 

for epoch in range(opt.n_epochs):
    start = time.time()
    i = 0
    StepLR_G.step()
    StepLR_E.step()
    StepLR_D.step()
    dxx_list = []
    dxz_list = []
    for (batch_pos,batch_neg) in zip(train_pos,cycle(train_neg)):
        discriminator.train()
        generator.train()
        encoder.train()
        
    
        i+=1
        
        img_pos = batch_pos[0]
        img_neg = batch_neg[0]
        
        target_pos = batch_pos[1]
        target_neg = batch_neg[1]        
        optimizer_D.zero_grad()   
        valid = torch.ones([img_pos.size(0), 1])
        fake = torch.zeros([img_pos.size(0), 1])       
        img_pos = img_pos.to(device)  
        img_neg = img_neg.to(device)
        valid = valid.to(device)
        fake = fake.to(device)
    
        pos_imgs = img_pos.type(Tensor)
        neg_imgs = img_neg.type(Tensor)
    
        z_out_fake = Variable(Tensor(np.random.normal(0, 1, (img_pos.shape[0], opt.latent_dim))))
        z_out_fake = z_out_fake.to(device)
        
        
        
        
        img = torch.cat([img_pos,img_neg])
        z_out = encoder(img)
        z_out_real = z_out[:img_pos.shape[0]]
        z_out_neg = z_out[img_pos.shape[0]:]
        
        z = torch.cat([z_out_real,z_out_fake])
        gen = generator(z)
        gen_imgs_real = gen[:img_pos.shape[0]]
        gen_imgs_fake = gen[img_pos.shape[0]:]
        
        D_pos_xz = adversarial_loss(discriminator(pos_imgs,z_out_real,'xz')[0], a*valid)
        D_fake_xz = adversarial_loss(discriminator(gen_imgs_fake,z_out_fake,'xz')[0], b*valid)
        D_neg_xz = adversarial_loss(discriminator(neg_imgs,z_out_neg,'xz')[0],bn*(torch.ones([img_neg.size(0), 1])).to(device))
        
        
        d_loss_xz = D_pos_xz+D_fake_xz+D_neg_xz
       
        dxz_list.append(d_loss_xz.data.cpu().numpy())
        
        
        D_pos_xx = adversarial_loss(discriminator(img_pos,img_pos,'xx')[0], a*valid)
        D_fake_xx = adversarial_loss(discriminator(pos_imgs,gen_imgs_real,'xx')[0], b*valid)
        D_neg_xx = adversarial_loss(discriminator(neg_imgs,neg_imgs,'xx')[0],bn*(torch.ones([img_neg.size(0), 1])).to(device))
        
        
        d_loss_xx = D_pos_xx+D_fake_xx+D_neg_xx
        
        
        dxx_list.append(d_loss_xx.data.cpu().numpy())
        
        d_loss = d_loss_xz+d_loss_xx
        
        d_loss.backward(retain_graph=True)
        optimizer_D.step()
        
        
        
        cycle_loss = adversarial_loss(discriminator(img_pos,img_pos,'xx')[0],c*valid)+adversarial_loss(discriminator(img_pos,gen_imgs_real,'xx')[0],c*valid)+adversarial_loss(discriminator(neg_imgs,neg_imgs,'xx')[0],c*(torch.ones([img_neg.size(0), 1])).to(device))
       
        
       
        g_loss = adversarial_loss(discriminator(gen_imgs_fake,z_out_fake,'xz')[0], c*valid)+(1/3)*cycle_loss
  
            
        optimizer_G.zero_grad()
        g_loss.backward(retain_graph=True)
        optimizer_G.step()
        
        optimizer_E.zero_grad()
       
        
    
        
        e_loss = adversarial_loss(discriminator(pos_imgs,z_out_real,'xz')[0],c*valid)+ adversarial_loss(discriminator(neg_imgs,z_out_neg,'xz')[0], c*(torch.ones([img_neg.size(0), 1])).to(device))+(1/3)*cycle_loss
        
        e_loss.backward()
        optimizer_E.step()
        
        discriminator.eval()
        generator.eval()
        encoder.eval()
        recon_pos = torch.mean(torch.sum((generator(encoder(img_pos))-img_pos)**2,dim=(1,2,3)))
        
        recon_neg = torch.mean(torch.sum((generator(encoder(img_neg))-img_neg)**2,dim=(1,2,3)))
        
        print(
                "[Epoch %d/%d] [Batch %d/%d] [recon_pos:%.3f][reconneg:%.3f]"
                % (epoch, opt.n_epochs, i, len(train_pos), recon_pos.item(),recon_neg.item())
            )
   

    if((np.mean(dxx_list)<0.015 or np.mean(dxz_list)<0.015) and epoch>300):
        break
    eva_dic = test_eva(generator,encoder,discriminator,epoch,val_loader,test_loader,device,opt)
    auc_re = auc_re.append(eva_dic,ignore_index=True)
    end = time.time()
    time_epoch = end-start
    
    if(eva_dic['val_recon']>best_val_recon):
        best_test_recon = eva_dic['test_recon']
        best_val_recon = eva_dic['val_recon']
    if(eva_dic['val_zs']>best_val_zs):
        best_test_zs = eva_dic['test_zs']
        best_val_zs = eva_dic['val_zs']
    print(
                "[Epoch %d/%d] [val_recon:%.3f][test_recon:%.3f] [val_zs:%.3f][test_zs:%.3f] [best_recon:%.3f][best_zs:%.3f][epoch_time:%.3f]"
                % (epoch, opt.n_epochs,eva_dic['val_recon'],eva_dic['test_recon'],eva_dic['val_zs'],eva_dic['test_zs'],best_test_recon,best_test_zs,time_epoch)
            )
    
if not os.path.exists(PACK_PATH +opt.dir):
    os.makedirs(PACK_PATH+opt.dir)
auc_re.to_csv(PACK_PATH+opt.dir+opt.name+str(opt.normal_digit)+'vs'+str(opt.auxiliary_digit)+".csv")
    
