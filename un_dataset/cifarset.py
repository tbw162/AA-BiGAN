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



def create_loader(opt,kwargs):
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
    dataset1 = datasets.CIFAR10('data-cifar', train=True, download=True,transform=data_transform)
    data1 = dataset1.data
    target1 = np.array(dataset1.targets)
   
    if(opt.gamma_p==0):
        data1 = data1[target1==opt.normal_digit]
        target1 = target1[target1==opt.normal_digit]
        dataset1.data=data1
        dataset1.targets = np.array(target1)
    else:
        data1_p = data1[target1==opt.normal_digit]
        data1_n = data1[target1!=opt.normal_digit]
        target1_p = target1[target1==opt.normal_digit]
        target1_n = target1[target1!=opt.normal_digit]
        randIdx = np.arange(data1_n.shape[0])
        np.random.shuffle(randIdx)
        normal_num = data1_p.shape[0]
        abnormal_num = int((normal_num*opt.gamma_p)/(1-opt.gamma_p))
   
      
        dataset1.data = np.concatenate((data1_p,data1_n[randIdx[:abnormal_num]]),axis=0)
        dataset1.targets = np.concatenate((target1_p,target1_n[randIdx[:abnormal_num]]),axis=0)
    train_pos = torch.utils.data.DataLoader(dataset1, batch_size=opt.batch_size, shuffle=True, drop_last = True,**kwargs)
  
    
   
        
    
    dataset_val = datasets.CIFAR10('data-cifar', train=False, download=True,transform=data_transform)
    data_val = dataset_val.data
    target_val = np.array(dataset_val.targets)
    data_val_normal = data_val[target_val==opt.normal_digit]
    target_val_normal = target_val[target_val==opt.normal_digit]
    data_val_abnormal = data_val[target_val!=opt.normal_digit]
    target_val_abnormal = target_val[target_val!=opt.normal_digit]

    randIdx_normal = np.arange(data_val_normal.shape[0])
    randIdx_abnormal = np.arange(data_val_abnormal.shape[0])
    np.random.shuffle(randIdx_normal)
    np.random.shuffle(randIdx_abnormal)
    dataset_val.data=np.concatenate((data_val_normal[randIdx_normal[:200]],data_val_abnormal[randIdx_abnormal[:1800]]),axis=0)
    dataset_val.targets = np.concatenate((target_val_normal[randIdx_normal[:200]],target_val_abnormal[randIdx_abnormal[:1800]]),axis=0)

    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, drop_last = True,**kwargs)

    dataset_test = datasets.CIFAR10('data-cifar', train=False, download=True,transform=data_transform)
    dataset_test.data=np.concatenate((data_val_normal[randIdx_normal[200:]],data_val_abnormal[randIdx_abnormal[1800:]]),axis=0)
    dataset_test.targets = np.concatenate((target_val_normal[randIdx_normal[200:]],target_val_abnormal[randIdx_abnormal[1800:]]),axis=0)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=True, drop_last = True,**kwargs)
    return train_pos,val_loader,test_loader
