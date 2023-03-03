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
    dataset1 = datasets.FashionMNIST(root ='data-fashion-mnist', train=True, download=True,transform=data_transform)
    data1 = dataset1.data
    target1 = dataset1.targets

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
        dataset1.data = torch.cat((data1_p,data1_n[randIdx[:abnormal_num]]),dim=0)
        dataset1.targets = torch.cat((target1_p,target1_n[randIdx[:abnormal_num]]),dim=0)
    train_pos = torch.utils.data.DataLoader(dataset1, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    
    dataset2 = datasets.FashionMNIST(root ='data-fashion-mnist', train=True, download=True,transform=data_transform)
    data2 = dataset2.data
    target2 = dataset2.targets
    if(opt.gamma_p==0):
        data2 = data2[target2!=opt.normal_digit]
        target2 = target2[target2!=opt.normal_digit]
    else:
        data2 = data1_n[randIdx[abnormal_num:]]
        target2 = target1_n[randIdx[abnormal_num:]]
        
        
    if(opt.k==1):
        data2 = data2[target2==opt.auxiliary_digit]
        target2 = target2[target2==opt.auxiliary_digit]
    else:
        anomaly_list = list(np.arange(0,10))
        anomaly_list.remove(opt.normal_digit)
        randIdx_list = np.arange(len(anomaly_list))
        np.random.shuffle(randIdx_list)
        if(opt.k==2):
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])] 
        elif(opt.k==3):
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]]) |(target2==anomaly_list[randIdx_list[2]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]])] 
        else:
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]]) |(target2==anomaly_list[randIdx_list[3]]) |(target2==anomaly_list[randIdx_list[4]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]]) |(target2==anomaly_list[randIdx_list[3]]) |(target2==anomaly_list[randIdx_list[4]])] 
    randIdx = np.arange(data2.shape[0])
    np.random.shuffle(randIdx)
    unlabeled_num = dataset1.data.shape[0]
    auxiliary_num = int((unlabeled_num*opt.gamma_l)/(1-opt.gamma_l))
    dataset2.data = data2[randIdx[:auxiliary_num]]
    dataset2.targets = np.array(target2)[randIdx[:auxiliary_num]]
    if(opt.gamma_l == 0.2):
        train_neg = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size//4, shuffle=True, drop_last = False,**kwargs)
    elif(opt.gamma_l == 0.05):
        train_neg = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size//19, shuffle=True, drop_last = False,**kwargs)
    else:
        train_neg = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size//9, shuffle=True, drop_last =False,**kwargs)
    
    
    
    dataset_val = datasets.FashionMNIST(root = 'data-fashion-mnist', train=False, download=True,transform=data_transform)

    data_val = dataset_val.data
    target_val = dataset_val.targets
    data_val_normal = data_val[target_val==opt.normal_digit]
    target_val_normal = target_val[target_val==opt.normal_digit]
    data_val_abnormal = data_val[target_val!=opt.normal_digit]
    target_val_abnormal = target_val[target_val!=opt.normal_digit]
    
    randIdx_normal = np.arange(data_val_normal.shape[0])
    randIdx_abnormal = np.arange(data_val_abnormal.shape[0])
    np.random.shuffle(randIdx_normal)
    np.random.shuffle(randIdx_abnormal)
    
    dataset_val.data=torch.cat((data_val_normal[randIdx_normal[:200]],data_val_abnormal[randIdx_abnormal[:1800]]),dim=0)
    dataset_val.targets = torch.cat((target_val_normal[randIdx_normal[:200]],target_val_abnormal[randIdx_abnormal[:1800]]),dim=0)
    
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    dataset_test = datasets.FashionMNIST('data-fashion-mnist', train=False, download=True,transform=data_transform)
    dataset_test.data=torch.cat((data_val_normal[randIdx_normal[200:]],data_val_abnormal[randIdx_abnormal[1800:]]),dim=0)
    dataset_test.targets = torch.cat((target_val_normal[randIdx_normal[200:]],target_val_abnormal[randIdx_abnormal[1800:]]),dim=0)
   


    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    return train_pos,train_neg,val_loader,test_loader
