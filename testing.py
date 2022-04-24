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



def test_eva(G,E,D,epoch,val_loader,test_loader,device,opt):
    #data_path=data_path=PACK_PATH+"/normal_test"
    #test_loader = load_dataset(256, data_path, 1)
    
    G.eval()
    E.eval()
    D.eval()
    
    target_all_val = []
    rec_all_val = []
    z_score_val = []
   
    target_all_test = []
    rec_all_test = []
    z_score_test = []
    with torch.no_grad():
        
        
        for idx, (image, target) in enumerate(val_loader):
            image = image.to(device)
            target = target.to(device)
            target_all_val.append(target.data.cpu().numpy())
           
            score1= torch.sum((G(E(image))-image)**2,dim=(1,2,3))
            rec_all_val.append(score1.data.cpu().numpy())
            
            
            score4 = (torch.sum(E(image)**2,dim=1))
            z_score_val.append(score4.data.cpu().numpy())
            
        
        for idx, (image, target) in enumerate(test_loader):
            image = image.to(device)
            target = target.to(device)
            target_all_test.append(target.data.cpu().numpy())
           
            score1= torch.sum((G(E(image))-image)**2,dim=(1,2,3))
            rec_all_test.append(score1.data.cpu().numpy())
            
            score4 = (torch.sum(E(image)**2,dim=1))
            z_score_test.append(score4.data.cpu().numpy())
            
            
    target_all_val = np.concatenate(target_all_val,axis=0)
    rec_all_val = np.concatenate(rec_all_val,axis=0)
    z_score_val = np.concatenate(z_score_val,axis=0)
    
    target_all_test = np.concatenate(target_all_test,axis=0)
    rec_all_test = np.concatenate(rec_all_test,axis=0)
    z_score_test = np.concatenate(z_score_test,axis=0)
    
    
    gt_val = (target_all_val == opt.normal_digit).astype(int)
    auc_recon_val = roc_auc_score(gt_val,-1*rec_all_val) 
    auc_score_val = roc_auc_score(gt_val,-1*z_score_val)
    
    

    
    gt_test = (target_all_test == opt.normal_digit).astype(int)
    auc_recon_test = roc_auc_score(gt_test,-1*rec_all_test)
    auc_score_test = roc_auc_score(gt_test,-1*z_score_test)
  
    
    eva_dic = {}
    eva_dic['val_recon'] = auc_recon_val
    eva_dic['val_zs'] = auc_score_val
    eva_dic['test_recon'] = auc_recon_test
    eva_dic['test_zs'] = auc_score_test
    eva_dic['epoch'] = epoch
    return eva_dic
