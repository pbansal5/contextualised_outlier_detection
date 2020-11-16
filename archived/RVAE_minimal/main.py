#!/usr/bin/env python3

import sys
sys.path.append("/mnt/a99/d0/pbansal/RVAE_minimal/")

#import argparse
import json
import os, errno
import pandas as pd
import numpy as np
from RVAE import VAE
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from train_eval_models import training_phase
batch_size = 64
from helper import Dataset_

def main():
    dataset = Dataset_('/mnt/a99/d0/pbansal/dataset/nyc_taxi_norm_meadian_numpy_train.npy')
    train_loader = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size = batch_size,drop_last=True)
    val_set = Dataset_('/mnt/a99/d0/pbansal/dataset/nyc_taxi_norm_meadian_numpy_test.npy')
    val_loader = torch.utils.data.DataLoader(val_set,shuffle=True,batch_size = batch_size,drop_last=True)
    model = VAE().cuda()
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3, weight_decay=0)  # excludes frozen params / layers
    writer = SummaryWriter(log_dir = 'runs/run1')
    step = 0
    maeloss = torch.nn.L1Loss()
    for epoch in range(1,50 + 1):
        print ("Starting Epoch : %d"%epoch)
        step = training_phase(model, optimizer, train_loader, step,writer)
        torch.save(model.state_dict(), os.path.join('/mnt/blossom/more/pbansal/rvae_nyc_checkpoints/','checkpoint_%d'%epoch))
        loss_ = 0
        for (x,_) in val_loader :
            with torch.no_grad():
                p_params,q_params,q_samples = model(x.cuda())
                loss_ += maeloss(x[:,-1].cuda(),p_params['x'][:,-1]).data
        loss_ = loss_/int(len(val_set)/batch_size)
        writer.add_scalar('validation/loss',loss_,step)

def test_ ():
    val_set = Dataset_('/mnt/a99/d0/pbansal/dataset/nyc_taxi_norm_meadian_numpy_test.npy')
    val_loader = torch.utils.data.DataLoader(val_set,shuffle=True,batch_size = batch_size,drop_last=True)
    model = VAE().cuda()
    maeloss = torch.nn.L1Loss()
    for epoch in range(1,51):
        model.load_state_dict(torch.load(os.path.join('/mnt/blossom/more/pbansal/rvae_nyc_checkpoints/','checkpoint_%d'%epoch)))
        loss_ = 0
        for (x,_) in val_loader :
            with torch.no_grad():
                p_params,q_params,q_samples = model(x.cuda())
                loss_ += maeloss(x[:,-1].cuda(),p_params['x'][:,-1]).data
        loss_ = loss_/int(len(val_set)/batch_size)
        print (epoch,loss_)
        
if __name__ == '__main__':
    main()
    #test_()
