import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
from helper_copula import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=int,help='device')
parser.add_argument('-l', '--log-file', type=str,help='log_file')
parser.add_argument('-o', '--out-dir', type=str,help='dir to save checkpoints')
parser.add_argument('-t', '--test', action='store_true',help='test')
parser.add_argument('-w', '--warm-start', action='store_true',help='warm_start')
parser.add_argument('-f', '--test-file', type=str,help='test-file')
parser.add_argument('-e', '--start-epoch', type=int,help='checkpoint')
args = parser.parse_args()

args.device = 0
args.out_dir = '/mnt/infonas/blossom/pbansal/dump'
device = torch.device('cuda:%d'%args.device)
batch_size = 64
lr = 1e-3


train_set = CopulaDataset_('dataset/sanity_complete.npy','dataset/sanity_train_examples.npy',k=3)
val_set3 = ValidationCopulaDataset_('dataset/sanity_complete.npy','dataset/sanity_test_examples.npy',rank=4)

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = copula_collate)
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = batch_size,drop_last = False,shuffle=True)

model = CopulaModel(hidden_dim=4,rank=4).to(device)
optim = torch.optim.Adam(model.parameters(),lr=lr)
writer = SummaryWriter(os.path.join('runs','sanity'))

max_epoch = 100
iteration = 0
start_epoch = 0
best_loss = 1000
best_sigma = 0

for epoch in range(start_epoch,max_epoch):
    print ("Starting Epoch : %d"%epoch)

    for x_right,x_left,y in train_loader :
        loss = model(x_right.to(device),x_left.to(device),y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        iteration += 1
        writer.add_scalar('training/time_series_loss',loss,iteration)

        #print (iteration,float(loss.data.cpu().numpy()))
    
    torch.save(model.state_dict(), os.path.join(args.out_dir,'checkpoint_%d'%epoch))
    if ((epoch+1) % 1 == 0):
        loss_ = 0
        with torch.no_grad() :
            this_sigma = val_set3.fill(model)
            for i,err in enumerate(val_loader3):
                loss_ += err.data.sum()
        loss_ = loss_/int(len(val_set3))
        #print ('Validation Loss : ', iteration,float(loss.data.cpu().numpy()))
        writer.add_scalar('validation/time_series_loss',loss_,iteration)
        if (loss_ < best_loss):
            best_loss = float(loss_.data.cpu().numpy())
            best_sigma = this_sigma
            
np.save('dataset/best_sigma.npy',best_sigma)
        
