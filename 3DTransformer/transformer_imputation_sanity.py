import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
from transformer_imputation_helper import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=int,help='device')
parser.add_argument('-l', '--log-file', type=str,help='log_file')
parser.add_argument('-o', '--out-dir', type=str,help='dir to save checkpoints')
parser.add_argument('-t', '--test', action='store_true',help='test')
parser.add_argument('-w', '--warm-start', action='store_true',help='warm_start')
parser.add_argument('-f', '--test-file', type=str,help='test-file')
parser.add_argument('-e', '--start-epoch', type=int,help='checkpoint')
parser.add_argument('-n', '--exp-name', type=str,help='std')
args = parser.parse_args()


exp_name = args.exp_name
log_file = 'transformer_sanity_%s'%exp_name
data_file = 'dataset/1d_sanity_complete_%s.npy'%exp_name


args.device = 0
args.out_dir = '/mnt/infonas/blossom/pbansal/dump'
device = torch.device('cuda:%d'%args.device)
batch_size = 48
lr = 1e-3

train_set = TransformerDataset1D_(data_file,'dataset/1d_sanity_train_examples.npy')
val_set3 = TransformerDataset1D_(data_file,'dataset/1d_sanity_test_examples.npy')

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate1d)
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate1d)

model = TransformerModel(ninp=16, nhead=2, nhid=100, nlayers=2).to(device)
optim = torch.optim.Adam(model.parameters(),lr=lr)
writer = SummaryWriter(os.path.join('runs',log_file))

max_epoch = 1000
iteration = 0
start_epoch = 0

for epoch in range(start_epoch,max_epoch):
    print ("Starting Epoch : %d"%epoch)

    for inp_,out_ in train_loader :
        y_pred = model(inp_.to(device))
        loss = model.calc_loss(out_.to(device),y_pred) 
        optim.zero_grad()
        loss.backward()
        optim.step()
        iteration += 1
        writer.add_scalar('training/time_series_loss',loss,iteration)

    
    torch.save(model.state_dict(), os.path.join(args.out_dir,'checkpoint_%d'%epoch))
    if ((epoch+1) % 1 == 0):
        loss_ = 0
        with torch.no_grad():
            for inp_,out_ in val_loader3 :
                y_pred = model(inp_.to(device))
                loss_ += model.calc_loss(out_.to(device),y_pred)*inp_.shape[0]
            writer.add_scalar('validation/time_series_loss',loss_/len(val_set3),iteration)
            
        
