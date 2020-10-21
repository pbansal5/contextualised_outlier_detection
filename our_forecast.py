import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
from helper_forecast import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=int,help='device')
parser.add_argument('-l', '--log-file', type=str,help='log_file')
parser.add_argument('-o', '--out-dir', type=str,help='dir to save checkpoints')
parser.add_argument('-t', '--test', action='store_true',help='test')
parser.add_argument('-w', '--warm-start', action='store_true',help='warm_start')
parser.add_argument('-f', '--test-file', type=str,help='test-file')
parser.add_argument('-e', '--start-epoch', type=int,help='checkpoint')
args = parser.parse_args()

device = torch.device('cuda:%d'%args.device)
batch_size = 64
lr = 1e-5

mse_loss = torch.nn.MSELoss()

useNormalised = True

train_set = CopulaDataset_('dataset/jantahack_forecast.npy',end_time=100,useNormalised=useNormalised)
val_set3 = ValidationCopulaDataset_('dataset/jantahack_forecast.npy',start_time = 100,useNormalised=useNormalised)

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = copula_collate)
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = 1,drop_last = False,shuffle=True)

model = ForecastModel().to(device)

optim = torch.optim.Adam(model.parameters(),lr=lr)

writer = SummaryWriter(os.path.join('runs',args.log_file))
max_epoch = 800
iteration = 0
start_epoch = 0

if (args.warm_start):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(os.path.join(args.out_dir,'checkpoint_%d'%(args.start_epoch-1)))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    iteration = int(args.start_epoch*len(train_set)/batch_size)
    start_epoch = args.start_epoch

for epoch in range(start_epoch,max_epoch):
    print ("Starting Epoch : %d"%epoch)

    for x,y in train_loader :
        loss = model(x.to(device),y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        iteration += 1
        writer.add_scalar('training/time_series_loss',loss,iteration)
    torch.save(model.state_dict(), os.path.join(args.out_dir,'checkpoint_%d'%epoch))

    if ((epoch+1) % 1 == 0):
        loss_ = 0
        with torch.no_grad() :
            for x,y in val_loader3:
                loss = model.infer(x.to(device),y.to(device))
                loss_ += loss
        loss_ = loss_/len(val_set3)
        writer.add_scalar('validation/time_series_loss',loss_,iteration)


