import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
from helper import TimeSeries,TimeSeriesDataset_,time_series_collate

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
lr = 1e-4

mse_loss = torch.nn.MSELoss()

train_set = TimeSeriesDataset_('dataset/2djantahack_complete.npy','dataset/2djantahack_train_examples.npy')
val_set3 = TimeSeriesDataset_('dataset/2djantahack_complete.npy','dataset/2djantahack_test_examples.npy')

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = time_series_collate)
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = time_series_collate)

model = TimeSeries(hidden_dim=128).to(device)
optim = torch.optim.Adam(model.parameters(),lr=lr)

writer = SummaryWriter(os.path.join('runs',args.log_file))
max_epoch = 800
iteration = 0
start_epoch = 0

if (args.warm_start):
    model.load_state_dict(torch.load(os.path.join(args.out_dir,'checkpoint_%d'%(args.start_epoch-1))))
    iteration = int(args.start_epoch*len(train_set)/batch_size)
    start_epoch = args.start_epoch

for epoch in range(start_epoch,max_epoch):
    print ("Starting Epoch : %d"%epoch)
    for x_right,x_left,y,_ in train_loader :
        y_pred,var = model(x_right.to(device),x_left.to(device))
        y = y.to(device)
        loss_ = (((y_pred-y)**2)/torch.exp(var) + var).mean()
        optim.zero_grad()
        loss_.backward()
        optim.step()
        iteration += 1
        writer.add_scalar('training/loss',loss_,iteration)
        
    if (epoch % 1 == 0):
        torch.save(model.state_dict(), os.path.join(args.out_dir,'checkpoint_%d'%epoch))
        loss_ = 0
        with torch.no_grad() :
            for x_right,x_left,y,index_info in val_loader3 :
                y_pred,var = model(x_right.to(device),x_left.to(device))
                y = y.to(device)
                loss_ += (((y_pred-y)**2)/torch.exp(var) + var).mean().data*x_right.shape[0]
                y_pred = y_pred.squeeze().data.cpu().numpy()
                for i in range(x_right.shape[0]):
                    val_set3.predicted_values[index_info['time'][i]][index_info['index1'][i]][index_info['index2'][i]] = y_pred[i]
                    
        loss_ = loss_/int(len(val_set3))
        loss = val_set3.mse_loss()
        writer.add_scalar('validation/truly_inliers_loss',loss_,iteration)
        writer.add_scalar('validation/mse_loss',loss,iteration)
