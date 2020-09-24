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
parser.add_argument('-c', '--checkpoint', type=str,help='checkpoint')
args = parser.parse_args()

device = torch.device('cuda:%d'%args.device)
batch_size = 128
lr = 1e-4

mse_loss = torch.nn.MSELoss()


# train_set = TimeSeriesDataset_('dataset/nyc_taxi_unnorm_numpy.npy','dataset/nyc_taxi_train_non_zero_examples.npy')
# val_set3 = TimeSeriesDataset_('dataset/nyc_taxi_unnorm_numpy.npy','dataset/nyc_taxi_test_non_zero_examples.npy')
train_set = TimeSeriesDataset_('dataset/jantahack_nonorm_numpy_complete.npy','dataset/jantahack_train_examples.npy')
val_set3 = TimeSeriesDataset_('dataset/jantahack_nonorm_numpy_complete.npy','dataset/jantahack_test_examples.npy')


train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = time_series_collate)
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = time_series_collate)

model = TimeSeries().to(device)

if (args.warm_start):
    model.load_state_dict(torch.load(args.checkpoint))
optim = torch.optim.Adam(model.parameters(),lr=lr)

writer = SummaryWriter(os.path.join('runs',args.log_file))
max_epoch = 100
iteration = 0

for epoch in range(max_epoch):
    print ("Starting Epoch : %d"%epoch)
    for x_right,x_left,y in train_loader :
        y_pred = model(x_right.to(device),x_left.to(device))
        y = y.to(device)
        loss_ = mse_loss(y,y_pred)
        optim.zero_grad()
        loss_.backward()
        optim.step()
        iteration += 1
        writer.add_scalar('training/loss',loss_,iteration)
    print ("Done training")
    if (epoch % 1 == 0):
        torch.save(model.state_dict(), os.path.join(args.out_dir,'checkpoint_%d'%epoch))
        loss_ = 0
        for x_right,x_left,y in val_loader3 :
            with torch.no_grad() :
                y_pred = model(x_right.to(device),x_left.to(device))
                y = y.to(device)
                loss_ += mse_loss(y,y_pred).data*x_right.shape[0]
        loss_ = loss_/int(len(val_set3))
        writer.add_scalar('validation/truly_inliers_loss',loss_,iteration)
