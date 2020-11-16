import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
from helper import *

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
batch_size = 128
lr = 1e-4

mse_loss = torch.nn.MSELoss()


train_set = Proposal1Dataset_('dataset/nyc_taxi_complete.npy','dataset/nyc_taxi_train_non_zero_examples.npy')
val_set3 = Proposal1Dataset_('dataset/nyc_taxi_complete.npy','dataset/nyc_taxi_test_non_zero_examples.npy')
update_set = Proposal1Dataset_('dataset/nyc_taxi_complete.npy','dataset/nyc_taxi_all_examples.npy')

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = proposal1_collate)
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = proposal1_collate)
update_loader = torch.utils.data.DataLoader(update_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = proposal1_collate)

model = Proposal1Model().to(device)
optim = torch.optim.Adam(model.parameters(),lr=lr)
model.embeddings1 = torch.load('saved_embeddings/pulocation.pt').weight
model.embeddings2 = torch.load('saved_embeddings/dolocation.pt').weight


writer = SummaryWriter(os.path.join('runs',args.log_file))
max_epoch = 400
iteration = 0
start_epoch = 0

if (args.warm_start):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(os.path.join(args.out_dir,'checkpoint_%d'%(args.start_epoch-1)))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    # model.load_state_dict(torch.load(os.path.join(args.out_dir,'checkpoint_%d'%(args.start_epoch-1))))
    iteration = int(args.start_epoch*len(train_set)/batch_size)
    start_epoch = args.start_epoch

for epoch in range(start_epoch,max_epoch):
    print ("Starting Epoch : %d"%epoch)
    
    for x_right,x_left,y,context_info in train_loader :

        loss1,loss2 = model(x_right.to(device),x_left.to(device),y.to(device),context_info)
        optim.zero_grad()
        (loss1+loss2).backward()
        optim.step()
        iteration += 1
        writer.add_scalar('training/time_series_loss',loss1,iteration)
        writer.add_scalar('training/outlier_loss',loss2,iteration)
    
    if (epoch % 1 == 0):
        torch.save(model.state_dict(), os.path.join(args.out_dir,'checkpoint_%d'%epoch))
        loss1_,loss2_ = 0,0
        for x_right,x_left,y,context_info in val_loader3:
            with torch.no_grad() :
                loss1,loss2 = model(x_right.to(device),x_left.to(device),y.to(device),context_info)
                loss1_ += loss1.data*x_right.shape[0]
                loss2_ += loss2.data*x_right.shape[0]
        loss1_ = loss1_/int(len(val_set3))
        loss2_ = loss2_/int(len(val_set3))
        writer.add_scalar('validation/time_series_loss',loss1_,iteration)
        writer.add_scalar('validation/outlier_loss',loss2_,iteration)
        train_set.update_residuals(model,update_loader,device)
        val_set3.residuals = train_set.residuals
