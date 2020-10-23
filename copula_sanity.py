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
parser.add_argument('-n', '--exp-name', type=str,help='std')
args = parser.parse_args()


exp_name = args.exp_name
log_file = 'copula_sanity_%s'%exp_name
data_file = 'dataset/1d_sanity_complete_%s.npy'%exp_name


args.device = 0
args.out_dir = '/mnt/infonas/blossom/pbansal/dump'
device = torch.device('cuda:%d'%args.device)
batch_size = 32
lr = 1e-3


train_set = CopulaDataset1D_(data_file,'dataset/1d_sanity_train_examples.npy',k=3)
val_set3 = ValidationCopulaDataset1D_(data_file,'dataset/1d_sanity_test_examples.npy',rank=4)

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = copula_collate)
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = batch_size,drop_last = False,shuffle=True)

model = CopulaModel(hidden_dim=4,rank=4).to(device)
optim = torch.optim.Adam(model.parameters(),lr=lr)
writer = SummaryWriter(os.path.join('runs',log_file))

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
        val_set3.predicted_values = np.zeros(val_set3.predicted_values.shape)
        with torch.no_grad() :
            this_sigma = val_set3.fill(model)
            for i,(err,mean,time,index) in enumerate(val_loader3):
                loss_ += err.data.sum()
                for i in range(err.shape[0]):
                    val_set3.predicted_values[time[i]][index[i]] = mean[i]
                
        loss_ = loss_/int(len(val_set3))
        loss = val_set3.mse_loss()
        
        #print ('Validation Loss : ', iteration,float(loss.data.cpu().numpy()))
        writer.add_scalar('validation/time_series_loss',loss_,iteration)
        writer.add_scalar('validation/mse_loss',loss,iteration)
        if (loss < best_loss):
            best_loss = loss
            best_sigma = this_sigma
            predictions = val_set3.predicted_values
            
np.save('dataset/best_sigma_copula_%s.npy'%exp_name,best_sigma)
np.save('dataset/best_predictions_copula_%s.npy'%exp_name,predictions)
        
