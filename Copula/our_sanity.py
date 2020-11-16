import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
from helper_our import *

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
log_file = 'exp_our_sanity_%s'%exp_name
data_file = 'dataset/1d_sanity_complete_%s.npy'%exp_name

args.device = 0
args.out_dir = '/mnt/infonas/blossom/pbansal/dump'
device = torch.device('cuda:%d'%args.device)
batch_size = 48
lr = 1e-3

model = Proposal1Model1D(size1=9,hidden_dim=4,embedding_size=4).to(device)
optim = torch.optim.Adam(model.parameters(),lr=lr)

train_set = Proposal1Dataset1D_(data_file,'dataset/1d_sanity_train_examples.npy')
val_set3 = Proposal1Dataset1D_(data_file,'dataset/1d_sanity_test_examples.npy')
update_set = Proposal1Dataset1D_(data_file,'dataset/1d_sanity_all_examples.npy')

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = proposal1_collate1d)
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = proposal1_collate1d)
update_loader = torch.utils.data.DataLoader(update_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = proposal1_collate1d)

writer = SummaryWriter(os.path.join('runs',log_file))
if (exp_name == 'special'):
    max_epoch = 500
else :
    max_epoch = 100
iteration = 0
start_epoch = 0
best_loss = float('inf')

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

    for x_right,x_left,y,context_info in train_loader :

        loss1,loss2,_ = model.detached_forward(x_right.to(device),x_left.to(device),y.to(device),context_info)
        optim.zero_grad()
        if (epoch > 300):
            (loss2).backward()
        else :
            loss1.backward()
        optim.step()
        iteration += 1
        writer.add_scalar('training/time_series_loss',loss1,iteration)
        writer.add_scalar('training/outlier_loss',loss2,iteration)
            
    
    if (epoch % 1 == 0):
        torch.save(model.state_dict(), os.path.join(args.out_dir,'checkpoint_%d'%epoch))
        loss1_,loss2_ = 0,0
        val_set3.predicted_values = np.zeros(val_set3.predicted_values.shape)
        for x_right,x_left,y,context_info in val_loader3:
            with torch.no_grad() :
                loss1,loss2,value = model.detached_forward(x_right.to(device),x_left.to(device),y.to(device),context_info,(epoch>300))
                loss1_ += loss1.data*x_right.shape[0]
                loss2_ += loss2.data*x_right.shape[0]
                for i in range(x_right.shape[0]):
                    val_set3.predicted_values[context_info['time'][i],context_info['index1'][i]] = value[i].data.cpu().numpy()
                    
        loss1_ = loss1_/int(len(val_set3))
        loss2_ = loss2_/int(len(val_set3))
        loss3 = val_set3.mse_loss()

        writer.add_scalar('validation/time_series_loss',loss1_,iteration)
        writer.add_scalar('validation/outlier_loss',loss2_,iteration)
        writer.add_scalar('validation/mse_loss',loss3,iteration)
        
        if (epoch > 300):
            train_set.update_residuals(model,update_loader,device)
            val_set3.residuals = train_set.residuals
        if (loss3 < best_loss):
            best_loss = loss3
            best_sigma = model.get_sigma()
            predictions = val_set3.predicted_values

            
np.save('dataset/best_sigma_our_%s.npy'%exp_name,best_sigma.data.cpu().numpy())
np.save('dataset/best_predictions_our_%s.npy'%exp_name,predictions)
        
