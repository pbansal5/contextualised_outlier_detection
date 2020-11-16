import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os,copy
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
args = parser.parse_args()


#log_file = 'transformer_janta'
log_file = 'ablation_janta_rbf_try'

args.device = 0
args.out_dir = '/mnt/infonas/blossom/pbansal/dump'
device = torch.device('cuda:%d'%args.device)
batch_size = 64
lr = 1e-4

train_set = TransformerDataset2D_('dataset/2djantahack_complete.npy','dataset/2djantahack_train_examples.npy')
val_set3 = TransformerDataset2D_('dataset/2djantahack_complete.npy','dataset/2djantahack_test_examples.npy')
update_set = TransformerDataset2D_('dataset/2djantahack_complete.npy','dataset/2djantahack_all_examples.npy')

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate2d)
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate2d)
update_loader = torch.utils.data.DataLoader(update_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate2d)

writer = SummaryWriter(os.path.join('runs',log_file))



print ("Starting Stage 2")

model_outlier = OutlierModel(size1=76,size2=28,embedding_size=32)
residuals = torch.zeros(update_set.feats.shape).to(device)
means = torch.zeros(update_set.feats.shape).to(device)
stds = torch.zeros(update_set.feats.shape).to(device)

with torch.no_grad():
        for inp_,out_,context_info in update_loader :
            y_pred = 0
            sigma_pred =2*torch.log(torch.from_numpy(update_set.stds[context_info['index1'],context_info['index2']])).to(device)
            temp = (out_.to(device)-y_pred)/torch.exp(sigma_pred/2)
            residuals[context_info['time'],context_info['index1'],context_info['index2']] = temp
            means[context_info['time'],context_info['index1'],context_info['index2']] = y_pred
            stds[context_info['time'],context_info['index1'],context_info['index2']] = sigma_pred
        
model_outlier.residuals = residuals
model_outlier.means = means
model_outlier.stds = stds

model_outlier.to(device)
optim = torch.optim.Adam(model_outlier.parameters(),lr=lr)

max_epoch = 200
start_epoch = 0
iteration = 0

for epoch in range(start_epoch,max_epoch):
    print ("Starting Epoch : %d"%epoch)

    if ((epoch+1) % 1 == 0):
        loss_ = 0
        with torch.no_grad():
            for inp_,out_,context_info in val_loader3 :
                y_pred,_ = model_outlier(context_info)
                loss_ += model_outlier.calc_loss_org_mse(out_.to(device),y_pred,val_set3.stds[context_info['index1'],context_info['index2']])
                model_outlier.add_calc_loss_org_mre(out_.to(device),y_pred,val_set3.stds[context_info['index1'],context_info['index2']])
            writer.add_scalar('validation/outlier_mre_loss',model_outlier.compute_calc_loss_org_mre(),iteration)
            writer.add_scalar('validation/outlier_mse_loss',loss_/len(val_set3),iteration)
        
    for inp_,out_,context_info in train_loader :
        y_pred,sigma_pred = model_outlier(context_info)
        loss = model_outlier.calc_loss_mle(out_.to(device),y_pred,sigma_pred) 
        optim.zero_grad()
        loss.backward()
        optim.step()
        iteration += 1
        writer.add_scalar('training/outlier_mle_loss',loss,iteration)

