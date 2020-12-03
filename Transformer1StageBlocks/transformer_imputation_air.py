import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os,copy
import matplotlib.pyplot as plt
import _pickle as cPickle
from transformer_imputation_helper import *

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=int,help='device')
parser.add_argument('-l', '--log-file', type=str,help='log_file')
parser.add_argument('-o', '--out-dir', type=str,help='dir to save checkpoints')
parser.add_argument('-t', '--test', action='store_true',help='test')
parser.add_argument('-w', '--warm-start', action='store_true',help='warm_start')
parser.add_argument('-f', '--test-file', type=str,help='test-file')
parser.add_argument('-e', '--start-epoch', type=int,help='checkpoint')
args = parser.parse_args()


log_file = 'transformer_air_mae_1stage_36len'

args.device = 0
args.out_dir = '/mnt/infonas/blossom/pbansal/dump'
device = torch.device('cuda:%d'%args.device)
batch_size = 128
lr = 1e-3

train_set = TransformerDataset('../dataset/1dair_complete.npy','../dataset/1dair_train_examples.npy',time_context=18,mask=True)
val_set = TransformerDataset('../dataset/1dair_complete.npy','../dataset/1dair_test_examples.npy',time_context=18,mask=True)

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)
val_loader = torch.utils.data.DataLoader(val_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)

model = OurModel(sizes=[36],ninp=64,embedding_size=32,nhid=128,nlayers=4,nhead=4,kernel_size=5).to(device)

best_state_dict = model.state_dict()
best_loss = float('inf')
writer = SummaryWriter(os.path.join('../Transformer/runs',log_file))


print ("Starting Stage 1")

optim = torch.optim.Adam(model.parameters(),lr=lr)

max_epoch = 500
switch_epoch = 500
iteration = 0
start_epoch = 0
residuals = copy.deepcopy(train_set.feats)
print (residuals.shape)
residuals -= np.nanmean(residuals,axis=0)
residuals /= np.maximum(np.nanstd(residuals,axis=0),1e-1)
residuals = np.nan_to_num(residuals)
model.residuals = torch.from_numpy(residuals).to(device)

for epoch in range(start_epoch,max_epoch):
    print ("Starting Epoch : %d"%epoch)

    for inp_,out_,mask,context_info in train_loader :
        loss = model(inp_.to(device),out_.to(device),context_info)
        optim.zero_grad()
        #loss['nll'].backward()
        loss['mae'].backward()
        optim.step()
        iteration += 1
        writer.add_scalar('training/nll_loss',loss['nll'],iteration)
        writer.add_scalar('training/mae_loss',loss['mae'],iteration)
        writer.add_scalar('training/variance',loss['var'],iteration)
        
    if (epoch % 1 == 0):
        loss_mre_num,loss_mre_den,loss_crps = 0,0,0
        with torch.no_grad():
            for inp_,out_,mask,context_info in val_loader :
                loss = model.validate(inp_.to(device),out_.to(device),context_info)
                loss_mre_num += loss['mae']*inp_.shape[0]
                loss_mre_den += loss['sum']*inp_.shape[0]
                loss_crps += loss['crps']*inp_.shape[0]
            writer.add_scalar('validation/mre_loss',loss_mre_num/loss_mre_den,iteration)
            writer.add_scalar('validation/mae_loss',loss_mre_num/len(val_set),iteration)
            writer.add_scalar('validation/crps_loss',loss_crps/len(val_set),iteration)
            
        if (loss_crps < best_loss):
            best_loss = loss_crps
            best_state_dict = model.state_dict()
            
    print ('done validation')
    
    
