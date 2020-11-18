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


log_file = 'transformer_taxi'

args.device = 0
args.out_dir = '/mnt/infonas/blossom/pbansal/dump'
device = torch.device('cuda:%d'%args.device)
batch_size = 64
lr = 1e-4

train_set = TransformerDataset('../dataset/2dnyc_taxi_complete.npy','../dataset/2dnyc_taxi_train_examples.npy')
val_set = TransformerDataset('../dataset/2dnyc_taxi_complete.npy','../dataset/2dnyc_taxi_test_examples.npy')
update_set = TransformerDataset('../dataset/2dnyc_taxi_complete.npy','../dataset/2dnyc_taxi_all_examples.npy')

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)
val_loader = torch.utils.data.DataLoader(val_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)
update_loader = torch.utils.data.DataLoader(update_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)

model = OurModel(sizes=[266,266],ninp=32,embedding_size=32,nhid=32,nlayers=4,nhead=2).to(device)

best_state_dict = model.state_dict()
best_loss = float('inf')
writer = SummaryWriter(os.path.join('runs',log_file))


print ("Starting Stage 1")

optim = torch.optim.Adam(model.parameters(),lr=lr)

max_epoch = 100
switch_epoch = 40
iteration = 0
start_epoch = 0

for epoch in range(start_epoch,max_epoch):
    print ("Starting Epoch : %d"%epoch)

    if (epoch == switch_epoch):
        print ("Starting Stage 2")
        model.load_state_dict(best_state_dict)
        residuals = torch.zeros(update_set.feats.shape).to(device)
        means = torch.zeros(update_set.feats.shape).to(device)
        stds = torch.zeros(update_set.feats.shape).to(device)
        with torch.no_grad():
                for inp_,context_info in update_loader :
                    out_,y_pred,sigma_pred = model.core(inp_.to(device),context_info)
                    temp = (out_.to(device)-y_pred)/torch.exp(sigma_pred/2)
                    residuals[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = temp
                    means[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = y_pred
                    stds[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = sigma_pred
        model.residuals = residuals
        model.means = means
        model.stds = stds
        model.switch()
        optim.zero_grad()
    if (epoch % 1 == 0):
        loss_mre_num,loss_mre_den,loss_crps = 0,0,0
        with torch.no_grad():
            for inp_,context_info in val_loader :
                loss = model.validate(inp_.to(device),context_info)
                loss_mre_num += loss['mae']*inp_.shape[0]
                loss_mre_den += loss['sum']*inp_.shape[0]
                loss_crps += loss['crps']*inp_.shape[0]
            writer.add_scalar('validation/mre_loss',loss_mre_num/loss_mre_den,iteration)
            writer.add_scalar('validation/mae_loss',loss_mre_num/len(val_set),iteration)
            writer.add_scalar('validation/crps_loss',loss_crps/len(val_set),iteration)
            
        if ((not model.stage2) and loss_crps < best_loss):
            best_loss = loss_crps
            best_state_dict = model.state_dict()
            
    print ('done validation')
    
    for inp_,context_info in train_loader :
        loss = model(inp_.to(device),context_info)
        optim.zero_grad()
        loss['nll'].backward()
        optim.step()
        iteration += 1
        writer.add_scalar('training/nll_loss',loss['nll'],iteration)
        writer.add_scalar('training/mae_loss',loss['mae'],iteration)




