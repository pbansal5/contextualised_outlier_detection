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
parser.add_argument('-o', '--out-dir', type=str,help='dir to save checkpoints')
parser.add_argument('-t', '--test', action='store_true',help='test')
parser.add_argument('-w', '--warm-start', action='store_true',help='warm_start')
parser.add_argument('-f', '--test-file', type=str,help='test-file')
parser.add_argument('-e', '--start-epoch', type=int,help='checkpoint')
parser.add_argument('-n', '--expname', type=str,help='checkpoint')
parser.add_argument('-l', '--load-epoch', type=int,help='checkpoint')
args = parser.parse_args()


log_file = 'transformer_janta_%d_%s'%(args.load_epoch,args.expname)

args.device = 0
args.out_dir = '/mnt/infonas/blossom/pbansal/dump'
device = torch.device('cuda:%d'%args.device)
batch_size = 64
lr = 1e-4

train_set = TransformerDataset('../dataset/2djantahack_complete.npy','../dataset/2djantahack_train_examples.npy')
val_set = TransformerDataset('../dataset/2djantahack_complete.npy','../dataset/2djantahack_test_examples.npy')
update_set = TransformerDataset('../dataset/2djantahack_complete.npy','../dataset/2djantahack_all_examples.npy')

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)
val_loader = torch.utils.data.DataLoader(val_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)
update_loader = torch.utils.data.DataLoader(update_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)

if (args.expname == 'both' or args.expname == 'both2'):
    model = OurModel(sizes=[76,28],ninp=32,embedding_size=16,nhid=32,nlayers=4,nhead=2,other_residuals=True).to(device)
else :
    model = OurModel(sizes=[76,28],ninp=32,embedding_size=16,nhid=32,nlayers=4,nhead=2,other_residuals=False).to(device)
    
best_state_dict = model.state_dict()
best_loss = float('inf')
writer = SummaryWriter(os.path.join('runs',log_file))


print ("Starting Stage 1")

optim = torch.optim.Adam(model.parameters(),lr=lr)

max_epoch = 200
switch_epoch = args.load_epoch
iteration = int((args.load_epoch*len(train_set))/batch_size)
start_epoch = args.load_epoch

for epoch in range(start_epoch,max_epoch):
    print ("Starting Epoch : %d"%epoch)

    if (epoch == switch_epoch):
        print ("Starting Stage 2") 
        model_dict = model.state_dict()
        best_state_dict = torch.load('best_janta_%depochs'%args.load_epoch)
        best_state_dict = {k: v for k, v in best_state_dict.items() if k not in ['outlier_layer1.weight','outlier_layer1.bias'] }
        model_dict.update(best_state_dict)
        model.load_state_dict(model_dict)
        residuals = torch.zeros(update_set.feats.shape).to(device)
        other_residuals = torch.zeros(update_set.feats.shape).to(device)
        means = torch.zeros(update_set.feats.shape).to(device)
        stds = torch.zeros(update_set.feats.shape).to(device)
        with torch.no_grad():
                for inp_,context_info in update_loader :
                    out_,y_pred,sigma_pred = model.core(inp_.to(device),context_info)
                    temp = (out_.to(device)-y_pred)/torch.exp(sigma_pred/2)
                    if (args.expname == 'both'):
                        other_residuals[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = temp.to(device) # change here
                        residuals[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = out_.to(device) # change here
                    elif (args.expname == 'both2'):
                        other_residuals[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = (out_.to(device)-y_pred) # change here
                        residuals[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = out_.to(device) # change here
                    elif (args.expname == 'residuals'):
                        residuals[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = temp.to(device) # change here
                    elif (args.expname == 'residuals2'):
                        residuals[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = (out_.to(device)-y_pred) # change here
                    elif (args.expname == 'normalised'):
                        residuals[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = out_.to(device) # change here
                    else :
                        raise Exception
                    means[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = y_pred
                    stds[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]] = sigma_pred
        if (args.expname == 'both' or args.expname == 'both2'):
            model.other_residuals = other_residuals
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



