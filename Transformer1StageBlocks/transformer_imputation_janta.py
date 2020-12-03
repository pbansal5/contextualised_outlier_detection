import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os,copy
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


log_file = 'transformer_janta_1stage_block_siblings_fast_by_bigbatch_highlr_noembedinconv'
#log_file = 'transformer_janta_dummy'

args.device = 0
args.out_dir = '/mnt/infonas/blossom/pbansal/dump'
device = torch.device('cuda:%d'%args.device)
batch_size = 256
lr = 1e-3

train_set = TransformerDataset('../dataset/2d_block_jantahack_train.npy','../dataset/2d_block_jantahack_train_examples.npy')
val_set = ValidationTransformerDataset('../dataset/2d_block_jantahack_train.npy','../dataset/2d_block_jantahack_test.npy','../dataset/2d_block_jantahack_test_examples.npy')

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)
val_loader = torch.utils.data.DataLoader(val_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)

residuals = copy.deepcopy(train_set.feats)
residuals -= np.nanmean(residuals,axis=0)
residuals /= np.maximum(np.nanstd(residuals,axis=0),1e-1)
residuals = torch.from_numpy(np.nan_to_num(residuals)).to(device)

model = OurModel(sizes=[76,28],nkernel=16,embedding_size=16,nhid=32,nlayers=4,nhead=2,residuals = residuals).to(device)
#model = OurModel(sizes=[76,28],nkernel=16,embedding_size=16,nhid=16,nlayers=2,nhead=1,residuals = residuals).to(device)
model = torch.jit.script(model)

best_state_dict = model.state_dict()
best_loss = float('inf')
writer = SummaryWriter(os.path.join('../Transformer/runs',log_file))


print ("Starting Stage 1")

optim = torch.optim.Adam(model.parameters(),lr=lr)

max_epoch = 200
iteration = 0
start_epoch = 0

for epoch in range(start_epoch,max_epoch):
    print ("Starting Epoch : %d"%epoch)

    if (epoch == 40):
        torch.save(best_state_dict,'best_janta_40epochs')
    if (epoch == 100):
        torch.save(best_state_dict,'best_janta_100epochs')

        
    for inp_,out_,mask,context_info in train_loader :
        loss = model(inp_.to(device),out_.to(device),mask.to(device),context_info)
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
                loss = model.validate(inp_.to(device),out_.to(device),mask.to(device),context_info)
                loss_mre_num += loss['mae']*inp_.shape[0]
                loss_mre_den += loss['sum']*inp_.shape[0]
                loss_crps += loss['crps']*inp_.shape[0]
            writer.add_scalar('validation/mre_loss',loss_mre_num/loss_mre_den,iteration)
            writer.add_scalar('validation/mae_loss',loss_mre_num/len(val_set),iteration)
            writer.add_scalar('validation/crps_loss',loss_crps/len(val_set),iteration)
            
        if (float(loss_crps) < best_loss):
            best_loss = loss_crps
            best_state_dict = model.state_dict()
            
    print ('done validation')
