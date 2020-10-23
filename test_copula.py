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
args = parser.parse_args()

device = torch.device('cuda:%d'%args.device)
batch_size = 32
lr = 1e-3

mse_loss = torch.nn.MSELoss()

val_set3 = ValidationCopulaDataset1D_('dataset/1djantahack_complete.npy','dataset/1djantahack_test_examples.npy')
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = batch_size,drop_last = False,shuffle=True)


model = CopulaModel().to(device)
optim = torch.optim.Adam(model.parameters(),lr=lr)

for epoch in range(20):
    print ("Testing Epoch : %d"%epoch)

    model_dict = model.state_dict()
    pretrained_dict = torch.load(os.path.join(args.out_dir,'checkpoint_%d'%(epoch)))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    loss_ = 0
    val_set3.predicted_values = np.zeros(val_set3.predicted_values.shape)
    with torch.no_grad() :
        val_set3.fill(model)
        for i,(err,mean,time,index) in enumerate(val_loader3):
            loss_ += err.data.sum()
            for i in range(err.shape[0]):
                val_set3.predicted_values[time[i]][index[i]] = mean[i]
    loss_ = loss_/int(len(val_set3))
    loss = val_set3.mse_loss()
    print ("MSE Loss is %f"%loss)
