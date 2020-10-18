import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
from helper import NN, Dataset_

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


train_set = Dataset_('dataset/jantahack_noshift_numpy_complete.npy','dataset/jantahack_train_examples.npy')
val_set3 = Dataset_('dataset/jantahack_noshift_numpy_complete.npy','dataset/jantahack_test_examples.npy')

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True)
# val_loader1 = torch.utils.data.DataLoader(val_set1,batch_size = batch_size,drop_last = False,shuffle=True)
# val_loader2 = torch.utils.data.DataLoader(val_set2,batch_size = batch_size,drop_last = False,shuffle=True)
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = batch_size,drop_last = False,shuffle=True)

model = NN().to(device)
if (args.warm_start):
    model.load_state_dict(torch.load(args.checkpoint))
optim = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)#,weight_decay = 1e-3)
#optim = torch.optim.Adam(model.parameters(),lr=lr)#,weight_decay = 1e-3)


writer = SummaryWriter(os.path.join('runs',args.log_file))
max_epoch = 100
iteration = 0

for epoch in range(max_epoch):
    print ("Starting Epoch : %d"%epoch)

    for x,y,_ in train_loader :
        x = x.to(device)
        y_pred,var = model(x)
        y = y.to(device)
        loss_ = (((y_pred-y)**2)/torch.exp(var) + var).mean()  #nllloss(y,y_pred,var)
        optim.zero_grad()
        loss_.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optim.step()
        iteration += 1
        writer.add_scalar('training/loss',loss_,iteration)

    if (epoch % 1 == 0):
        torch.save(model.state_dict(), os.path.join(args.out_dir,'checkpoint_%d'%epoch))
        
        # loss_ = 0
        # for x,y,_ in val_loader1 :
        #     with torch.no_grad():
        #         x = x.to(device)
        #         y_pred,var = model(x)
        #         y = y.to(device)
        #         loss_ += (((y_pred-y)**2)/var + var).mean().data*x.shape[0]
        # loss_ = loss_/int(len(val_set1))
        # writer.add_scalar('validation/outliers_loss',loss_,iteration)

        # loss_ = 0
        # for x,y,_ in val_loader2 :
        #     with torch.no_grad():
        #         x = x.to(device)
        #         y_pred,var = model(x)
        #         y = y.to(device)
        #         loss_ += (((y_pred-y)**2)/var + var).mean().data*x.shape[0]
        # loss_ = loss_/int(len(val_set2))
        # writer.add_scalar('validation/inliers_loss',loss_,iteration)

        loss_ = 0
        for x,y,_ in val_loader3 :
            with torch.no_grad():
                x = x.to(device)
                y_pred,var = model(x)
                y = y.to(device)
                loss_ += (((y_pred-y)**2)/torch.exp(var) + var).mean().data*x.shape[0]
        loss_ = loss_/int(len(val_set3))
        writer.add_scalar('validation/truly_inliers_loss',loss_,iteration)
