import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
from helper import Dataset_,NN

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
lr = 1e-3

loss = torch.nn.L1Loss()


if args.test:
    test_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_test.npy')
    test_loader = torch.utils.data.DataLoader(test_set,batch_size = batch_size,drop_last = True)
    model = NN().to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    pred_ = []
    for x,y,_ in test_loader :
        loss_ = 0
        with torch.no_grad():
            x = x.to(device)
            y_pred = model(x)
            y = y.to(device)
            loss_ += loss(y,y_pred).data
            for i in range(batch_size):
                pred_.append([y[i].data.cpu().numpy(),y_pred[i].data.cpu().numpy()])
            
    loss_ = (loss_/int(len(test_set)/batch_size)).cpu().data.numpy()
    print (loss_)
    pred_ = np.array(pred_)
    print (pred_.shape)
    np.save('../dataset/predictions_russian.npy',pred_)
    exit()


    
# train_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_complete.npy','../dataset/nyc_taxi_train_non_zero_2zones_examples.npy')
# val_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_complete.npy','../dataset/nyc_taxi_test_non_zero_2zones_examples.npy')

# train_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_complete.npy','../dataset/nyc_taxi_train_non_zero_examples.npy')
# val_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_complete.npy','../dataset/nyc_taxi_test_non_zero_examples.npy')

train_set = Dataset_('../dataset/nyc_taxi_norm_noshift_numpy_complete.npy','../dataset/nyc_taxi_train_non_zero_examples.npy')
# val_set1 = Dataset_('../dataset/nyc_taxi_norm_mean_numpy_our.npy','../dataset/nyc_taxi_test_our_outlier_examples.npy')
# val_set2 = Dataset_('../dataset/nyc_taxi_norm_mean_numpy_our.npy','../dataset/nyc_taxi_test_our_inlier_examples.npy')
val_set3 = Dataset_('../dataset/nyc_taxi_norm_noshift_numpy_complete.npy','../dataset/nyc_taxi_test_non_zero_examples.npy')

# train_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_20_modified.npy','../dataset/nyc_taxi_train_non_zero_no20_examples.npy')
# val_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_20_modified.npy','../dataset/nyc_taxi_test_non_zero_20_examples.npy')

# train_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_20_138_modified.npy','../dataset/nyc_taxi_train_non_zero_zonetime_100%_examples.npy')
# val_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_20_138_modified.npy','../dataset/nyc_taxi_test_non_zero_zonetime_100%_examples.npy')

# train_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_complete.npy','../dataset/nyc_taxi_train_examples.npy')
# val_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_complete.npy','../dataset/nyc_taxi_test_examples.npy')

# train_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_20_modified.npy','../dataset/nyc_taxi_train_all_no20_examples.npy')
# val_set = Dataset_('../dataset/nyc_taxi_norm_meadian_numpy_20_modified.npy','../dataset/nyc_taxi_test_all_20_examples.npy')

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = True,shuffle=True)
# val_loader1 = torch.utils.data.DataLoader(val_set1,batch_size = batch_size,drop_last = False,shuffle=True)
# val_loader2 = torch.utils.data.DataLoader(val_set2,batch_size = batch_size,drop_last = False,shuffle=True)
val_loader3 = torch.utils.data.DataLoader(val_set3,batch_size = batch_size,drop_last = False,shuffle=True)

model = NN().to(device)
if (args.warm_start):
    model.load_state_dict(torch.load(args.checkpoint))
optim = torch.optim.Adam(model.parameters(),lr=lr)#,weight_decay = 1e-3)


writer = SummaryWriter(os.path.join('runs',args.log_file))
max_epoch = 50
iteration = 0

for epoch in range(max_epoch):
    print ("Starting Epoch : %d"%epoch)

    for x,y,_ in train_loader :
        x = x.to(device)
        y_pred,var = model(x)
        y = y.to(device)
        loss_ = (((y_pred-y)**2)/torch.exp(var) + var).mean()
        optim.zero_grad()
        loss_.backward()
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
