import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import norm
import random

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        in_dim = 20
        #out_dim = in_dim
        out_dim = 1024
        out_dim1 = 128
        #out_dim2 = 32
        #self.conv1 = nn.Sequential(nn.Conv2d(1,1,(3,1),1),nn.ReLU())
        self.linear1 = nn.Sequential(nn.Linear(in_dim,out_dim),nn.ReLU())
        #self.linear2 = nn.Sequential(nn.Linear(out_dim,out_dim1),nn.ReLU())
        #self.linear3 = nn.Sequential(nn.Linear(out_dim1,out_dim2),nn.ReLU())
        self.linear4 = nn.Linear(out_dim,1)
        self.linear5 = nn.Linear(out_dim,1)
        
        
    def forward(self, x):
        out = self.linear1(x)
        return self.linear4(out).squeeze(),self.linear5(out).squeeze()

class TimeSeries(nn.Module):
    def __init__(self,hidden_dim=64,num_layers=2):
        super(TimeSeries, self).__init__()
        self.gru_left = nn.GRU(input_size=1,hidden_size=hidden_dim,bidirectional=False)
        self.gru_right = nn.GRU(input_size=1,hidden_size=hidden_dim,bidirectional=False)
        self.mean = nn.Linear(2*hidden_dim,1)
        self.std = nn.Linear(2*hidden_dim,1)
        
    def forward(self, x_left,x_right):
        out_left,_ = self.gru_left(x_left.transpose(0,1).unsqueeze(axis=2))
        out_right,_ = self.gru_right(x_right.transpose(0,1).unsqueeze(axis=2))
        out_left = out_left[-1,:,:]
        out_right = out_right[-1,:,:]
        temp = torch.cat([out_left,out_right],axis=1)
        return self.mean(temp).squeeze(),self.std(temp).squeeze()

class TimeSeriesDataset_(torch.utils.data.Dataset):
    def __init__(self,feats_file,examples_file):
        print ('loading dataset')
        self.feats = np.load(feats_file).astype(np.float32)
        self.examples_file = np.load(examples_file).astype(np.int)
        self.means = self.feats.mean(axis=0)
        self.stds = np.maximum(self.feats.std(axis=0),1e-7)
        self.normalised_feats = (self.feats - self.means)/self.stds
        self.predicted_values = np.zeros(self.feats.shape)
        
    def __getitem__(self,index):
        time_ = self.examples_file[index][0]
        shop_ = self.examples_file[index][1]
        prod_ = self.examples_file[index][2]
        out_ = self.feats[time_,shop_,prod_]
        out_right = self.feats[time_+1:,shop_,prod_]
        out_left = self.feats[:time_,shop_,prod_]
        temp = np.concatenate([out_left,out_right])
        std = np.maximum(np.std(temp),1e-7)
        mean = np.mean(temp)
        out_left = torch.from_numpy(out_left)
        out_right = torch.flip(torch.from_numpy(out_right),dims=(0,))
        out_left = (out_left-mean)/std
        out_right = (out_right-mean)/std
        out_ = (out_-mean)/std
        
        return out_left,out_right,out_,time_,shop_,prod_
    

    def mse_loss(self):
        loss = 0
        self.predicted_values = self.predicted_values*self.stds + self.means
        for x in self.examples_file:
            loss += (self.predicted_values[x[0],x[1],x[2]]-self.feats[x[0],x[1],x[2]])**2
        return loss/self.examples_file.shape[0]

    def __len__(self):
        return self.examples_file.shape[0]

    
def time_series_collate(batch):
    (x_left, x_right, y,time,shop,prod) = zip(*batch)
    x_left = torch.nn.utils.rnn.pad_sequence(x_left,batch_first=True,padding_value=0)
    x_right = torch.nn.utils.rnn.pad_sequence(x_right,batch_first=True,padding_value=0)
    index_info = {'time':time ,'index1' : shop,'index2':prod}
    return x_left,x_right,torch.FloatTensor(y),index_info

    
class Dataset_(torch.utils.data.Dataset):
    def __init__(self,feats_file,examples_file):
        print ('loading dataset')
        self.feats = torch.from_numpy(np.load(feats_file).astype(np.float32))
        self.examples_file = np.load(examples_file).astype(np.int)
        
    def __getitem__(self,index):
        time_ = self.examples_file[index][0]
        shop_ = self.examples_file[index][1]
        prod_ = self.examples_file[index][2]
        out_ = self.feats[time_,shop_,prod_]
        diff_shop_prev = self.feats[time_,shop_,:] - self.feats[time_-1,shop_,:]
        diff_shop_next = self.feats[time_,shop_,:] - self.feats[time_+1,shop_,:]
        diff_shop_prev2 = self.feats[time_,shop_,:] - self.feats[time_-2,shop_,:]
        diff_shop_next2 = self.feats[time_,shop_,:] - self.feats[time_+2,shop_,:]

        diff_prod_prev = self.feats[time_,:,prod_] - self.feats[time_-1,:,prod_]
        diff_prod_next = self.feats[time_,:,prod_] - self.feats[time_+1,:,prod_]
        diff_prod_prev2 = self.feats[time_,:,prod_] - self.feats[time_-2,:,prod_]
        diff_prod_next2 = self.feats[time_,:,prod_] - self.feats[time_+2,:,prod_]

        diff_shop_prev = np.concatenate([diff_shop_prev[:prod_],diff_shop_prev[prod_+1:]])
        diff_shop_next = np.concatenate([diff_shop_next[:prod_],diff_shop_next[prod_+1:]])
        diff_shop_prev2 = np.concatenate([diff_shop_prev2[:prod_],diff_shop_prev2[prod_+1:]])
        diff_shop_next2 = np.concatenate([diff_shop_next2[:prod_],diff_shop_next2[prod_+1:]])

        diff_prod_prev = np.concatenate([diff_prod_prev[:shop_],diff_prod_prev[shop_+1:]])
        diff_prod_next = np.concatenate([diff_prod_next[:shop_],diff_prod_next[shop_+1:]])
        diff_prod_prev2 = np.concatenate([diff_prod_prev2[:shop_],diff_prod_prev2[shop_+1:]])
        diff_prod_next2 = np.concatenate([diff_prod_next2[:shop_],diff_prod_next2[shop_+1:]])

        mean_ratios_shop = np.array([np.nanmean(diff_shop_prev),
                                     np.nanmean(diff_shop_prev2),
                                     np.nanmean(diff_shop_next),
                                     np.nanmean(diff_shop_next2)])
        
        var_ratios_shop = np.array([np.nanstd(diff_shop_prev),
                                     np.nanstd(diff_shop_prev2),
                                     np.nanstd(diff_shop_next),
                                     np.nanstd(diff_shop_next2)])
        
        mean_ratios_prod = np.array([np.nanmean(diff_prod_prev),
                                     np.nanmean(diff_prod_prev2),
                                     np.nanmean(diff_prod_next),
                                     np.nanmean(diff_prod_next2)])

        var_ratios_prod = np.array([np.nanstd(diff_prod_prev),
                                    np.nanstd(diff_prod_prev2),
                                    np.nanstd(diff_prod_next),
                                    np.nanstd(diff_prod_next2)])
        
        most_detailed = np.array([self.feats[time_-1,shop_,prod_],self.feats[time_+1,shop_,prod_],self.feats[time_-2,shop_,prod_],self.feats[time_+2,shop_,prod_]])
        in_ = np.concatenate([most_detailed,mean_ratios_shop,mean_ratios_prod,var_ratios_shop,var_ratios_prod]).astype(np.float32)
        return (in_,out_,(time_,shop_,prod_))
    
    def __len__(self):
        return self.examples_file.shape[0]

