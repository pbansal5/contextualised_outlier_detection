import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle

# class NN(nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()
#         latent_dim = 256
#         self.linear_most_detailed = nn.Sequential(nn.Linear(4,latent_dim),nn.ReLU())
#         self.linear_mean = nn.Sequential(nn.Linear(8,latent_dim),nn.ReLU())
#         self.linear_std = nn.Sequential(nn.Linear(8,latent_dim),nn.ReLU())
#         self.linear_mean2 = nn.Linear(3*latent_dim,1)
#         self.linear_std2 = nn.Linear(3*latent_dim,1)
        
        
#     def forward(self, x):
#         out = torch.cat([self.linear_most_detailed(x[:,:4]),self.linear_mean(x[:,4:12]),self.linear_std(x[:,-8:])],axis=1)
#         mean = self.linear_mean2(out).squeeze()
#         std = torch.exp(self.linear_std2(out).squeeze())
#         return mean,std


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
    def __init__(self):
        super(TimeSeries, self).__init__()
        hidden_dim = 64
        self.gru_left = nn.GRU(input_size=1,hidden_size=hidden_dim,bidirectional=True)
        self.gru_right = nn.GRU(input_size=1,hidden_size=hidden_dim,bidirectional=True)
        self.mean = nn.Linear(4*hidden_dim,1)
        self.std = nn.Linear(4*hidden_dim,1)
        
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
        self.feats = torch.from_numpy(np.load(feats_file).astype(np.float32))
        self.examples_file = np.load(examples_file).astype(np.int)
        
    def __getitem__(self,index):
        time_ = self.examples_file[index][0]
        shop_ = self.examples_file[index][1]
        prod_ = self.examples_file[index][2]
        out_left = self.feats[:time_,shop_,prod_]
        out_right = self.feats[time_+1:,shop_,prod_]
        out_ = self.feats[time_,shop_,prod_]
        return out_left,out_right,out_

    def __len__(self):
        return self.examples_file.shape[0]

def time_series_collate(batch):
    (x_left, x_right, y) = zip(*batch)
    # x_l_lens = [len(x) for x in x_left]
    # x_r_lens = [len(x) for x in x_right]
    # y_lens = [len(x) for x in y]
    x_left = torch.nn.utils.rnn.pad_sequence(x_left,batch_first=True,padding_value=0)
    x_right = torch.nn.utils.rnn.pad_sequence(x_right,batch_first=True,padding_value=0)
    return x_left,x_right,torch.FloatTensor(y)

    
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

