import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle

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
        
        
    def forward(self, x):
        return self.linear4(self.linear1(x)).squeeze()


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

        ratios_shop = np.array([np.nanmean(diff_shop_prev),
                                np.nanmean(diff_shop_prev2),
                                np.nanmean(diff_shop_next),
                                np.nanmean(diff_shop_next2),
                                np.nanstd(diff_shop_prev),
                                np.nanstd(diff_shop_prev2),
                                np.nanstd(diff_shop_next),
                                np.nanstd(diff_shop_next2)])

        ratios_prod = np.array([np.nanmean(diff_prod_prev),
                                np.nanmean(diff_prod_prev2),
                                np.nanmean(diff_prod_next),
                                np.nanmean(diff_prod_next2),
                                np.nanstd(diff_prod_prev),
                                np.nanstd(diff_prod_prev2),
                                np.nanstd(diff_prod_next),
                                np.nanstd(diff_prod_next2)])
        most_detailed = np.array([self.feats[time_-1,shop_,prod_],self.feats[time_+1,shop_,prod_],self.feats[time_-2,shop_,prod_],self.feats[time_+2,shop_,prod_]])
        in_ = np.concatenate([most_detailed,ratios_shop,ratios_prod])[None,:]
        return (in_,out_,(time_,shop_,prod_))
    
    def __len__(self):
        return self.examples_file.shape[0]

