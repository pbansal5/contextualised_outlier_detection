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
        out_dim = 512
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

        ratios_shop = np.array([diff_shop_prev.mean(),
                           diff_shop_prev2.mean(),
                           diff_shop_next.mean(),
                           diff_shop_next2.mean(),
                           diff_shop_prev.std(),
                           diff_shop_prev2.std(),
                           diff_shop_next.std(),
                           diff_shop_next2.std()])
        ratios_prod = np.array([diff_prod_prev.mean(),
                           diff_prod_prev2.mean(),
                           diff_prod_next.mean(),
                           diff_prod_next2.mean(),
                           diff_prod_prev.std(),
                           diff_prod_prev2.std(),
                           diff_prod_next.std(),
                           diff_prod_next2.std()])
        most_detailed = np.array([self.feats[time_-1,shop_,prod_],self.feats[time_+1,shop_,prod_],self.feats[time_-2,shop_,prod_],self.feats[time_+2,shop_,prod_]])
        in_ = np.concatenate([most_detailed,ratios_shop,ratios_prod])[None,:]
        return (in_,out_,(time_,shop_,prod_))
    
    def __len__(self):
        return self.examples_file.shape[0]

