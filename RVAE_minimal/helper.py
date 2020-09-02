import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle

class Dataset_(torch.utils.data.Dataset):
    def __init__(self,feats_file):
        print ('loading dataset')
        self.feats = torch.from_numpy(np.load(feats_file).astype(np.float32))
        self.context_size = 4
        self.time_size = self.feats.shape[0]-self.context_size
        self.num_shops = self.feats.shape[1]
        self.num_prods = self.feats.shape[2]
    def __getitem__(self,index):
        time_ = int(index%self.time_size)
        index = int((index - time_)/self.time_size)
        shop_ = int(index%self.num_shops)
        index = int((index - shop_)/self.num_shops)
        prod_ = int(index%self.num_prods)
        time_ += self.context_size
        time_ -= 2
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
        most_detailed = np.array([self.feats[time_-1,shop_,prod_],self.feats[time_+1,shop_,prod_],self.feats[time_-2,shop_,prod_],self.feats[time_+2,shop_,prod_],self.feats[time_,shop_,prod_]])
        in_ = np.concatenate([ratios_shop,ratios_prod,most_detailed])
        return in_,(time_,shop_,prod_)
    
    def __len__(self):
        return (self.time_size*self.num_shops*self.num_prods)
