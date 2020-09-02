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
        self.time_size = self.feats.shape[0]
        self.num_shops = self.feats.shape[1]
        self.num_prods = self.feats.shape[2]
    def __getitem__(self,index):
        time_ = int(index%self.time_size)
        index = int((index - time_)/self.time_size)
        shop_ = int(index%self.num_shops)
        index = int((index - shop_)/self.num_shops)
        prod_ = int(index%self.num_prods)
        return torch.FloatTensor([time_,shop_,prod_,self.feats[time_,shop_,prod_]]),(time_,shop_,prod_)
    
    def __len__(self):
        return (self.time_size*self.num_shops*self.num_prods)
