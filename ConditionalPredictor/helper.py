import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
from torch import nn
from torch.nn import functional as F
    

class NN(nn.Module):
    def __init__(self):

        super(NN, self).__init__()

        feats = 3
        embedding_size = 50
        layer_size = 400
        latent_size = 5
        
        self.feat_info = [["time",'categ',31],['pulocation','categ',266],['dolocation','categ',266]]
        self.size_input = feats*50
        self.size_output = 1

        self.feat_embedd = nn.ModuleList([nn.Embedding(c_size, embedding_size, max_norm=1)
                                          for _, col_type, c_size in self.feat_info
                                          if col_type=="categ"])

        self.fc1 = nn.Linear(self.size_input, layer_size)
        self.fc2 = nn.Linear(layer_size, latent_size)

        self.fc3 = nn.Linear(latent_size,1)
        self.activ = nn.ReLU()


    def get_inputs(self, x_data):
        input_list = []
        cursor_embed = 0
        start = 0
        
        for feat_idx, ( _, col_type, feat_size ) in enumerate(self.feat_info):
            if col_type == "categ":
                aux_categ = self.feat_embedd[cursor_embed](x_data[:,feat_idx].long())#*drop_mask[:,feat_idx].view(-1,1)
                input_list.append(aux_categ)
                cursor_embed += 1
                    
            elif col_type == "real": 
                input_list.append((x_data[:,feat_idx]).view(-1,1).float())#*drop_mask[:,feat_idx]
                    
        return torch.cat(input_list, 1)

    def forward(self, x_data):
        input_values = self.get_inputs(x_data)
        fc1_out = self.activ(self.fc1(input_values))
        fc2_out = self.activ(self.fc2(fc1_out))
        out = self.fc3(fc2_out)
        return out.squeeze() 

    
class Dataset_(torch.utils.data.Dataset):
    def __init__(self,feats_file,examples_file):
        print ('loading dataset')
        self.feats = torch.from_numpy(np.load(feats_file).astype(np.float32))
        self.examples_file = np.load(examples_file).astype(np.int)

    def __getitem__(self,index):
        time_ = self.examples_file[index][0]
        shop_ = self.examples_file[index][1]
        prod_ = self.examples_file[index][2]
        return torch.FloatTensor([time_,shop_,prod_]),self.feats[time_,shop_,prod_],(time_,shop_,prod_)
    
    def __len__(self):
        return (self.examples_file.shape[0])
