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
        
class Proposal1Model(nn.Module):
    def __init__(self,size1=266,size2=266):
        super(Proposal1Model, self).__init__()
        hidden_dim = 64
        embedding_size = 32
        self.k = 20
        self.tau = 1
        self.num_layers = 2
        self.gru_left = nn.GRU(input_size=1,hidden_size=hidden_dim,num_layers = self.num_layers,bidirectional=False)
        self.gru_right = nn.GRU(input_size=1,hidden_size=hidden_dim,num_layers = self.num_layers,bidirectional=False)
        self.mean = nn.Linear(2*hidden_dim,1)
        self.std = nn.Linear(2*hidden_dim,1)
        self.embeddings1 = nn.Embedding(size1, embedding_size, max_norm=1)
        self.embeddings2 = nn.Embedding(size2, embedding_size, max_norm=1)
        self.outlier_layer1 = nn.Linear(8,64)
        self.mean_outlier_layer = nn.Linear(64,1)
        self.std_outlier_layer = nn.Linear(64,1)
        # self.embeddings1.requires_grad = True#False
        # self.embeddings2.requires_grad = True#False

    def compute_feats(self,y_context1,y_context2,index1,index2):
        similarity1 = torch.cdist(self.embeddings1.weight[index1].unsqueeze(0),self.embeddings1.weight.unsqueeze(0),p=2).squeeze()
        similarity2 = torch.cdist(self.embeddings2.weight[index2].unsqueeze(0),self.embeddings2.weight.unsqueeze(0),p=2).squeeze()
        indices1 = torch.argsort(similarity1)[:,-(self.k+1):-1]
        indices2 = torch.argsort(similarity2)[:,-(self.k+1):-1]
        weights1 = torch.exp(-similarity1[:,-(self.k+1):-1]/self.tau)
        weights2 = torch.exp(-similarity2[:,-(self.k+1):-1]/self.tau)
        selected1 = y_context1[np.arange(y_context1.shape[0])[:,None],indices1]
        selected2 = y_context2[np.arange(y_context2.shape[0])[:,None],indices2]
        #print (selected1.shape)
        #print (weights1.shape)
        return torch.cat([(selected1*weights1).sum(dim=1,keepdim=True)/weights1.sum(dim=1,keepdim=True),weights1.sum(dim=1,keepdim=True),torch.std(selected1,dim=1,keepdim=True),
                          (selected2*weights2).sum(dim=1,keepdim=True)/weights2.sum(dim=1,keepdim=True),weights2.sum(dim=1,keepdim=True),torch.std(selected2,dim=1,keepdim=True)],dim=1)#.transpose(0,1)
    
    def forward(self, x_left,x_right,y,context_info):
        out_left,_ = self.gru_left(x_left.transpose(0,1).unsqueeze(axis=2))
        out_right,_ = self.gru_right(x_right.transpose(0,1).unsqueeze(axis=2))
        out_left = out_left[-1,:,:]
        out_right = out_right[-1,:,:]
        temp = torch.cat([out_left,out_right],axis=1)
        mean_time_series = self.mean(temp)
        std_time_series = self.std(temp)
        
        index1 = context_info['index1']
        index2 = context_info['index2']
        y1_context = context_info['y1_context']
        y2_context = context_info['y2_context']

        feats = torch.cat([self.compute_feats(y1_context.to(x_left.device),y2_context.to(x_left.device),index1,index2).to(x_left.device),mean_time_series,std_time_series],axis=1)
        feats = self.outlier_layer1(feats).clamp(min=0)
        mean_outlier = self.mean_outlier_layer(feats)
        std_outlier = self.std_outlier_layer(feats)
        #residual2 = (0-mean_outlier)/torch.exp(std_outlier/2)
        
        err1 = (((y.unsqueeze(1)-mean_time_series)**2/torch.exp(std_time_series)) + std_time_series).mean()
        err2 = (((y.unsqueeze(1)-mean_outlier)**2/torch.exp(std_outlier)) + std_outlier).mean()

        return err1,err2
    
    def compute_residuals(self, x_left,x_right,y):
        out_left,_ = self.gru_left(x_left.transpose(0,1).unsqueeze(axis=2))
        out_right,_ = self.gru_right(x_right.transpose(0,1).unsqueeze(axis=2))
        out_left = out_left[-1,:,:]
        out_right = out_right[-1,:,:]
        temp = torch.cat([out_left,out_right],axis=1)
        mean_time_series = self.mean(temp)
        std_time_series = self.std(temp)
        residual1 = (y.unsqueeze(1)-mean_time_series)/torch.exp(std_time_series/2)
        return residual1.squeeze()



class Proposal1Dataset_(torch.utils.data.Dataset):
    def __init__(self,feats_file,examples_file):
        print ('loading dataset')
        self.feats = np.load(feats_file).astype(np.float32)
        self.examples_file = np.load(examples_file).astype(np.int)
        self.residuals = np.zeros(self.feats.shape).astype(np.float)
        
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
        #out_right = torch.from_numpy(out_right)[::-1]
        out_left = (out_left-mean)/std
        out_right = (out_right-mean)/std
        out_ = (out_-mean)/std
        
        #return out_left,out_right,out_,self.residuals[time_,:,prod_],self.residuals[time_,shop_,:],time_,shop_,prod_
        shop_context = self.feats[time_,:,prod_]
        prod_context = self.feats[time_,shop_,:]
        shop_context = (shop_context - np.mean(shop_context))/np.std(shop_context)
        prod_context = (prod_context - np.mean(prod_context))/np.std(prod_context)
        return out_left,out_right,out_,shop_context,prod_context,time_,shop_,prod_

    def update_residuals (self,model,loader,device):
        with torch.no_grad() :
            for x_right,x_left,y,context_info in loader:
                residual = model.compute_residuals(x_right.to(device),x_left.to(device),y.to(device)).cpu().data.numpy()
                shops = context_info['index1']
                prods = context_info['index2']
                times = context_info['time']
                for val,shop,prod,time in zip(residual,shops,prods,times):
                    self.residuals[time,shop,prod] = val
        return
    
    def __len__(self):
        return self.examples_file.shape[0]

def proposal1_collate(batch):
    (x_left,x_right,y,y1_context,y2_context,time_,shop_,prod_) = zip(*batch)
    x_left = torch.nn.utils.rnn.pad_sequence(x_left,batch_first=True,padding_value=0)
    x_right = torch.nn.utils.rnn.pad_sequence(x_right,batch_first=True,padding_value=0)
    # print (len(y1_context))
    # print (len(y1_context[0]))
    context_info = dict({'y1_context':torch.FloatTensor(y1_context),'y2_context':torch.FloatTensor(y2_context),
                         'index1':np.array(shop_).astype(np.int),'index2':np.array(prod_).astype(np.int),
                         'time':np.array(time_).astype(np.int)})
    return x_left,x_right,torch.FloatTensor(y),context_info


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

