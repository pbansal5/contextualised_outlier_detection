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
        weights1 = torch.exp(-similarity1/self.tau)
        weights2 = torch.exp(-similarity2/self.tau)
        indices1 = torch.argsort(weights1)[:,-(self.k+1):-1]
        indices2 = torch.argsort(weights2)[:,-(self.k+1):-1]
        selected1 = y_context1[np.arange(y_context1.shape[0])[:,None],indices1]
        selected2 = y_context2[np.arange(y_context2.shape[0])[:,None],indices2]
        weights1 = weights1[np.arange(y_context1.shape[0])[:,None],indices1]
        weights2 = weights2[np.arange(y_context2.shape[0])[:,None],indices2]
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

        return err1,err2,mean_outlier

    def detached_forward(self, x_left,x_right,y,context_info,use_outlier=True):
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

        feats = torch.cat([self.compute_feats(y1_context.to(x_left.device),y2_context.to(x_left.device),index1,index2).to(x_left.device),mean_time_series.detach().requires_grad_(True),std_time_series.detach().requires_grad_(True)],axis=1)
        feats = self.outlier_layer1(feats).clamp(min=0)
        mean_outlier = self.mean_outlier_layer(feats)
        std_outlier = self.std_outlier_layer(feats)
        #residual2 = (0-mean_outlier)/torch.exp(std_outlier/2)
        
        err1 = (((y.unsqueeze(1)-mean_time_series)**2/torch.exp(std_time_series)) + std_time_series).mean()
        err2 = (((y.unsqueeze(1)-mean_outlier)**2/torch.exp(std_outlier)) + std_outlier).mean()

        if use_outlier:
            return err1,err2,mean_outlier
        else:
            return err1,err2,mean_time_series

    
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
        
        return out_left,out_right,out_,self.residuals[time_,:,prod_],self.residuals[time_,shop_,:],time_,shop_,prod_
        # shop_context = self.normalised_feats[time_,:,prod_]
        # prod_context = self.normaslied_feats[time_,shop_,:]
        # return out_left,out_right,out_,shop_context,prod_context,time_,shop_,prod_

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
    

    def mse_loss(self):
        loss = 0
        self.predicted_values = self.predicted_values*self.stds + self.means
        for x in self.examples_file:
            loss += (self.predicted_values[x[0],x[1],x[2]]-self.feats[x[0],x[1],x[2]])**2

        return loss/self.examples_file.shape[0]

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

        
class Proposal1Model1D(nn.Module):
    def __init__(self,size1=266,hidden_dim=64,embedding_size=32):
        super(Proposal1Model1D, self).__init__()
        self.k = 20
        self.tau = 1
        self.num_layers = 2
        self.gru_left = nn.GRU(input_size=1,hidden_size=hidden_dim,num_layers = self.num_layers,bidirectional=False)
        self.gru_right = nn.GRU(input_size=1,hidden_size=hidden_dim,num_layers = self.num_layers,bidirectional=False)
        self.mean = nn.Linear(2*hidden_dim,1)
        self.std = nn.Linear(2*hidden_dim,1)
        self.embeddings1 = nn.Embedding(size1, embedding_size, max_norm=1)
        self.outlier_layer1 = nn.Linear(5,64)
        self.mean_outlier_layer = nn.Linear(64,1)
        self.std_outlier_layer = nn.Linear(64,1)

    def compute_feats(self,y_context1,index1):
        similarity1 = torch.cdist(self.embeddings1.weight[index1].unsqueeze(0),self.embeddings1.weight.unsqueeze(0),p=2).squeeze()
        weights1 = torch.exp(-similarity1/self.tau)
        indices1 = torch.argsort(weights1)[:,-(self.k+1):-1]
        selected1 = y_context1[np.arange(y_context1.shape[0])[:,None],indices1]
        weights1 = weights1[np.arange(y_context1.shape[0])[:,None],indices1]
        return torch.cat([(selected1*weights1).sum(dim=1,keepdim=True)/weights1.sum(dim=1,keepdim=True),weights1.sum(dim=1,keepdim=True),torch.std(selected1,dim=1,keepdim=True)],dim=1)#.transpose(0,1)
    
    def forward(self, x_left,x_right,y,context_info):
        out_left,_ = self.gru_left(x_left.transpose(0,1).unsqueeze(axis=2))
        out_right,_ = self.gru_right(x_right.transpose(0,1).unsqueeze(axis=2))
        out_left = out_left[-1,:,:]
        out_right = out_right[-1,:,:]
        temp = torch.cat([out_left,out_right],axis=1)
        mean_time_series = self.mean(temp)
        std_time_series = self.std(temp)
        
        index1 = context_info['index1']
        y1_context = context_info['y1_context']

        feats = torch.cat([self.compute_feats(y1_context.to(x_left.device),index1).to(x_left.device),mean_time_series,std_time_series],axis=1)
        feats = self.outlier_layer1(feats).clamp(min=0)
        mean_outlier = self.mean_outlier_layer(feats)
        std_outlier = self.std_outlier_layer(feats)
        
        err1 = (((y.unsqueeze(1)-mean_time_series)**2/torch.exp(std_time_series)) + std_time_series).mean()
        err2 = (((y.unsqueeze(1)-mean_outlier)**2/torch.exp(std_outlier)) + std_outlier).mean()

        return err1,err2,mean_outlier
    
    
    def detached_forward(self, x_left,x_right,y,context_info,use_outlier=True):
        out_left,_ = self.gru_left(x_left.transpose(0,1).unsqueeze(axis=2))
        out_right,_ = self.gru_right(x_right.transpose(0,1).unsqueeze(axis=2))
        out_left = out_left[-1,:,:]
        out_right = out_right[-1,:,:]
        temp = torch.cat([out_left,out_right],axis=1)
        mean_time_series = self.mean(temp)
        std_time_series = self.std(temp)
        
        index1 = context_info['index1']
        y1_context = context_info['y1_context']

        feats = torch.cat([self.compute_feats(y1_context.to(x_left.device),index1).to(x_left.device),mean_time_series.detach().requires_grad_(True),std_time_series.detach().requires_grad_(True)],axis=1)
        feats = self.outlier_layer1(feats).clamp(min=0)
        mean_outlier = self.mean_outlier_layer(feats)
        std_outlier = self.std_outlier_layer(feats)
        #residual2 = (0-mean_outlier)/torch.exp(std_outlier/2)
        
        err1 = (((y.unsqueeze(1)-mean_time_series)**2/torch.exp(std_time_series)) + std_time_series).mean()
        err2 = (((y.unsqueeze(1)-mean_outlier)**2/torch.exp(std_outlier)) + std_outlier).mean()

        if use_outlier:
            return err1,err2,mean_outlier
        else:
            return err1,err2,mean_time_series

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

    def get_sigma(self):
        similarity1 = torch.exp(-torch.cdist(self.embeddings1.weight.unsqueeze(0),self.embeddings1.weight.unsqueeze(0),p=2).squeeze()/self.tau)
        return similarity1



class Proposal1Dataset1D_(torch.utils.data.Dataset):
    def __init__(self,feats_file,examples_file):
        print ('loading dataset')
        self.feats = np.load(feats_file).astype(np.float32)
        self.examples_file = np.load(examples_file).astype(np.int)
        self.residuals = np.zeros(self.feats.shape).astype(np.float)
        self.means = self.feats.mean(axis=0)
        self.stds = np.maximum(self.feats.std(axis=0),1e-7)
        self.normalised_feats = (self.feats - self.means)/self.stds
        self.predicted_values = np.zeros(self.feats.shape)
        
    def __getitem__(self,index):
        time_ = self.examples_file[index][0]
        shop_ = self.examples_file[index][1]
        out_ = self.feats[time_,shop_]
        out_right = self.feats[time_+1:,shop_]
        out_left = self.feats[:time_,shop_]
        temp = np.concatenate([out_left,out_right])
        std = np.maximum(np.std(temp),1e-7)
        mean = np.mean(temp)
        out_left = torch.from_numpy(out_left)
        out_right = torch.flip(torch.from_numpy(out_right),dims=(0,))
        #out_right = torch.from_numpy(out_right)[::-1]
        out_left = (out_left-mean)/std
        out_right = (out_right-mean)/std
        out_ = (out_-mean)/std
        
        return out_left,out_right,out_,self.residuals[time_,:],time_,shop_
        # shop_context = self.normalised_feats[time_,:]
        # return out_left,out_right,out_,shop_context,time_,shop_

    def update_residuals (self,model,loader,device):
        with torch.no_grad() :
            for x_right,x_left,y,context_info in loader:
                residual = model.compute_residuals(x_right.to(device),x_left.to(device),y.to(device)).cpu().data.numpy()
                shops = context_info['index1']
                times = context_info['time']
                for val,shop,time in zip(residual,shops,times):
                    self.residuals[time,shop] = val
        return

    def mse_loss(self):
        loss = 0
        self.predicted_values = self.predicted_values*self.stds + self.means
        for x in self.examples_file:
            loss += (self.predicted_values[x[0],x[1]]-self.feats[x[0],x[1]])**2
        return loss/self.examples_file.shape[0]
    
    def __len__(self):
        return self.examples_file.shape[0]

def proposal1_collate1d(batch):
    (x_left,x_right,y,y1_context,time_,shop_) = zip(*batch)
    x_left = torch.nn.utils.rnn.pad_sequence(x_left,batch_first=True,padding_value=0)
    x_right = torch.nn.utils.rnn.pad_sequence(x_right,batch_first=True,padding_value=0)
    # print (len(y1_context))
    # print (len(y1_context[0]))
    context_info = dict({'y1_context':torch.FloatTensor(y1_context),
                         'index1':np.array(shop_).astype(np.int),
                         'time':np.array(time_).astype(np.int)})
    return x_left,x_right,torch.FloatTensor(y),context_info
