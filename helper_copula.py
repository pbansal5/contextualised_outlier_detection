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
    
class CopulaModel(nn.Module):
    def __init__(self):
        super(CopulaModel, self).__init__()
        hidden_dim = 64
        rank = 64
        self.num_layers = 2
        self.gru_left = nn.GRU(input_size=1,hidden_size=hidden_dim,num_layers = self.num_layers,bidirectional=False)
        self.gru_right = nn.GRU(input_size=1,hidden_size=hidden_dim,num_layers = self.num_layers,bidirectional=False)
        self.mean = nn.Linear(2*hidden_dim,1)
        self.d = nn.Linear(2*hidden_dim,1)
        self.v = nn.Linear(2*hidden_dim,rank)

    def forward (self,x_right,x_left,y):
        mus = []
        vs = []
        ds = []
        for i in range(x_left.shape[1]):
            out_left,_ = self.gru_left(x_left[:,i,:].transpose(0,1).unsqueeze(axis=2))
            out_right,_ = self.gru_right(x_right[:,i,:].transpose(0,1).unsqueeze(axis=2))
            out_left = out_left[-1,:,:]
            out_right = out_right[-1,:,:]
            temp = torch.cat([out_left,out_right],axis=1)
            mus.append(self.mean(temp))
            vs.append(self.v(temp))
            ds.append(torch.log(1+torch.exp(self.d(temp))))
            
        mu = torch.stack(mus,axis=0).transpose(0,1)
        d = torch.stack(ds,axis=0).transpose(0,1)
        v = torch.stack(vs,axis=0).transpose(0,1)
        sigma = torch.bmm(v,v.transpose(1,2))+torch.diag_embed(d.squeeze())
        temp = torch.bmm((y-mu).transpose(1,2),torch.bmm(torch.inverse(sigma),y-mu)).squeeze() + torch.logdet(sigma)
        return temp.mean()

    def infer (self,x_right,x_left):
        if (x_left.shape[1] != 0):
            out_left,_ = self.gru_left(x_left.transpose(0,1).unsqueeze(axis=2))
        if (x_right.shape[1] != 0):
            out_right,_ = self.gru_right(x_right.transpose(0,1).unsqueeze(axis=2))
        if (x_left.shape[1] != 0):
            out_left = out_left[-1,:,:]
        else :
            out_left = torch.zeros(out_right[-1,:,:].shape).to(out_right.device)
        if (x_right.shape[1] != 0):
            out_right = out_right[-1,:,:]
        else :
            out_right = torch.zeros(out_left.shape).to(out_left.device)
            
        temp = torch.cat([out_left,out_right],axis=1)
        return self.mean(temp),self.v(temp),torch.log(1+torch.exp(self.d(temp)))
        

class CopulaDataset_(torch.utils.data.Dataset):
    def __init__(self,feats_file,examples_file):
        print ('loading dataset')
        self.feats = np.load(feats_file).astype(np.float32)
        #self.feats = self.feats.view(self.feats.shape[0],-1)
        self.k = 10
        self.examples_file = np.load(examples_file).astype(np.int)
        self.residuals = np.zeros(self.feats.shape).astype(np.float)
        self.shop_prods = [[] for x in range(np.max(self.examples_file[:,0])+1)]
        for i in range (self.examples_file.shape[0]):
            temp = self.examples_file[i]
            self.shop_prods[temp[0]].append((temp[1],temp[2]))
        self.m = self.feats.shape[0]-1
        self.probs = [norm.ppf(i/self.m) for i in range(1,self.m)]
        self.probsdelta = 1/((self.m**0.25)*4*np.sqrt(3.14*np.log(self.m)))
        self.probs.append(norm.ppf(1-self.probsdelta))
        
    def __getitem__(self,index):
        time_ = self.examples_file[index][0]
        indices = random.sample(self.shop_prods[time_],self.k-1)
        indices.append((self.examples_file[index,1],self.examples_file[index,2]))
        out_ = []
        out_right = []
        out_left = []
        for x in indices :
            temp_out = self.feats[time_,x[0],x[1]]
            temp = np.concatenate([self.feats[:time_,x[0],x[1]],self.feats[time_+1:,x[0],x[1]]])
            temp = temp + np.random.rand(*temp.shape)/1000
            a = (temp<temp_out).sum()
            if (a != 0 and a != self.m):
                upper = temp[temp <= temp_out].max()
                lower = temp[temp > temp_out].min()
                temp_out = self.probs[a-1] + (self.probs[a]-self.probs[a-1])*(temp_out-lower)/(upper-lower)
            elif a == 0 :
                temp_out = norm.ppf(self.probsdelta)
            else :
                temp_out = norm.ppf(1-self.probsdelta)
            std = np.maximum(np.std(temp),1e-7)
            mean = np.mean(temp)
            temp = (temp-mean)/std
            out_right.append(torch.flip(torch.from_numpy(temp[time_:]),dims=(0,)))
            out_left.append(torch.from_numpy(temp[:time_]))
            out_.append(temp_out)
        out_ = torch.FloatTensor(out_)
        out_left = torch.stack(out_left,dim=0).transpose(0,1)
        out_right = torch.stack(out_right,dim=0).transpose(0,1)
        
        return out_left.float(),out_right.float(),out_.float()
    
    def __len__(self):
        return self.examples_file.shape[0]

class ValidationCopulaDataset_(torch.utils.data.Dataset):
    def __init__(self,feats_file,examples_file):
        print ('loading dataset')
        rank = 64
        self.feats = np.load(feats_file).astype(np.float32)
        self.shape = self.feats.shape
        self.feats = np.reshape(self.feats,(self.feats.shape[0],-1))
        self.examples_file = np.load(examples_file).astype(np.int)
        self.mus = np.zeros(self.feats.shape).astype(np.float)
        self.sigmas = np.zeros((self.feats.shape[0],self.feats.shape[1],self.feats.shape[1])).astype(np.float)
        self.gauss = np.zeros(self.feats.shape)
        self.normalised = np.zeros(self.feats.shape)
        
        self.m = self.feats.shape[0]
        
        temp = np.argsort(self.feats,axis=0)
        ranks = np.empty_like(temp)
        np.put_along_axis(ranks, temp, np.repeat(np.arange(temp.shape[0])[:,None],temp.shape[1],axis=1), axis=0)
        self.gauss = ranks/self.feats.shape[0]
        self.gauss[self.gauss == 0] = 1/((self.m**0.25)*4*np.sqrt(3.14*np.log(self.m)))

        mean = np.mean(self.feats,axis=0)
        std = np.maximum(np.std(self.feats,axis=0),1e-7)
        self.normalised = (self.feats-mean)/std

    def __getitem__(self,index):
        time_ = self.examples_file[index][0]
        index = self.examples_file[index][1]*self.shape[2] + self.examples_file[index][2]
        
        mu_others = np.concatenate([self.mus[time_][:index],self.mus[time_][index+1:]],axis=0)[None,:]
        sigma_others = np.concatenate([self.sigmas[time_][:index,:],self.sigmas[time_][index+1:,:]],axis=0)
        sigma_others = np.concatenate([sigma_others[:,:index],sigma_others[:,index+1:]],axis=1)
        sigma12 = self.sigmas[time_,index,:]
        sigma12 = np.concatenate([sigma12[:index],sigma12[index+1:]],axis=0)[None,:]
        x_others = np.concatenate([self.gauss[time_][:index],self.gauss[time_][index+1:]],axis=0)[None,:]
        sigma_others_inverse = np.linalg.inv(sigma_others)
        temp = np.matmul(sigma_others_inverse,(x_others-mu_others)[0][:,None])
        mean = self.mus[time_,index] - np.matmul(sigma12,temp)
        var = self.sigmas[time_,index,index] - np.matmul(sigma12,np.matmul(sigma_others_inverse,sigma12[0][:,None]))
        
        err = (((self.gauss[time_,index]-mean)**2/var) + np.log(var)).mean()
        
        return err

    def fill(self,model):
        device  = next(model.parameters()).device
        for time_ in range(self.feats.shape[0]):
            x_left = torch.from_numpy(self.normalised[:time_]).transpose(0,1)
            x_right = torch.flip(torch.from_numpy(self.normalised[time_+1:]),dims=(0,)).transpose(0,1)
            mut,vt,dt = model.infer(x_right.to(device),x_left.to(device))
            self.mus[time_,:] = mut.squeeze().cpu().data.numpy()
            self.sigmas[time_,:,:] = (torch.matmul(vt.squeeze(),vt.squeeze().transpose(0,1))+torch.diag(dt.squeeze())).data.cpu().numpy()
        
    def __len__(self):
        return self.examples_file.shape[0]

def copula_collate(batch):
    (x_left,x_right,y) = zip(*batch)
    x_left = torch.nn.utils.rnn.pad_sequence(x_left,batch_first=True,padding_value=0).transpose(1,2)
    x_right = torch.nn.utils.rnn.pad_sequence(x_right,batch_first=True,padding_value=0).transpose(1,2)
    y = torch.stack(y,dim=0)
    # print (len(y1_context))
    # print (len(y1_context[0]))
    return x_left,x_right,y.unsqueeze(2)
