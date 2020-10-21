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
    def __init__(self,hidden_dim=64,rank=32):
        super(CopulaModel, self).__init__()
        self.num_layers = 2
        self.gru = nn.GRU(input_size=1,hidden_size=hidden_dim,num_layers = self.num_layers,bidirectional=False)
        self.mean = nn.Linear(hidden_dim,1)
        self.d = nn.Linear(hidden_dim,1)
        self.v = nn.Linear(hidden_dim,rank)

    def forward (self,x,y):
        mus = []
        vs = []
        ds = []
        for i in range(x.shape[1]):
            out,_ = self.gru(x[:,i,:].transpose(0,1).unsqueeze(axis=2))
            temp = out[-1,:,:]
            mus.append(self.mean(temp))
            vs.append(self.v(temp))
            ds.append(torch.log(1+torch.exp(self.d(temp))))
            
        mu = torch.stack(mus,axis=0).transpose(0,1)
        d = torch.stack(ds,axis=0).transpose(0,1)
        v = torch.stack(vs,axis=0).transpose(0,1)
        sigma = torch.bmm(v,v.transpose(1,2))+torch.diag_embed(d.squeeze())
        temp = torch.bmm((y-mu).transpose(1,2),torch.bmm(torch.inverse(sigma),y-mu)).squeeze() + torch.logdet(sigma)
        return temp.mean()

    def infer (self,x):
        out,_ = self.gru(x[0,:,:].unsqueeze(axis=2))
        temp = out[-1,:,:]
        return self.mean(temp),self.v(temp),torch.log(1+torch.exp(self.d(temp)))
        
class ForecastModel(nn.Module):
    def __init__(self,hidden_dim=64,rank=32):
        super(ForecastModel, self).__init__()
        self.num_layers = 2
        self.gru = nn.GRU(input_size=1,hidden_size=hidden_dim,num_layers = self.num_layers,bidirectional=False)
        self.mean = nn.Linear(hidden_dim,1)
        self.d = nn.Linear(hidden_dim,1)
        self.v = nn.Linear(hidden_dim,rank)
        revision_size = 32
        self.revision_network = nn.Sequential(nn.Linear(5,revision_size),nn.ReLU())
        self.mean_final = nn.Linear(revision_size,1)
        self.sigma_final = nn.Linear(revision_size,1)
        
    def forward (self,x,y):
        mus = []
        vs = []
        ds = []
        for i in range(x.shape[1]):
            out,_ = self.gru(x[:,i,:].transpose(0,1).unsqueeze(axis=2))
            temp = out[-1,:,:]
            mus.append(self.mean(temp))
            vs.append(self.v(temp))
            ds.append(self.d(temp))
            
        mu = torch.stack(mus,axis=0).transpose(0,1).squeeze()
        d = torch.stack(ds,axis=0).transpose(0,1).squeeze()
        v = torch.stack(vs,axis=0).transpose(0,1)
        v = torch.bmm(v,v.transpose(1,2))
        errors = (y.squeeze()-mu)/torch.exp(d/2)
        mus = []
        sigmas = []
        # print (mu.shape,d.shape,v.shape,errors.shape)
        for i in range(x.shape[1]):
            weighted_errors = (v[:,i]*errors[:,:]).sum(dim=1,keepdim=True)-(errors[:,i]*v[:,i,i])[:,None]
            relevance = torch.cat([v[:,i,:i],v[:,i,i+1:]],dim=1).sum(dim=1,keepdim=True)
            stdinerrors = torch.cat([errors[:,:i],errors[:,i+1:]],dim=1).std(dim=1,keepdim=True)
            feats = torch.cat([mu[:,i:i+1],d[:,i:i+1],weighted_errors,relevance,stdinerrors],dim=1)
            temp = self.revision_network(feats)
            mus.append(self.mean_final(temp))
            sigmas.append(self.sigma_final(temp))
        mu = torch.stack(mus,axis=0).transpose(0,1)
        log_sigma = torch.stack(sigmas,axis=0).transpose(0,1)
        sigma = torch.exp(log_sigma)
        sigma_inv = torch.diag_embed(1/sigma.squeeze())
        temp = torch.bmm((y-mu).transpose(1,2),torch.bmm(sigma_inv,y-mu)).squeeze() + log_sigma.sum()
        return temp.mean()

    def infer (self,x,y):
        y = y.transpose(0,1)
        out,_ = self.gru(x[0,:,:].unsqueeze(axis=2))
        temp = out[-1,:,:]
        mu,v,d = self.mean(temp),self.v(temp),self.d(temp)
        v = torch.matmul(v,v.transpose(0,1))
        errors = (y-mu)/torch.exp(d/2)
        weighted_errors = torch.matmul(v,errors)-errors*(v[np.arange(v.shape[0]),np.arange(v.shape[0])][:,None])
        relevance = v.sum(dim=1,keepdim=True) - v[np.arange(v.shape[0]),np.arange(v.shape[0])][:,None]
        stdinerrors = torch.zeros(*relevance.shape).cuda()
        stdinerrors[:,:] = float(errors.std())
        temp = self.revision_network(torch.cat([mu,d,weighted_errors,relevance,stdinerrors],dim=1))
        return calc_loss_our(y,self.mean_final(temp),torch.exp(self.sigma_final(temp)))
        

class CopulaDataset_(torch.utils.data.Dataset):
    def __init__(self,feats_file,end_time,useNormalised=False,k=10):
        print ('loading dataset')
        self.feats = np.load(feats_file).astype(np.float32)[:end_time,:]
        self.k = k
        self.end_time = end_time
        self.num_indices = self.feats.shape[1]
        self.normalised = np.zeros(self.feats.shape)
        self.useNormalised = useNormalised
        if(useNormalised == False):
            self.m = self.end_time
            self.probs = [norm.ppf(i/self.m) for i in range(1,self.m)]
            self.probsdelta = 1/((self.m**0.25)*4*np.sqrt(3.14*np.log(self.m)))
            self.probs.append(norm.ppf(1-self.probsdelta))
            self.gauss = np.zeros(self.feats.shape)
            temp = np.argsort(self.feats,axis=0)
            ranks = np.empty_like(temp)
            np.put_along_axis(ranks, temp, np.repeat(np.arange(temp.shape[0])[:,None],temp.shape[1],axis=1), axis=0)
            self.gauss = norm.ppf(ranks/self.feats.shape[0])
            self.gauss[self.gauss == float('-inf')] = 1/((self.m**0.25)*4*np.sqrt(3.14*np.log(self.m)))
            self.gauss[self.gauss == float('inf')] = 1-(1/((self.m**0.25)*4*np.sqrt(3.14*np.log(self.m))))

        mean = np.mean(self.feats,axis=0)
        std = np.maximum(np.std(self.feats,axis=0),1e-7)
        self.normalised = (self.feats-mean)/std
        
    def __getitem__(self,index):
        time_ = index%self.end_time
        idx = int(index/self.end_time)
        indices = np.random.randint(low=0,high=self.num_indices, size=self.k-1)
        indices = np.concatenate([indices,np.array([idx])],axis=0)
        out_ = []
        out_feats = []
        for x in indices :
            out_feats.append(torch.from_numpy(self.normalised[:time_,x]))
            if (self.useNormalised):
                out_.append(self.normalised[time_,x])
            else :
                out_.append(self.gauss[time_,x])
                
        out_ = torch.FloatTensor(out_)
        out_feats = torch.stack(out_feats,dim=0).transpose(0,1)
        
        return out_feats.float(),out_.float()
    
    def __len__(self):
        return self.end_time*self.num_indices

class ValidationCopulaDataset_(torch.utils.data.Dataset):
    def __init__(self,feats_file,start_time,useNormalised = False,rank = 32):
        print ('loading dataset')
        self.feats = np.load(feats_file).astype(np.float32)
        self.start_time = start_time
        self.mus = np.zeros(self.feats.shape).astype(np.float)
        self.vs = np.zeros((self.feats.shape[0],self.feats.shape[1],rank)).astype(np.float)
        self.ds = np.zeros(self.feats.shape).astype(np.float)
        self.normalised = np.zeros(self.feats.shape)
        self.useNormalised = useNormalised
        
        if (useNormalised == False):
            self.m = self.feats.shape[0]
            self.gauss = np.zeros(self.feats.shape)
            temp = np.argsort(self.feats,axis=0)
            ranks = np.empty_like(temp)
            np.put_along_axis(ranks, temp, np.repeat(np.arange(temp.shape[0])[:,None],temp.shape[1],axis=1), axis=0)
            self.gauss = norm.ppf(ranks/self.feats.shape[0])
            self.gauss[self.gauss == float('-inf')] = 1/((self.m**0.25)*4*np.sqrt(3.14*np.log(self.m)))
            self.gauss[self.gauss == float('inf')] = 1-(1/((self.m**0.25)*4*np.sqrt(3.14*np.log(self.m))))

        mean = np.mean(self.feats,axis=0)
        std = np.maximum(np.std(self.feats,axis=0),1e-7)
        self.normalised = (self.feats-mean)/std

    def __getitem__(self,time_):
        if (self.useNormalised):
            out_ = torch.FloatTensor(self.normalised[self.start_time+time_,:])
        else :
            out_ = torch.FloatTensor(self.gauss[self.start_time+time_,:])
            
        out_feats = torch.FloatTensor(self.normalised[:self.start_time+time_,:])
        return out_feats.float(),out_.float()
    
    def __len__(self):
        return self.feats.shape[0]-self.start_time

def copula_collate(batch):
    (x,y) = zip(*batch)
    x = torch.nn.utils.rnn.pad_sequence(x,batch_first=True,padding_value=0).transpose(1,2)
    y = torch.stack(y,dim=0)
    return x,y.unsqueeze(2)

def calc_loss_copula(y,mu,v,d):
    x_others = y.squeeze().data.cpu().numpy()[None,:]
    mu_others = mu.squeeze().data.cpu().numpy()[None,:]
    d_others_inv = np.diag(1/d.squeeze().data.cpu().numpy())
    v = v.squeeze().data.cpu().numpy()
    middle_factor = np.matmul(v.transpose(),np.matmul(d_others_inv,v))
    middle_factor += np.diag(np.ones(middle_factor.shape[0]))
    middle_factor = np.linalg.inv(middle_factor)
    other_factor = np.matmul(d_others_inv,v)
    sigma_others_inverse = d_others_inv - np.matmul(other_factor,np.matmul(middle_factor,other_factor.transpose()))

    _,logdet = np.linalg.slogdet(sigma_others_inverse)
    temp = np.matmul((x_others-mu_others),np.matmul(sigma_others_inverse,(x_others-mu_others)[0][:,None])).squeeze() - logdet
    return temp


def calc_loss_our(y,mu,d):
    x_others = y.squeeze().data.cpu().numpy()[None,:]
    mu_others = mu.squeeze().data.cpu().numpy()[None,:]
    sigma_others_inverse = np.diag(1/d.squeeze().data.cpu().numpy())

    _,logdet = np.linalg.slogdet(sigma_others_inverse)
    temp = np.matmul((x_others-mu_others),np.matmul(sigma_others_inverse,(x_others-mu_others)[0][:,None])).squeeze() - logdet
    return temp
