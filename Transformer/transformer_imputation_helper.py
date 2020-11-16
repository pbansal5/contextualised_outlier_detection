import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
from scipy.stats import norm
import random
import math,copy
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)
        
class TransformerConvModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerConvModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Conv1d(1,ninp,kernel_size = 9, padding=4)
        self.ninp = ninp
        self.decoder_mean = nn.Linear(ninp, 1)
        self.decoder_std = nn.Linear(ninp, 1)
        self.mse_loss = nn.MSELoss()
        #self.init_weights()
        self.numerator = 0
        self.denominator = 0

    def reset(self):
        self.numerator = 0
        self.denominator = 0
        
    def init_weights(self):
        initrange = 0.1
        self.decoder_mean.bias.data.zero_()
        self.decoder_mean.weight.data.uniform_(-initrange, initrange)
        self.decoder_std.bias.data.zero_()
        self.decoder_std.weight.data.uniform_(-initrange, initrange)

    def forward(self, src,context_info):
        src = self.encoder(src.unsqueeze(1)).transpose(1,2)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        time = context_info['time']
        output_mean = self.decoder_mean(output).squeeze()[np.arange(len(time)),time]
        output_std = self.decoder_std(output).squeeze()[np.arange(len(time)),time]
        return output_mean,output_std
    
    def calc_loss_mle(self,y,y_pred,sigma_pred):
        err1 = (((y-y_pred)**2/torch.exp(sigma_pred)) + sigma_pred)
        return err1.mean()
    
    def calc_loss_org_mse(self,y,y_pred,std):
        temp = (((y_pred-y).cpu().numpy()*std)**2)
        return temp.sum()

    def add_calc_loss_org_mre(self,y,y_pred,std):
        self.numerator += np.abs((y_pred-y).cpu().numpy()).sum()
        self.denominator += np.abs((y).cpu().numpy()).sum()
        
    def compute_calc_loss_org_mre(self):
        temp = self.numerator/self.denominator
        self.reset()
        return temp
    
class TransformerDataset1D_(torch.utils.data.Dataset):
     
    def __init__(self,feats_file,examples_file):
        print ('loading dataset')
        self.feats = np.load(feats_file).astype(np.float32)
        print (self.feats.shape)
        self.examples_file = np.load(examples_file).astype(np.int)
        self.means = self.feats.mean(axis=0)
        self.stds = np.maximum(self.feats.std(axis=0),1e-7)
        #self.k = int(self.feats.shape[0]/10)
        
    def __getitem__(self,index):
        time_ = self.examples_file[index][0]
        shop_ = self.examples_file[index][1]
        
        inp_ = np.array(copy.deepcopy(self.feats[:,shop_]))        
        mean = inp_.mean()
        std = np.maximum(inp_.std(),1e-7)

        inp_ = (inp_-mean)/std
        temp = inp_[time_]
        
        indices = np.floor(np.random.sample(size = int(inp_.shape[0]/10))*inp_.shape[0])
        indices = indices.astype(np.int)
        inp_[time_] = 0
        inp_[indices] = 0        
                
        return torch.FloatTensor(inp_),torch.FloatTensor([temp]),time_,shop_

    def __len__(self):
        return self.examples_file.shape[0]

class TransformerDataset2D_(torch.utils.data.Dataset):
    def __init__(self,feats_file,examples_file):
        print ('loading dataset')
        self.feats = np.load(feats_file).astype(np.float32)
        print (self.feats.shape)
        self.examples_file = np.load(examples_file).astype(np.int)
        self.means = self.feats.mean(axis=0)
        self.stds = np.maximum(self.feats.std(axis=0),1e-7)
        #self.k = int(self.feats.shape[0]/10)
        
    def __getitem__(self,index):
        time_ = self.examples_file[index][0]
        shop_ = self.examples_file[index][1]
        prod_ = self.examples_file[index][2]
        
        inp_ = np.array(copy.deepcopy(self.feats[:,shop_,prod_]))
        mean = inp_.mean()
        std = np.maximum(inp_.std(),1e-7)

        inp_ = (inp_-mean)/std
        temp = inp_[time_]
        
        indices = np.floor(np.random.sample(size = int(inp_.shape[0]/10))*inp_.shape[0])
        indices = indices.astype(np.int)
        inp_[time_] = 0
        inp_[indices] = 0        
                
        return torch.FloatTensor(inp_),torch.FloatTensor([temp]),time_,shop_,prod_
        

    def __len__(self):
        return self.examples_file.shape[0]

# class TransformerDataset2D_(torch.utils.data.Dataset):
#     def __init__(self,feats_file,examples_file):
#         print ('loading dataset')
#         self.feats = np.load(feats_file).astype(np.float32)
#         print (self.feats.shape)
#         self.examples_file = np.load(examples_file).astype(np.int)
#         self.means = self.feats.mean(axis=0)
#         self.stds = np.maximum(self.feats.std(axis=0),1e-7)
#         #self.k = int(self.feats.shape[0]/10)
        
#     def __getitem__(self,index):
#         time_ = self.examples_file[index][0]
#         shop_ = self.examples_file[index][1]
#         prod_ = self.examples_file[index][2]
        
#         inp_ = np.array(copy.deepcopy(self.feats[:,shop_,prod_]))
#         mean = inp_.mean()
#         std = np.maximum(inp_.std(),1e-7)

#         inp_ = (inp_-mean)/std
#         temp = inp_[time_]
        
#         indices = np.floor(np.random.sample(size = int(inp_.shape[0]/10))*inp_.shape[0])
#         indices = indices.astype(np.int)
#         inp_[time_] = 0
#         inp_[indices] = 0        
                
#         return torch.FloatTensor(inp_),torch.FloatTensor([temp]),time_,shop_,prod_
        

#     def __len__(self):
#         return self.examples_file.shape[0]

def transformer_collate1d(batch):
    (inp_,out_,time,shop) = zip(*batch)
    return torch.stack(inp_,dim=0),torch.stack(out_,dim=0).squeeze(),dict({'time':torch.LongTensor(list(time)),'index1':torch.LongTensor(list(shop))})

def transformer_collate2d(batch):
    (inp_,out_,time,shop,prod) = zip(*batch)
    return torch.stack(inp_,dim=0),torch.stack(out_,dim=0).squeeze(),dict({'time':torch.LongTensor(list(time)),'index1':torch.LongTensor(list(shop)),'index2':torch.LongTensor(list(prod))})
        
class OutlierModel(nn.Module):
    def __init__(self,size1=266,size2=266,embedding_size=32):
        super(OutlierModel, self).__init__()
        hidden_dim = 64
        self.k = 20
        self.tau = 1
        self.embeddings1 = nn.Embedding(size1, embedding_size, max_norm=1)
        self.embeddings2 = nn.Embedding(size2, embedding_size, max_norm=1)
        self.outlier_layer1 = nn.Linear(8,hidden_dim)
        self.mean_outlier_layer = nn.Linear(hidden_dim,1)
        self.std_outlier_layer = nn.Linear(hidden_dim,1)
        self.residuals = None
        self.means = None
        self.stds = None
        self.numerator = 0
        self.denominator = 0

    def reset(self):
        self.numerator = 0
        self.denominator = 0

    def compute_feats(self,y_context1,y_context2,index1,index2):
        similarity1 = torch.cdist(self.embeddings1.weight[index1].unsqueeze(0),self.embeddings1.weight.unsqueeze(0),p=2).squeeze()+1e-3
        similarity2 = torch.cdist(self.embeddings2.weight[index2].unsqueeze(0),self.embeddings2.weight.unsqueeze(0),p=2).squeeze()+1e-3
        similarity1[np.arange(similarity1.shape[0]),index1] = 0
        similarity2[np.arange(similarity2.shape[0]),index2] = 0
        weights1 = torch.exp(-similarity1/self.tau)
        weights2 = torch.exp(-similarity2/self.tau)
        indices1 = torch.argsort(weights1)[:,-(self.k+1):-1]
        indices2 = torch.argsort(weights2)[:,-(self.k+1):-1]
        selected1 = y_context1[np.arange(y_context1.shape[0])[:,None],indices1]
        selected2 = y_context2[np.arange(y_context2.shape[0])[:,None],indices2]
        weights1 = weights1[np.arange(y_context1.shape[0])[:,None],indices1]
        weights2 = weights2[np.arange(y_context2.shape[0])[:,None],indices2]
        return torch.cat([(selected1*weights1).sum(dim=1,keepdim=True)/weights1.sum(dim=1,keepdim=True),weights1.sum(dim=1,keepdim=True),torch.std(selected1,dim=1,keepdim=True),
                          (selected2*weights2).sum(dim=1,keepdim=True)/weights2.sum(dim=1,keepdim=True),weights2.sum(dim=1,keepdim=True),torch.std(selected2,dim=1,keepdim=True)],dim=1)#.transpose(0,1)

    # def compute_feats(self,y_context1,y_context2,index1,index2):
    #     similarity1 = self.embeddings1.weight[index1]@self.embeddings1.weight.transpose(0,1)
    #     similarity2 = self.embeddings2.weight[index2]@self.embeddings2.weight.transpose(0,1)
    #     similarity1[np.arange(similarity1.shape[0]),index1] = 0
    #     similarity2[np.arange(similarity2.shape[0]),index2] = 0
    #     weights1 = similarity1
    #     weights2 = similarity2
    #     indices1 = torch.argsort(torch.abs(weights1))[:,-(self.k):]
    #     indices2 = torch.argsort(torch.abs(weights2))[:,-(self.k):]
    #     selected1 = y_context1[np.arange(y_context1.shape[0])[:,None],indices1]
    #     selected2 = y_context2[np.arange(y_context2.shape[0])[:,None],indices2]
    #     weights1 = weights1[np.arange(y_context1.shape[0])[:,None],indices1]
    #     weights2 = weights2[np.arange(y_context2.shape[0])[:,None],indices2]
    #     return torch.cat([(selected1*weights1).sum(dim=1,keepdim=True)/weights1.sum(dim=1,keepdim=True),torch.abs(weights1).sum(dim=1,keepdim=True),torch.std(selected1,dim=1,keepdim=True),
    #                       (selected2*weights2).sum(dim=1,keepdim=True)/weights2.sum(dim=1,keepdim=True),torch.abs(weights2).sum(dim=1,keepdim=True),torch.std(selected2,dim=1,keepdim=True)],dim=1)#.transpose(0,1)
    
    def forward(self,context_info):
        time = context_info['time']
        index1 = context_info['index1']
        index2 = context_info['index2']
        mean = self.means[time,index1,index2]
        std = self.stds[time,index1,index2]
        y2_context = self.residuals[time,index1,:]
        y1_context = self.residuals[time,:,index2]
        temp = self.compute_feats(y1_context,y2_context,index1,index2)
        feats = torch.cat([temp,mean.unsqueeze(1),std.unsqueeze(1)],axis=1)
        feats = self.outlier_layer1(feats).clamp(min=0)
        mean_outlier = self.mean_outlier_layer(feats).squeeze()
        std_outlier = self.std_outlier_layer(feats).squeeze()
        
        return mean_outlier,std_outlier
    
    def calc_loss_mle(self,y,y_pred,sigma_pred):
        err1 = (((y-y_pred)**2/torch.exp(sigma_pred)) + sigma_pred).mean()
        return err1
    
    def calc_loss_org_mse(self,y,y_pred,std):
        return (((y_pred-y).cpu().numpy()*std)**2).sum()
    
    def add_calc_loss_org_mre(self,y,y_pred,std):
        self.numerator += np.abs((y_pred-y).cpu().numpy()).sum()
        self.denominator += np.abs((y).cpu().numpy()).sum()
        
    def compute_calc_loss_org_mre(self):
        temp = self.numerator/self.denominator
        self.reset()
        return temp



# class AttentionModel(nn.Module):

#     def __init__(self, nseries ,ninp):
#         super(AttentionModel, self).__init__()
#         self.key = nn.Embedding(nseries,ninp)
#         self.query = nn.Embedding(nseries,ninp)
#         self.softmax = torch.nn.Softmax(dim=-1)
#         self.mse_loss = nn.MSELoss()
        
#     def forward(self, src,time):
#         key = self.key (torch.LongTensor(np.arange(src.shape[1])).cuda())
#         key = torch.stack([key for _ in range(src.shape[0])],dim=0)
#         query = self.query(time).unsqueeze(2)
#         weights = torch.bmm(key,query).squeeze()
#         weights = self.softmax(weights)
#         return (weights*src).sum(axis=1)
    
#     def calc_loss(self,y,y_pred):
#         #y = y[y!=0]
#         return self.mse_loss(y_pred,y.squeeze()).mean()
            
#     def val_calc_loss(self,y,y_pred,std):
#         #y = y[y!=0]
#         return (((y_pred-y).cpu().numpy()*std)**2).sum()
#         #return (((y_pred-y).cpu().numpy())**2).sum()

# class TransformerModel(nn.Module):

#     def __init__(self, nseries ,ninp, nhead, nhid, nlayers, dropout=0.5):
#         super(TransformerModel, self).__init__()
#         from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(ninp, dropout)
#         encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.k = 3
#         self.encoder = nn.Embedding(nseries,ninp-self.k)
#         self.ninp = ninp
#         self.decoder = nn.Linear(ninp, 1)
#         self.mse_loss = nn.MSELoss()
#         self.init_weights()
        
#     def init_weights(self):
#         initrange = 0.1
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)

#     def repeat (self,x,k):
#         out = [x for _ in range(k)]
#         out = torch.stack(out,dim=2)
#         return out
    
#     def forward(self, src,time):
#         encoder_output = self.encoder (torch.LongTensor(np.arange(src.shape[1])).cuda())
#         encoder_output = torch.stack([encoder_output for _ in range(src.shape[0])],dim=0)
#         # src = torch.cat([encoder_output,src.unsqueeze(2)],axis=2)
#         src = torch.cat([encoder_output,self.repeat(src,self.k)],axis=2)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src)
#         output = self.decoder(output)
#         return output.squeeze()

#     # def forward(self, src,time):
#     #     src = self.repeat(src,self.ninp)
#     #     src = self.pos_encoder(src)
#     #     output = self.transformer_encoder(src)
#     #     output = self.decoder(output)
#     #     return output.squeeze()

#     def calc_loss(self,y,y_pred):
#         y_pred[y==0] = 0
#         return self.mse_loss(y_pred,y).mean()
            
#     def val_calc_loss(self,y,y_pred,std):
#         y_pred[y==0] = 0
#         print (y_pred[y!=0],y[y!=0])
#         return (((y_pred-y).cpu().numpy()*std[:,None])**2).sum()
#         #return (((y_pred-y).cpu().numpy())**2).sum()
