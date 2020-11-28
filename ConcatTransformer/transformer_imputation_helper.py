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
import properscoring as ps

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
    def __init__(self, num_feats,ninp=64, nhead=2, nhid=64, nlayers=4,kernel_size=9,dropout=0.5,conv=True):
        super(TransformerConvModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        if (conv==True):
            self.encoder = nn.Conv1d(num_feats,ninp,kernel_size = kernel_size, padding=int((kernel_size-1)/2))
        else : 
            self.encoder = nn.Linear(num_feats,ninp)
        self.ninp = ninp
        self.conv = conv
        self.decoder_mean = nn.Linear(ninp, 1)
        self.decoder_std = nn.Linear(ninp, 1)
        self.mse_loss = nn.MSELoss()
        self.numerator = 0
        self.denominator = 0
        # self.init_weights() # changed
        
    def reset(self):
        self.numerator = 0
        self.denominator = 0
        
        
    def init_weights(self):
        initrange = 0.1
        self.decoder_mean.bias.data.zero_()
        self.decoder_mean.weight.data.uniform_(-initrange, initrange)
        self.decoder_std.bias.data.zero_()
        self.decoder_std.weight.data.uniform_(-initrange, initrange)

    def forward(self, src,time):
        if (self.conv):
            src = self.encoder(src.transpose(1,2)).transpose(1,2)
            src = self.pos_encoder(src)
        else : 
            src = self.encoder(src)
        output = self.transformer_encoder(src).clamp(min=0)
        return output[np.arange(len(time)),time,:]

class OurModel(nn.Module):
    def __init__(self,sizes,embedding_size=16,ninp=64,nlayers=4,nhid=32,nhead=2,kernel_size=9):
        super(OurModel, self).__init__()
        hidden_dim = 64
        self.k = 20
        self.tau = 1
        self.embeddings = []
        for x in sizes:
            self.embeddings.append(nn.Embedding(x, embedding_size, max_norm=1))
        self.transformer_embeddings = nn.ModuleList(self.embeddings)
        
        self.transformer_time = TransformerConvModel(num_feats=embedding_size*(len(sizes))+1,nlayers=nlayers,nhid=nhid,ninp=ninp,kernel_size=kernel_size,nhead=nhead)
        self.transformer_prod = TransformerConvModel(num_feats=embedding_size*(len(sizes))+1,nlayers=1,nhid=16,ninp=ninp,nhead=1,conv=False)
        self.transformer_shop = TransformerConvModel(num_feats=embedding_size*(len(sizes))+1,nlayers=1,nhid=16,ninp=ninp,nhead=1,conv=False)

        self.outlier_layer1 = nn.Linear((len(sizes)+1)*ninp,hidden_dim)
        self.mean_outlier_layer = nn.Linear(hidden_dim,1)
        self.std_outlier_layer = nn.Linear(hidden_dim,1)

        self.means = None
        self.stds = None
        self.normalised = None
        
    def core(self,context_info):
        output = self.normalised[context_info['time'],context_info['index'][:,0],context_info['index'][:,1]]
        time_ = self.normalised[:,context_info['index'][:,0],context_info['index'][:,1]].transpose(0,1)
        shop_ = self.normalised[context_info['time'],:,context_info['index'][:,1]]
        prod_ = self.normalised[context_info['time'],context_info['index'][:,0],:]

        time_[np.arange(context_info['time'].shape[0]),context_info['time']] = 0
        shop_[np.arange(context_info['time'].shape[0]),context_info['index'][:,0]] = 0
        prod_[np.arange(context_info['time'].shape[0]),context_info['index'][:,1]] = 0

        A_array = torch.rand(time_.shape) > 0.9
        A_array = A_array.to(time_.device)
        time_ = time_.masked_fill(A_array, 0)
        
        shop_embeddings = self.transformer_embeddings[0].weight[context_info['index'][:,0]].unsqueeze(1)
        prod_embeddings = self.transformer_embeddings[1].weight[context_info['index'][:,1]].unsqueeze(1)

        shop_input = torch.cat([shop_.unsqueeze(2),self.transformer_embeddings[0].weight.unsqueeze(0).repeat(shop_.shape[0],1,1),prod_embeddings.repeat(1,shop_.shape[1],1)],dim=2)
        prod_input = torch.cat([prod_.unsqueeze(2),shop_embeddings.repeat(1,prod_.shape[1],1),self.transformer_embeddings[1].weight.unsqueeze(0).repeat(prod_.shape[0],1,1)],dim=2)
        time_input = torch.cat([time_.unsqueeze(2),shop_embeddings.repeat(1,time_.shape[1],1),prod_embeddings.repeat(1,time_.shape[1],1)],dim=2)


        time_output = self.transformer_time(time_input,context_info['time'])
        shop_output = self.transformer_shop(shop_input,context_info['index'][:,0])
        prod_output = self.transformer_prod(prod_input,context_info['index'][:,1])
        output1 = torch.cat([time_output,shop_output,prod_output],dim=1)
        out1 = self.outlier_layer1(output1).clamp(min=0)
        mean = self.mean_outlier_layer(out1).squeeze()
        std = self.std_outlier_layer(out1).squeeze()
        return output,mean,std

    def forward (self,context_info):
        output,mean,std = self.core(context_info)
        return {'mae':self.mae_loss(mean,output,context_info['std']),'nll':self.nll_loss(mean,std,output),'var':std.mean()}
    
    def validate(self,context_info):
        output,mean,std = self.core(context_info)
        return {'mae':self.mae_loss(mean,output,context_info['std']),'nll':self.nll_loss(mean,std,output),'sum':self.sum_(output,context_info['mean'],context_info['std']),
                'crps':self.crps_loss(output,mean,std,context_info['mean'],context_info['std']),'var':std.mean()}

    def nll_loss(self,y_pred,sigma_pred,y):
        err1 = (((y-y_pred)**2/torch.exp(sigma_pred)) + sigma_pred)
        return err1.mean()

    def sum_(self,y,mean,std):
        temp = (y.cpu()*std+mean)
        return temp.mean()
    
    def mae_loss(self,y,y_pred,std):
        temp = torch.abs((y_pred-y)*std.cuda())
        return temp.mean()

    def crps_loss(self,x,mu_pred,sigma_pred,mean,std):
        mu_pred = mu_pred.cpu()*std + mean
        x = x.cpu()*std + mean
        sigma_pred = torch.exp(sigma_pred.cpu()/2)*std
        temp = [ps.crps_gaussian(x_, mu=mu_, sig=sigma_) for x_,mu_,sigma_ in zip(x,mu_pred,sigma_pred)]
        return sum(temp)/len(temp)


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self,feats_file,examples_file,time_context=None,mask=True):
        print ('loading dataset')
        self.feats = np.load(feats_file).astype(np.float32)
        self.residuals = np.zeros(self.feats.shape)
        self.num_dim = len(self.feats.shape)
        self.examples_file = np.load(examples_file).astype(np.int)
        self.time_context = time_context
        self.mask = mask
        
    def __getitem__(self,index):
        this_example = self.examples_file[index]
        time_ = this_example[0]
        mean = self.feats[time_].mean()
        std = self.feats[time_].std()
        #hardcoded for now
        if (self.num_dim == 3):
            return time_,[this_example[1],this_example[2]],mean,std
        # elif (self.num_dim == 2):
        #     return torch.FloatTensor(series),time_,[this_example[1]],mean[this_example[1]],std[this_example[1]]
        else :
            raise Exception
    def __len__(self):
        return self.examples_file.shape[0]


def transformer_collate(batch):
    (time_,index,mean,std) = zip(*batch)
    return dict({'time':torch.LongTensor(list(time_))  ,'index':np.array(list(index)).astype(np.int),'mean':torch.FloatTensor(list(mean)),'std':torch.FloatTensor(list(std))})


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

# class TransformerDataset1D_(torch.utils.data.Dataset):
     
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
        
#         inp_ = np.array(copy.deepcopy(self.feats[:,shop_]))        
#         mean = inp_.mean()
#         std = np.maximum(inp_.std(),1e-7)

#         inp_ = (inp_-mean)/std
#         temp = inp_[time_]
        
#         indices = np.floor(np.random.sample(size = int(inp_.shape[0]/10))*inp_.shape[0])
#         indices = indices.astype(np.int)
#         inp_[time_] = 0
#         inp_[indices] = 0        
                
#         return torch.FloatTensor(inp_),torch.FloatTensor([temp]),time_,shop_

#     def __len__(self):
#         return self.examples_file.shape[0]

    # def compute_feats_dot(self,y_context1,y_context2,index1,index2):
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
