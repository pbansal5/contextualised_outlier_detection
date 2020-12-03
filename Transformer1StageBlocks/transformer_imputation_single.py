import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import random
import torch.nn.functional as F
from typing import Dict, List, Tuple
import os,copy

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)

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
    def __init__(self,ninp=64, nhead=2, nhid=64, nlayers=4,dropout=0.5):
        super(TransformerConvModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
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

    def forward(self, src):
        src = self.pos_encoder(src.transpose(1,2))
        output = self.transformer_encoder(src).clamp(min=0)
        return output

class OurModel(nn.Module):
    def __init__(self,sizes,embedding_size=16,nkernel=64,nlayers=4,nhid=32,nhead=2,kernel_size=9,residuals=None):
        super(OurModel, self).__init__()
        hidden_dim = 64
        self.k = 20
        self.tau = 1
        self.embeddings = []
        for x in sizes:
            self.embeddings.append(nn.Embedding(x, embedding_size))
        self.transformer_embeddings = nn.ModuleList(self.embeddings)
        self.conv = nn.Conv1d(1,nkernel,kernel_size = kernel_size, padding=int((kernel_size-1)/2))
        num_feats = embedding_size*len(sizes)+nkernel
        self.transformer = TransformerConvModel(nlayers=nlayers,nhid=nhid,ninp=num_feats,nhead=nhead)
        self.outlier_layer1 = nn.Linear(3*len(sizes)+num_feats,hidden_dim)
        self.mean_outlier_layer = nn.Linear(hidden_dim,1)
        self.std_outlier_layer = nn.Linear(hidden_dim,1)
        self.residuals = residuals
        self.sigmoid = torch.nn.Sigmoid()
        
    def compute_feats(self,y_context : List[torch.Tensor],indices):
        final1,final2,final3 = [],[],[]
        for i,embed in enumerate(self.transformer_embeddings):
            temp = torch.cdist(embed.weight[indices[i]].unsqueeze(0),embed.weight.unsqueeze(0),p=2.0).squeeze()+1e-3
            temp[torch.arange(temp.shape[0]),indices[i]] = 0
            temp_w = torch.exp(-temp/self.tau)
            temp_i = torch.argsort(temp_w)[:,-(self.k+1):-1]
            temp_s = y_context[i][torch.arange(y_context[i].shape[0])[:,None],:,temp_i]
            temp_w = temp_w[torch.arange(temp_w.shape[0])[:,None],None,temp_i].repeat(1,1,y_context[0].shape[1])
            temp_w[temp_s==0] = temp_w[temp_s==0]/1e3
            final1.append(temp_w.sum(dim=1,keepdim=True))
            final2.append((temp_w*temp_s).sum(dim=1,keepdim=True)/temp_w.sum(dim=1,keepdim=True))
            final3.append(torch.std(temp_s,dim=1,keepdim=True))

        return torch.cat(final1+final2+final3,dim=1).transpose(1,2)
    
    def core(self,in_series,out_series,context_info : List[torch.Tensor]):
        embeddings = [self.conv(in_series.unsqueeze(1))]

        for i,embed in enumerate(self.transformer_embeddings):
            temp = embed.weight[context_info[0][:,i]]
            embeddings.append(temp.unsqueeze(2).repeat(1,1,in_series.shape[1]))
            
        series = torch.cat(embeddings,dim=1)
        hidden_state = self.transformer(series)

        if (context_info[0].shape[1] == 2):
            residuals = [self.residuals[:,:,context_info[0][:,1]].transpose(1,2).transpose(0,1),self.residuals[:,context_info[0][:,0],:].transpose(0,1)]
        elif (context_info[0].shape[1] == 1):
            residuals = [self.residuals[:,:].unsqueeze(0).repeat(series.shape[0],1,1)]
        else :
            raise Exception
        
        temp = self.compute_feats(residuals,context_info[0].transpose(0,1))
        feats = torch.cat([temp,hidden_state],dim=2)
        feats = self.outlier_layer1(feats).clamp(min=0)
        mean = self.mean_outlier_layer(feats).squeeze()
        std = self.std_outlier_layer(feats).squeeze()
        return mean,std

    def forward (self,in_series,out_series,mask,context_info : List[torch.Tensor]):
        mean,std = self.core(in_series,out_series,context_info)
        return {'mae':self.mae_loss(mean,out_series,context_info[2],mask)}
    
    @torch.jit.export
    def validate(self,in_series,out_series,mask,context_info  : List[torch.Tensor]):
        mean,std = self.core(in_series,out_series,context_info)
        return {'mae':self.mae_loss(mean,out_series,context_info[2],mask),'sum':self.sum_(out_series,context_info[1],context_info[2],mask)}
    
    def sum_(self,y,mean,std,mask):
        temp = (y.cpu()*std.unsqueeze(1)+mean.unsqueeze(1))*mask.cpu()
        return temp[mask].mean()
    
    def mae_loss(self,y,y_pred,std,mask):
        temp = torch.abs((y_pred-y)*std.unsqueeze(1).cuda())*mask
        return temp[mask].mean()


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self,feats_file,examples_file,time_context=None):
        print ('loading dataset')
        self.feats = np.load(feats_file).astype(np.float32)
        self.examples_file = np.load(examples_file).astype(np.int)
        self.num_dim = len(self.feats.shape)
        self.time_context = time_context
        self.means = np.nansum(self.feats,axis=0)
        self.mean_of_squared = np.nansum(self.feats*self.feats,axis=0)
        self.counts = (~np.isnan(self.feats)).sum(axis=0)

        
    def __getitem__(self,index):
        this_example = self.examples_file[index]
        time_ = this_example[0]
        if (self.time_context != None):
            series = self.feats[time_-self.time_context:time_+self.time_context+1]
            time_ = self.time_context
        else :
            series = self.feats

        mean = self.means
        mean2 = self.mean_of_squared
        counts = self.counts
        for i in range(1,self.num_dim):
            series = series[:,this_example[i]]
            mean = mean[this_example[i]]
            mean2 = mean2[this_example[i]]
            counts = counts[this_example[i]]

        series = copy.deepcopy(series)
        
        upper_limit = np.random.uniform(5,15,1).astype(np.int)[0]
        #upper_limit = 10
        
        temp = series[time_:time_+upper_limit]
        mean -= np.nansum(temp)
        mean2 -= np.nansum(temp*temp)
        counts -= (10-np.isnan(temp).sum())

        out_series = copy.deepcopy(series)
        series[time_:time_+upper_limit] = np.nan

        mean /= counts
        std = np.sqrt((mean2/counts)-mean*mean)
        std = max(std,1e-1)
        series = (series-mean)/std
        out_series = (out_series-mean)/std
        
        mask = np.zeros(out_series.shape)
        mask[time_:time_+upper_limit] = 1
        mask[np.isnan(out_series)] = 0
        
        series = np.nan_to_num(series)
        out_series = np.nan_to_num(out_series)

        #hardcoded for now
        if (self.num_dim == 3):
            context = [this_example[1],this_example[2]]
        elif (self.num_dim == 2):
            context = [this_example[1]]
        else :
            raise Exception
        return torch.FloatTensor(series),torch.FloatTensor(out_series),torch.BoolTensor(mask>0),context,mean,std
    
        
    def __len__(self):
        return self.examples_file.shape[0]


class ValidationTransformerDataset(torch.utils.data.Dataset):
    def __init__(self,feats_file,validation_feats_file,examples_file,time_context=None):
        print ('loading dataset')
        self.feats = np.load(feats_file).astype(np.float32)
        self.validation_feats = np.load(validation_feats_file).astype(np.float32)
        self.examples_file = np.load(examples_file).astype(np.int)
        self.num_dim = len(self.feats.shape)
        self.time_context = time_context
        self.means = np.nanmean(self.feats,axis=0)
        self.stds = np.nanstd(self.feats,axis=0)
        
    def __getitem__(self,index):
        this_example = self.examples_file[index]
        time_ = this_example[0]
        out_series = self.validation_feats
        in_series = self.feats

        mean = self.means
        std = self.stds
        for i in range(1,self.num_dim):
            out_series = out_series[:,this_example[i]]
            in_series = in_series[:,this_example[i]]
            mean = mean[this_example[i]]
            std = std[this_example[i]]
            
        upper_limit = 10
        
        mean = np.nanmean(in_series)
        std = max(np.nanstd(in_series),1e-1)
        in_series = (in_series-mean)/std
        out_series = (out_series-mean)/std
        mask = np.zeros(out_series.shape)
        mask[time_:time_+upper_limit] = 1
        in_series = np.nan_to_num(in_series)
        out_series = np.nan_to_num(out_series)

        #hardcoded for now
        if (self.num_dim == 3):
            context = [this_example[1],this_example[2]]
        elif (self.num_dim == 2):
            context = [this_example[1]]
        else :
            raise Exception
        return torch.FloatTensor(in_series),torch.FloatTensor(out_series),torch.BoolTensor(mask>0),context,mean,std
    
        
    def __len__(self):
        return self.examples_file.shape[0]
        

def transformer_collate(batch):
    (series,out_series,mask,index,mean,std) = zip(*batch)
    return torch.stack(series,dim=0),torch.stack(out_series,dim=0),torch.stack(mask,dim=0),[torch.LongTensor(list(index)),torch.FloatTensor(list(mean)),torch.FloatTensor(list(std))]

log_file = 'transformer_janta_1stage_block_siblings_fast_by_bigbatch_highlr_noembedinconv'
#log_file = 'transformer_janta_dummy'

device = 0
device = torch.device('cuda:%d'%device)
batch_size = 256
lr = 1e-3


train_set = TransformerDataset('../dataset/2d_block_jantahack_train.npy','../dataset/2d_block_jantahack_train_examples.npy')
val_set = ValidationTransformerDataset('../dataset/2d_block_jantahack_train.npy','../dataset/2d_block_jantahack_test.npy','../dataset/2d_block_jantahack_test_examples.npy')

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)
val_loader = torch.utils.data.DataLoader(val_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)

residuals = copy.deepcopy(train_set.feats)
residuals -= np.nanmean(residuals,axis=0)
residuals /= np.maximum(np.nanstd(residuals,axis=0),1e-1)
residuals = torch.from_numpy(np.nan_to_num(residuals)).to(device)

model = OurModel(sizes=[76,28],nkernel=16,embedding_size=16,nhid=32,nlayers=4,nhead=2,residuals = residuals).to(device)
#model = OurModel(sizes=[76,28],nkernel=16,embedding_size=16,nhid=16,nlayers=2,nhead=1,residuals = residuals).to(device)
model = torch.jit.script(model)

best_state_dict = model.state_dict()
best_loss = float('inf')
writer = SummaryWriter(os.path.join('../Transformer/runs',log_file))


print ("Starting Stage 1")

optim = torch.optim.Adam(model.parameters(),lr=lr)

max_epoch = 200
iteration = 0
start_epoch = 0

for epoch in range(start_epoch,max_epoch):
    print ("Starting Epoch : %d"%epoch)

    for inp_,out_,mask,context_info in train_loader :
        loss = model(inp_.to(device),out_.to(device),mask.to(device),context_info)
        optim.zero_grad()
        #loss['nll'].backward()
        loss['mae'].backward()
        optim.step()
        iteration += 1
        writer.add_scalar('training/nll_loss',loss['nll'],iteration)
        writer.add_scalar('training/mae_loss',loss['mae'],iteration)
        writer.add_scalar('training/variance',loss['var'],iteration)

    if (epoch % 1 == 0):
        loss_mre_num,loss_mre_den,loss_crps = 0,0,0
        with torch.no_grad():
            for inp_,out_,mask,context_info in val_loader :
                loss = model.validate(inp_.to(device),out_.to(device),mask.to(device),context_info)
                loss_mre_num += loss['mae']*inp_.shape[0]
                loss_mre_den += loss['sum']*inp_.shape[0]
                loss_crps += loss['crps']*inp_.shape[0]
            writer.add_scalar('validation/mre_loss',loss_mre_num/loss_mre_den,iteration)
            writer.add_scalar('validation/mae_loss',loss_mre_num/len(val_set),iteration)
            writer.add_scalar('validation/crps_loss',loss_crps/len(val_set),iteration)
            
        if (float(loss_crps) < best_loss):
            best_loss = loss_crps
            best_state_dict = model.state_dict()
            
    print ('done validation')
