import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import random,math
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
        mean_values, values  = self.sum_(out_series,context_info[1],context_info[2],mask)
        return {'mae':self.mae_loss(mean,out_series,context_info[2],mask),'sum':mean_values,'values':values}
    
    def sum_(self,y,mean,std,mask):
        temp = (y.cpu()*std.unsqueeze(1)+mean.unsqueeze(1)).cpu()
        #return temp[mask].mean(),temp[mask]
        return temp[mask].mean(),temp
    
    def mae_loss(self,y,y_pred,std,mask):
        temp = torch.abs((y_pred-y)*std.unsqueeze(1).cuda())
        return temp[mask].mean()


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self,feats,block_size=10,time_context=None):
        print ('loading dataset')
        self.feats = feats.astype(np.float32)
        self.num_dim = len(self.feats.shape)
        self.time_context = time_context
        self.means = np.nansum(self.feats,axis=0)
        self.mean_of_squared = np.nansum(self.feats*self.feats,axis=0)
        self.counts = (~np.isnan(self.feats)).sum(axis=0)
        self.block_size = block_size
        self.size = self.feats.shape[0]*self.feats.shape[1]
        
    def __getitem__(self,index):
        upper_limit = np.random.uniform(1,2*self.block_size,1).astype(np.int)[0]
        
        if (self.time_context != None):
            time_ = index%(self.feats.shape[0])
            tsNumber = int(index/self.feats.shape[0])
            lower_limit = min(time_,self.time_context)
            series = self.feats[time_-lower_limit:time_+self.time_context+upper_limit]
            time_ = lower_limit
        else :
            time_ = index%(self.feats.shape[0])
            tsNumber = int(index/self.feats.shape[0])
            series = self.feats

        if (np.isnan(self.feats[time_,tsNumber])):
            return None
        

        mean = self.means
        mean2 = self.mean_of_squared
        counts = self.counts
        
        series = series[:,tsNumber]
        mean = mean[tsNumber]
        mean2 = mean2[tsNumber]
        counts = counts[tsNumber]

        series = copy.deepcopy(series)
        
        #upper_limit = self.block_size
        
        temp = series[time_:time_+upper_limit]
        mean -= np.nansum(temp)
        mean2 -= np.nansum(temp*temp)
        counts -= (~np.isnan(temp).sum())

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

        context = [tsNumber]
        return torch.FloatTensor(series),torch.FloatTensor(out_series),torch.BoolTensor(mask>0),context,mean,std
        
    def __len__(self):
        return self.size


class ValidationTransformerDataset(torch.utils.data.Dataset):
    def __init__(self,feats,validation_feats,examples_file,block_size,time_context=None):
        print ('loading dataset')
        self.feats = feats.astype(np.float32)
        self.validation_feats = validation_feats.astype(np.float32)
        self.examples_file = examples_file.astype(np.int)
        self.time_context = time_context
        self.means = np.nanmean(self.feats,axis=0)
        self.stds = np.nanstd(self.feats,axis=0)
        
    def __getitem__(self,index):
        this_example = self.examples_file[index]
        time_ = this_example[0]

        if (self.time_context != None):
            lower_limit = min(time_,self.time_context)
            series = self.feats[time_-lower_limit:time_+self.time_context+this_example[2]]
            time_ = lower_limit
        else :
            series = self.feats

        out_series = self.validation_feats
        in_series = self.feats

        mean = self.means
        std = self.stds
        out_series = out_series[:,this_example[1]]
        in_series = in_series[:,this_example[1]]
        mean = mean[this_example[1]]
        std = std[this_example[1]]
        
        upper_limit = this_example[2]
        
        mean = np.nanmean(in_series)
        std = max(np.nanstd(in_series),1e-1)
        in_series = (in_series-mean)/std
        out_series = (out_series-mean)/std
        mask = np.zeros(out_series.shape)
        mask[time_:time_+upper_limit] = 1
        mask[np.isnan(out_series)] = 0
        in_series = np.nan_to_num(in_series)
        out_series = np.nan_to_num(out_series)

        #hardcoded for now
        context = [this_example[1]]
        return torch.FloatTensor(in_series),torch.FloatTensor(out_series),torch.BoolTensor(mask>0),context,mean,std
    
        
    def __len__(self):
        return self.examples_file.shape[0]
        

def transformer_collate(batch):
    batch = list(filter(None, batch)) 
    (series,out_series,mask,index,mean,std) = zip(*batch)
    return torch.stack(series,dim=0),torch.stack(out_series,dim=0),torch.stack(mask,dim=0),[torch.LongTensor(list(index)),torch.FloatTensor(list(mean)),torch.FloatTensor(list(std))]

def get_block_length(matrix):
    num_missing = len(np.where(np.isnan(matrix))[0])
    temp = matrix[:,0]
    num_blocks = 0
    for i in range(len(temp)):
        if (np.isnan(temp[i]) and ~np.isnan(temp[i+1])):
            num_blocks += 1
    num_blocks *= matrix.shape[1]
    return int(num_missing/num_blocks)

def make_validation (matrix,block_size):
    validation_points = np.random.uniform(0,matrix.shape[0]-block_size,(matrix.shape[1])).astype(np.int)
    train_matrix = copy.deepcopy(matrix)
    val_points = []
    test_points = []
    for i,x in enumerate(validation_points) :
        train_matrix[x:x+block_size,i] = np.nan
        val_points.append([x,i,block_size])
    for i in range(matrix.shape[1]):
        j =0
        while j < matrix.shape[0]:
            if (np.isnan(matrix[j][i])):
                time = 0
                while j < matrix.shape[0] and np.isnan(matrix[j][i]):
                    time+= 1
                    j += 1
                test_points.append([j-time,i,time])
            else :
                j += 1
    return train_matrix,matrix,np.array(val_points),np.array(test_points)

def train(model):
    best_state_dict = model.state_dict()
    best_loss = float('inf')
    writer = SummaryWriter(os.path.join('../Transformer/runs',log_file))

    lr = 1e-3
    optim = torch.optim.Adam(model.parameters(),lr=lr)

    max_epoch = 1
#    max_epoch = 200
    iteration = 0
    start_epoch = 0
    tolerance_epoch = 0
    patience  = 0
    for epoch in range(start_epoch,max_epoch):
        print ("Starting Epoch : %d"%epoch)

        for inp_,out_,mask,context_info in train_loader :
            loss = model(inp_.to(device),out_.to(device),mask.to(device),context_info)
            optim.zero_grad()
            loss['mae'].backward()
            optim.step()
            iteration += 1
            writer.add_scalar('training/mae_loss',loss['mae'],iteration)
            break
        
        if (epoch % 1 == 0):
            loss_mre_num,loss_mre_den,loss_crps = 0,0,0
            with torch.no_grad():
                for inp_,out_,mask,context_info in val_loader :
                    loss = model.validate(inp_.to(device),out_.to(device),mask.to(device),context_info)
                    loss_mre_num += loss['mae']*inp_.shape[0]
                    loss_mre_den += loss['sum']*inp_.shape[0]
                    break
                writer.add_scalar('validation/mre_loss',loss_mre_num/loss_mre_den,iteration)
                writer.add_scalar('validation/mae_loss',loss_mre_num/len(val_set),iteration)
                print ('done validation')
            if (float(loss_mre_num) < 0.95*best_loss):
                tolerance_epoch = 0
                best_loss = loss_mre_num
                best_state_dict = model.state_dict()
            else :
                tolerance_epoch += 1
                if (tolerance_epoch == patience):
                    print ('Early Stopping')
                    return best_state_dict

    return best_state_dict

def test(model):
    output_matrix = copy.deepcopy(val_feats)
    with torch.no_grad():
        for inp_,out_,mask,context_info in test_loader :
            loss = model.validate(inp_.to(device),out_.to(device),mask.to(device),context_info)
            output_matrix[:,context_info[0][:,0]] = np.where(mask.cpu().transpose(0,1),loss['values'].transpose(0,1),output_matrix[:,context_info[0][:,0]])
    return output_matrix


if __name__ == "__main__":

    log_file = 'transformer_janta_single'

    device = torch.device('cuda:%d'%0)
    batch_size = 256

    input_feats = np.load('../dataset/single_file_test_janta.npy')
    block_size = get_block_length(input_feats)
    train_feats,val_feats,val_points,test_points = make_validation(input_feats,block_size)

    if (input_feats.shape[0]<400):
        time_context = None
    else :
        time_context = 200

    train_set = TransformerDataset(train_feats,block_size,time_context = time_context)
    val_set = ValidationTransformerDataset(train_feats,val_feats,val_points,block_size,time_context = time_context)
    test_set = ValidationTransformerDataset(val_feats,np.zeros(val_feats.shape),test_points,block_size,time_context = time_context)

    train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)

    residuals = copy.deepcopy(train_set.feats)
    residuals -= np.nanmean(residuals,axis=0)
    residuals /= np.maximum(np.nanstd(residuals,axis=0),1e-1)
    residuals = torch.from_numpy(np.nan_to_num(residuals)).to(device)

    model = OurModel(sizes=[train_feats.shape[1]],nkernel=16,embedding_size=32,nhid=32,nlayers=4,nhead=2,residuals = residuals).to(device)
    model = torch.jit.script(model)

    best_state_dict = train(model)
    model.load_state_dict(best_state_dict)

    residuals = copy.deepcopy(val_feats)
    residuals -= np.nanmean(residuals,axis=0)
    residuals /= np.maximum(np.nanstd(residuals,axis=0),1e-1)
    residuals = torch.from_numpy(np.nan_to_num(residuals)).to(device)
    model.residuals = residuals
    
    matrix = test(model)
