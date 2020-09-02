#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as F
#from model_utils import nll_categ_global, nll_gauss_global
#from EmbeddingMul import EmbeddingMul

def nll_categ_global(categ_logp_feat, input_idx_feat, cat_feat_size,
                     isRobust=False, w=None, isClean=False):

    # Normal
    if not isRobust:
        return F.nll_loss(categ_logp_feat, input_idx_feat, reduction='none').view(-1,1)

    # Robust
    w_r = w.view(-1,1)
    if isClean:
        return F.nll_loss(w_r*categ_logp_feat, input_idx_feat, reduction='none').view(-1,1)
    else:
        categ_logp_robust = torch.log(torch.tensor(1.0/cat_feat_size))
        categ_logp_robust = categ_logp_robust*torch.ones(categ_logp_feat.shape)
        categ_logp_robust = categ_logp_robust.type(categ_logp_feat.type())
        two_compnts = w_r*categ_logp_feat + (1-w_r)*categ_logp_robust

        return F.nll_loss(two_compnts, input_idx_feat, reduction='none').view(-1,1)


def nll_gauss_global(gauss_params, input_val_feat, logvar, isRobust=False, w=None,
                     std_0_scale=2, isClean=False, shape_feats=[1]):

    # Normal
    mu = gauss_params.view([-1] + shape_feats)
    logvar_r = (logvar.exp() + 1e-9).log()

    data_compnt = 0.5*logvar_r + (input_val_feat.view([-1] + shape_feats) - mu)**2 / (2.* logvar_r.exp() + 1e-9)

    if not isRobust:
        return data_compnt.view([-1] + shape_feats)

    # Robust
    w_r = w.view([-1] + shape_feats)
    if isClean:
        return (w_r*data_compnt).view([-1] + shape_feats)
    else:
        #Outlier model
        mu_0 = 0.0
        var_0 = torch.tensor(std_0_scale**2).type(gauss_params.type())
        robust_compnt = 0.5*torch.log(var_0) + (input_val_feat.view([-1] + shape_feats) - mu_0)**2 / (2.* var_0  + 1e-9)

        return (w_r*data_compnt + (1-w_r)*robust_compnt).view([-1] + shape_feats)


class EmbeddingMul(nn.Module):
    """This class implements a custom embedding mudule which uses matrix
    multiplication instead of a lookup. The method works in the functional
    way.
    Note: this class accepts the arguments from the original pytorch module
    but only with values that have no effects, i.e set to False, None or -1.
    """

    def __init__(self, depth, device):
        super(EmbeddingMul, self).__init__()
        # i.e the dictionnary size
        self.depth = depth
        self.device = device
        self.ones = torch.eye(depth, requires_grad=False, device=self.device)
        self._requires_grad = False
        # "oh" means One Hot
        self.last_oh = None
        self.last_weight = None

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value

    def forward(self, input, weight, padding_idx=None, max_norm=None,
                norm_type=2., scale_grad_by_freq=False, sparse=False, one_hot_input=True):
        """Declares the same arguments as the original pytorch implementation
        but only for backward compatibility. Their values must be set to have
        no effects.
        Args:
            - input: of shape (bptt, bsize)
            - weight: of shape (dict_size, emsize)
        Returns:
            - result: of shape (bptt, bsize, dict_size)
        """
        # ____________________________________________________________________
        # Checks if unsupported argument are used
        if padding_idx != -1:
            raise NotImplementedError(
                f"padding_idx must be -1, not {padding_idx}")
        # if max_norm is not None:
        #     raise NotImplementedError(f"max_norm must be None, not {max_norm}")
        if scale_grad_by_freq:
            raise NotImplementedError(f"scale_grad_by_freq must be False, "
                                      f"not {scale_grad_by_freq}")
        if sparse:
            raise NotImplementedError(f"sparse must be False, not {sparse}")
        # ____________________________________________________________________

        if self.last_oh is not None:
            del self.last_oh
        if one_hot_input:
            self.last_oh = input
        else:
            self.last_oh = self.to_one_hot(input)

        with torch.set_grad_enabled(self.requires_grad):
            result = torch.stack(
                [torch.mm(batch.float(), weight)
                 for batch in self.last_oh], dim=0)

            if max_norm is not None:
                # result = F.normalize(result, p=2, dim=-1)
                norm = result.norm(p=norm_type, dim=-1, keepdim=True)
                norm_mask = (norm > max_norm).float() # ).squeeze()
                result_new = result / norm * norm_mask + result * (1 - norm_mask)
                #result[:,norm_mask,:] = result[:,norm_mask,:].div(norm[:,norm_mask,:])
            else:
                result_new = result

        # self.last_weight = weight.clone() # NOTE: waste of memory?

        return result_new

    def to_one_hot(self, input):
        # Returns a new tensor that doesn't share memory
        result = torch.index_select(
            self.ones, 0, input.view(-1).long()).view(
            input.size()+(self.depth,))
        result.requires_grad = self.requires_grad
        return result

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

    
class VAE(nn.Module):
    def __init__(self):

        super(VAE, self).__init__()

        feats = 3
        embedding_size = 50
        layer_size = 400
        latent_size = 5

        self.feat_info = [["time",'categ',745],['pulocation','categ',266],['dozone','categ',7],['cnt','real',1]]
        self.size_input = feats*50+1
        self.size_output = feats + 1
        self.alpha = 0.95
        self.gauss = 2
        ## Encoder Params

        # define a different embedding matrix for each feature
        self.feat_embedd = nn.ModuleList([nn.Embedding(c_size, embedding_size, max_norm=1)
                                          for _, col_type, c_size in self.feat_info
                                          if col_type=="categ"])

        self.fc1 = nn.Linear(self.size_input, layer_size)
        self.fc21 = nn.Linear(layer_size, latent_size)
        self.fc22 = nn.Linear(layer_size, latent_size)

        ## Decoder Params

        self.fc3 = nn.Linear(latent_size,layer_size)

        self.out_cat_linears = nn.ModuleList([nn.Linear(layer_size, c_size) if col_type=="categ"
                                              else nn.Linear(layer_size, c_size)
                                              for _, col_type, c_size in self.feat_info])

        self.logvar_x = nn.Parameter(torch.zeros(1,1).float())

        ## Other

        self.activ = nn.ReLU()

        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # define encoder / decoder easy access parameter list
        encoder_list = [self.fc1, self.fc21, self.fc22]
        self.encoder_mod = nn.ModuleList(encoder_list)
        if self.feat_embedd:
            self.encoder_mod.append(self.feat_embedd)

        self.encoder_param_list = nn.ParameterList(self.encoder_mod.parameters())

        decoder_list = [self.fc3, self.out_cat_linears]
        self.decoder_mod = nn.ModuleList(decoder_list)
        self.decoder_param_list = nn.ParameterList(self.decoder_mod.parameters())
        if len(self.logvar_x):
            self.decoder_param_list.append(self.logvar_x)


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

    def encode(self, x_data):
        q_params = dict()
        input_values = self.get_inputs(x_data)
        fc1_out = self.fc1(input_values)
        h1_qz = self.activ(fc1_out)
        q_params['z'] = {'mu': self.fc21(h1_qz), 'logvar': self.fc22(h1_qz)}
        return q_params

    def sample_normal(self, q_params_z):
        if self.training:
            eps = torch.randn_like(q_params_z['mu'])
            std = q_params_z['logvar'].mul(0.5).exp_()
            return eps.mul(std).add_(q_params_z['mu'])
        else:
            return q_params_z['mu']

    def reparameterize(self, q_params):
        q_samples = dict()
        q_samples['z'] = self.sample_normal(q_params['z'])
        return q_samples

    def decode(self, z):
        p_params = dict()
        h3 = self.activ(self.fc3(z))
        out_cat_list = []

        for feat_idx, out_cat_layer in enumerate(self.out_cat_linears):
            if self.feat_info[feat_idx][1] == "categ": # coltype check
                out_cat_list.append(self.logSoftmax(out_cat_layer(h3)))
            elif self.feat_info[feat_idx][1] == "real":
                out_cat_list.append(out_cat_layer(h3))

        # tensor with dims (batch_size, self.size_output)
        p_params['x'] = torch.cat(out_cat_list, 1)
        p_params['logvar_x'] = self.logvar_x.clamp(-3,3)
        return p_params

    def forward(self, x_data, n_epoch=None):
        q_params = self.encode(x_data)
        q_samples = self.reparameterize(q_params)
        return self.decode(q_samples['z']), q_params, q_samples

    def loss_function(self, input_data, p_params, q_params, q_samples, clean_comp_only=False, data_eval_clean=False):

        """ ELBO: reconstruction loss for each variable + KL div losses summed over elements of a batch """

        dtype_float = torch.cuda.FloatTensor
        nll_val = torch.zeros(1).type(dtype_float)
        # mixed datasets, or just categorical / continuous with medium number of features
        start = 0
        cursor_num_feat = 0

        for feat_select, (_, col_type, feat_size) in enumerate(self.feat_info):
            pi_feat = torch.sigmoid(q_params['w']['logit_pi'][:,feat_select]).clamp(1e-6, 1-1e-6)
            
            if clean_comp_only and data_eval_clean:
                pi_feat = torch.ones_like(q_params['w']['logit_pi'][:,feat_select])
                    
            # compute NLL
            if col_type == 'categ':
                nll_val += nll_categ_global(p_params['x'][:,start:(start + feat_size)],
                                            input_data[:,feat_select].long(), feat_size, isRobust=True,
                                            w=pi_feat, isClean=clean_comp_only).sum()
                start += feat_size
            elif col_type == 'real':
                nll_val += nll_gauss_global(p_params['x'][:,start:(start + 1)], # 2
                                            input_data[:,feat_select],
                                            p_params['logvar_x'][:,cursor_num_feat], isRobust=True,
                                            w=pi_feat, isClean=clean_comp_only, 
                                            std_0_scale=self.gauss).sum()
                start += 1 # 2
                cursor_num_feat +=1


        # kld regularizer on the latent space
        z_kld = -0.5 * torch.sum(1 + q_params['z']['logvar'] - q_params['z']['mu'].pow(2) - q_params['z']['logvar'].exp())

        # prior on clean cells (higher values means more likely to be clean)
        prior_sig = torch.tensor(self.alpha).type(dtype_float)

        # kld regularized on the weights
        pi_mtx = torch.sigmoid(q_params['w']['logit_pi']).clamp(1e-6, 1-1e-6)
        w_kld = torch.sum(pi_mtx * torch.log(pi_mtx / prior_sig) + (1-pi_mtx) * torch.log((1-pi_mtx) / (1-prior_sig)))

        loss_ret = nll_val + z_kld if clean_comp_only else nll_val + z_kld + w_kld

        return loss_ret, nll_val, z_kld, w_kld 



    
