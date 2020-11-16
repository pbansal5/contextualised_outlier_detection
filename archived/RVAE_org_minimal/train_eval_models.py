#!/usr/bin/env python3

import torch
from torch import optim
import torch.nn.functional as F

import argparse
from sklearn.metrics import mean_squared_error
import numpy as np
import json

from model_utils import get_pi_exact_vec, rnn_vae_forward_one_stage, rnn_vae_forward_two_stage


def training_phase(model, optimizer, train_loader,  step,writer):

    model.train()

    train_loss_vae, train_nll_vae, train_z_kld_vae, train_w_kld_vae = 4*[0]
    train_loss_seq, train_nll_seq, train_z_kld_seq, train_w_kld_seq = 4*[0]

    train_total_loss_seq_vae, train_loss_seq_vae, train_nll_seq_vae, train_z_kld_seq_vae, train_w_kld_seq_vae = 5*[0]

    for batch_idx, (data_input,_) in enumerate(train_loader):

        data_input = data_input.cuda()
        optimizer.zero_grad()

        p_params, q_params, q_samples = model(data_input)

        get_pi_exact_vec(model, data_input, p_params, q_params, logit_ret=True)

        vae_loss, vae_nll, vae_z_kld, vae_w_kld = model.loss_function(data_input, p_params, q_params, q_samples)

        writer.add_scalar('train/loss_vae',vae_loss.item(),step)
        writer.add_scalar('train/nll_vae',vae_nll.item(),step)
        writer.add_scalar('train/x_kld_vae',vae_z_kld.item(),step)
        writer.add_scalar('train/w_kld_vae',vae_w_kld.item(),step)
        step += 1
        vae_loss.backward()

        optimizer.step()


    return step

