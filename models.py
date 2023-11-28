import numpy as np
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from transformer import Net
import ipdb
from transformers import BertTokenizer, BertTokenizerFast
import os
import math
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch.nn.init as nn_init
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import json
from joblib import Parallel, delayed
import pandas as pd
from einops import rearrange, repeat
from sklearn.decomposition import PCA
import Models


class Model(nn.Module):
    def __init__(self, input_num, model_type, out_dim, info, config, categories) -> None:
        super().__init__()

        self.input_num = input_num ## number of numerical features
        self.out_dim = out_dim
        self.model_type = model_type
        self.info = info
        self.num_list = np.arange(info.get('n_num_features'))
        self.cat_list = np.arange(info.get('n_num_features'), info.get('n_num_features') + info.get('n_cat_features')) if info.get('n_cat_features')!=None else None
        self.categories = categories

        self.config = config

        self.build_model()



    def build_model(self):

        if self.model_type == 'MLP':

            self.encoder = Models.mlp.MLP(self.input_num, self.config['model']['d_layers'], self.config['model']['dropout'], self.out_dim, self.categories, self.config['model']['d_embedding'])

            self.head = nn.Linear(self.config['model']['d_layers'][-1], self.out_dim)
            self.head_input = self.config['model']['d_layers'][-1]


        elif self.model_type == 'SNN':
            
            self.encoder = Models.snn.SNN(self.input_num, self.config['model']['d_layers'], self.config['model']['dropout'], self.out_dim, self.categories, self.config['model']['d_embedding'])

            self.head = nn.Linear(self.config['model']['d_layers'][-1], self.out_dim)
            self.head_input = self.config['model']['d_layers'][-1]


        elif self.model_type == 'FTTransformer':

            self.encoder = Models.fttransformer.FTTransformer(self.input_num, self.categories, True, self.config['model']['n_layers'], self.config['model']['d_token'],
                            self.config['model']['n_heads'], self.config['model']['d_ffn_factor'], self.config['model']['attention_dropout'], self.config['model']['ffn_dropout'], self.config['model']['residual_dropout'],
                            self.config['model']['activation'], self.config['model']['prenormalization'], self.config['model']['initialization'], None, None)

            self.head = nn.Linear(self.config['model']['d_token'], self.out_dim)
            self.head_input = self.config['model']['d_token']


        elif self.model_type == 'ResNet':

            self.encoder = Models.resnet.ResNet(self.input_num, self.categories, self.config['model']['d_embedding'], self.config['model']['d'], self.config['model']['d_hidden_factor'], self.config['model']['n_layers'],
                            self.config['model']['activation'], self.config['model']['normalization'], self.config['model']['hidden_dropout'], self.config['model']['residual_dropout'])

            self.head = nn.Linear(self.config['model']['d'], self.out_dim)
            self.head_input = self.config['model']['d']

        
        elif self.model_type == 'DCN2':

            self.encoder = Models.dcn2.DCN2(self.input_num, self.config['model']['d'], self.config['model']['n_hidden_layers'], self.config['model']['n_cross_layers'],
                            self.config['model']['hidden_dropout'], self.config['model']['cross_dropout'], self.out_dim, self.config['model']['stacked'], self.categories, self.config['model']['d_embedding'])

            self.head = nn.Linear(self.config['model']['d'] if self.config['model']['stacked'] else 2 * self.config['model']['d'], self.out_dim)
            self.head_input = self.config['model']['d'] if self.config['model']['stacked'] else 2 * self.config['model']['d']


        elif self.model_type == 'AutoInt':

            self.encoder = Models.autoint.AutoInt(self.input_num, self.categories, self.config['model']['n_layers'], self.config['model']['d_token'], self.config['model']['n_heads'],
                            self.config['model']['attention_dropout'], self.config['model']['residual_dropout'], self.config['model']['activation'], self.config['model']['prenormalization'], self.config['model']['initialization'], 
                            None, None, self.out_dim)

            self.head = nn.Linear(self.config['model']['d_token'] * self.encoder.tokenizer.n_tokens, self.out_dim)
            self.head_input = self.config['model']['d_token'] * self.encoder.tokenizer.n_tokens


    def forward(self, inputs_n, inputs_c):
        inputs_ = self.encoder(inputs_n, inputs_c)
        return self.head(inputs_)
        