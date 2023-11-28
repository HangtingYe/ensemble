import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from data import load
from models import Model
import random
from scipy.optimize import linear_sum_assignment
import json
#import ipdb
import argparse
#import pickle
import sklearn
import scipy
import gc
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from scipy.linalg import svd
import importlib
import toml
import os
import copy
#import ot
import ast

from train import *

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
def test_multi_ratios(encoder, head, test_loader, task_type, y_std, args, config):
    
    encoder.eval()
    head.eval()

    pred = []
    ground = []
    for bid, (X_n, X_c, y) in enumerate(test_loader):
        pred.append(head(encoder(X_n, X_c)).data.cpu().numpy())
        ground.append(y)
    pred = np.concatenate(pred, axis=0)
    y = torch.cat(ground, dim=0)
    
    y = y.data.cpu().numpy()

    if task_type == 'binclass':
        pred = np.round(scipy.special.expit(pred))
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
    elif task_type == 'multiclass':
        pred = pred.argmax(1)
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(y.reshape(-1,1), pred.reshape(-1,1)) ** 0.5 * y_std

    print(f'test result, {score.item()}')
    task_ids = ast.literal_eval(args.task_ids)
    np.save(open(f'./results/{args.dataname}_{args.model_type}_{args.pretrain}_{args.apply_kmeans}_{args.task_classes}_{task_ids}_{args.hyper}_{args.seed}_{args.ratios}.npy','wb'), score.item())

    # np.save(open(f'./results/{args.dataname}_{args.pretrain}_{args.norm_type}.npy','wb'), score.item())
    # torch.save(model.state_dict(), f'./results/{args.dataname}_{args.pretrain}_{args.norm_type}_model.pth')
if __name__ == '__main__':

    ## add hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--pretrain', type=str)
    parser.add_argument('--apply_kmeans', type=str, default=False)
    parser.add_argument('--task_classes', type=int, default=10)
    parser.add_argument('--task_ids', type=str, default='[0,1]')
    parser.add_argument('--ratios', type=str, default='[0.5]')
    parser.add_argument('--hyper', type=str, default='default')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    _set_seed(args.seed)

    config = toml.load(f'./hypers_{args.hyper}/{args.dataname}/{args.model_type}.toml')

    with open(f'./data/{args.dataname}/info.json') as f:
        info = json.load(f)

    gc.collect()
    torch.cuda.empty_cache()


# -------------------------
    ratios = ast.literal_eval(args.ratios)
    X, y, n_classes, y_mean, y_std, categories = load(args.dataname, info, config['data']['normalization'], 1.0)

    task_type = info.get('task_type')
    print(task_type)

    n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
    num_list = np.arange(n_num_features)
    cat_list = np.arange(n_num_features, n_num_features + n_cat_features) if n_cat_features!=None else None
   
    task_ids = ast.literal_eval(args.task_ids)

    ## generating pretrain data
    if args.apply_kmeans == 'True':
        full_ssl_dataloaders = generate_tasks_merge(task_ids, num_list, cat_list, X, categories, info, config, args)
        all_sub_ssl_dataloaders = []
        for ratio in ratios:
            sub_X, _, _, _, _, _ = load(args.dataname, info, config['data']['normalization'], ratio)
            print(sub_X['train'].shape)
            all_sub_ssl_dataloaders.append(generate_tasks_merge(task_ids, num_list, cat_list, sub_X, categories, info, config, args))
    else:
        full_ssl_dataloaders = generate_tasks(task_ids, num_list, cat_list, X, categories, info, config, args)
        all_sub_ssl_dataloaders = []
        for ratio in ratios:
            sub_X, _, _, _, _, _ = load(args.dataname, info, config['data']['normalization'], ratio)
            print(sub_X['train'].shape)
            all_sub_ssl_dataloaders.append(generate_tasks(task_ids, num_list, cat_list, sub_X, categories, info, config, args))

    ## initializing models
    model = Model(n_num_features, args.model_type, n_classes if task_type == 'multiclass' else 1, 
    info=info, config = config, categories = categories)
    model.cuda()
    encoder = model.encoder
    head_final = model.head

    ## initializing heads specific for pretrains
    heads = [nn.Linear(model.head_input, task_classes) for task_classes in full_ssl_dataloaders['task_classess']]
    heads = nn.ModuleList(heads)
    heads.cuda()

    all_sub_sample_heads = []
    for sub_ssl_dataloaders in all_sub_ssl_dataloaders:
        sub_sample_heads = [nn.Linear(model.head_input, task_classes) for task_classes in sub_ssl_dataloaders['task_classess']]
        sub_sample_heads = nn.ModuleList(sub_sample_heads)
        sub_sample_heads.cuda()
        all_sub_sample_heads.append(sub_sample_heads)
    
    ## pretrain
    if args.pretrain == 'True':
        for sub_sample_heads, sub_ssl_dataloaders in zip(all_sub_sample_heads, all_sub_ssl_dataloaders):
            encoder = fit_ssl(encoder, sub_sample_heads, sub_ssl_dataloaders, config)
        encoder = fit_ssl(encoder, heads, full_ssl_dataloaders, config)
    else:
        None
# ------------------------
    ## loading finetune data
    train_loader = build_data_loader(TensorDataset(X['train'][:,:n_num_features], X['train'][:,n_num_features:] if n_cat_features>0 else torch.empty(X['train'].shape[0], X['train'].shape[1]).cuda(), y['train']), config['training']['batch_size'], False)
    val_loader = build_data_loader(TensorDataset(X['val'][:,:n_num_features], X['val'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['val'].shape[0], X['val'].shape[1]).cuda(), y['val']), config['training']['batch_size'], False)
    test_loader = build_data_loader(TensorDataset(X['test'][:, :n_num_features], X['test'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['test'].shape[0], X['test'].shape[1]).cuda(), y['test']), config['training']['batch_size'], False)

    ## finetune
    loss_func = get_loss_function(task_type)

    best_encoder, best_head = fit(encoder, head_final, train_loader, val_loader, loss_func, args.model_type, config, task_type, y_std, args)
    # model.load_state_dict(torch.load(f'./results/{dataname}_{em_type}_{model_type}_{num}_model.pth'))

    test_multi_ratios(best_encoder, best_head, test_loader, task_type, y_std, args, config)
    # save_in_out(model, X, y, model_type, num)



