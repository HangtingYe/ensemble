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
import pickle
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

def build_data_loader(dataset, batch_size=128, shuffle=False):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def get_loss_function(task_type):
    if task_type == 'regression': 
        loss_func = F.mse_loss
    elif task_type == 'binclass':
        loss_func = F.binary_cross_entropy_with_logits
    elif task_type == 'multiclass':
        loss_func = F.cross_entropy
    return loss_func

def run_one_epoch(encoder, head, data_loader, loss_func, optimizer=None):
   
    running_loss = 0.0

    for bid, (X_n, X_c, y) in enumerate(data_loader):
        
        pred = head(encoder(X_n, X_c))

        if loss_func == F.cross_entropy:
            loss = loss_func(pred, y)
        else:
            loss = loss_func(pred, y.reshape(-1,1))

        running_loss += loss.item()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return running_loss / len(data_loader)
    
def fit(encoder, head, train_loader, val_loader, loss_func, model_type, config, task_type, y_std, args):
    best_val_loss = 1e30
    best_encoder = None
    best_head = None

    optimizer = optim.AdamW(list(encoder.parameters())+list(head.parameters()), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    early_stop = config['training']['patience']
    epochs = config['training']['n_epochs']

    patience = early_stop

    for eid in range(epochs):
        encoder.train()
        head.train()
        train_loss = run_one_epoch(
            encoder, head, train_loader, loss_func, optimizer
        )

        encoder.eval()
        head.eval()
        val_loss = run_one_epoch(
            encoder, head, val_loader, loss_func
        )

        print(f'Epoch {eid}, train loss {train_loss}, val loss {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_encoder = copy.deepcopy(encoder)
            best_head = copy.deepcopy(head)
            patience = early_stop
        else:
            patience = patience - 1

        if patience == 0:
            break
    return best_encoder, best_head

def test(encoder, head, test_loader, task_type, y_std, args, config):
    
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

    np.save(open(f'./results/{args.dataname}_{args.model_type}_{args.pretrain}_{args.apply_kmeans}_{args.task_classes}_{task_ids}_{args.hyper}_{args.seed}.npy','wb'), score.item())

    # np.save(open(f'./results/{args.dataname}_{args.pretrain}_{args.norm_type}.npy','wb'), score.item())
    # torch.save(model.state_dict(), f'./results/{args.dataname}_{args.pretrain}_{args.norm_type}_model.pth')

def save_in_out(model, X, y, model_type, num):
    X_train, y_train = X['train'], y['train']
    X_val, y_val = X['val'], y['val']
    X_test, y_test = X['test'], y['test']
    pred_train = model(X_train)
    pred_val = model(X_val)
    pred_test = model(X_test)
    result = {'X_train':X_train, 'y_train':y_train, 'pred_train':pred_train, 'X_val':X_val, 'y_val':y_val, 'pred_val':pred_val,'X_test':X_test, 'y_test':y_test, 'pred_test':pred_test}
    np.save(open(f'./results_visual/{dataname}_{em_type}_{model_type}_{num}_result_visual.npy','wb'), result)


def generate_per_task(task_id, num_list, cat_list, X, categories):
    # !!! not X_ = X.copy(), dict of tensors must use clone() to copy each items, directly copy dict() still can lead to change X_ will change X, a view of level form up to down
    X_ = {k: v.clone() for k, v in X.items()}
    y_ = {k: v[:, task_id].clone() for k, v in X.items()}

    if task_id in cat_list:
        if int(max(torch.cat([y_['train'], y_['val'], y_['test']]))) > 1:
            task_classes = int(max(torch.cat([y_['train'], y_['val'], y_['test']]))) + 1
            y_ = {k: v.long() for k, v in y_.items()}
            task_type = 'multiclass'
        else:
            task_classes = 1
            y_ = {k: v.float() for k, v in y_.items()}
            task_type = 'binclass'
    elif task_id in num_list:
        task_classes = 1
        y_ = {k: v.float() for k, v in y_.items()}
        task_type = 'regression'

    for k, v in X_.items():
        if task_id in cat_list:
            # v[:, task_id] = v[:, task_id] - v[:, task_id] + sum(categories)
            v[:, task_id] = v[:, task_id] - v[:, task_id] + categories[task_id-len(num_list)]-1
        elif task_id in num_list:
            v[:, task_id] = v[:, task_id] - v[:, task_id]

    return X_, y_, task_type, task_classes


def generate_tasks(task_ids, num_list, cat_list, X, categories, info, config, args):
    n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
    _train, _val, task_types, task_classess = [], [], [], []
    for task_id in task_ids:
        X_, y_, task_type, task_classes = generate_per_task(task_id, num_list, cat_list, X, categories)
        _train.append(X_['train'][:,:n_num_features])
        _train.append(X_['train'][:,n_num_features:] if n_cat_features>0 else torch.empty(X_['train'].shape[0], X_['train'].shape[1]).cuda())
        _train.append(y_['train'])
        _val.append(X_['val'][:,:n_num_features])
        _val.append(X_['val'][:,n_num_features:] if n_cat_features>0 else torch.empty(X_['val'].shape[0], X_['val'].shape[1]).cuda())
        _val.append(y_['val'])
        task_types.append(task_type)
        task_classess.append(task_classes)
    # for unlimited feature lists, i.e. the length of X_train is unsure
    _train, _val = TensorDataset(*_train), TensorDataset(*_val)
    _trainloader, _valloader = build_data_loader(_train, config['training']['batch_size'], False), build_data_loader(_val, config['training']['batch_size'], False)
    dataloaders = {'_train':_trainloader, '_val':_valloader, 'task_types':task_types, 'task_classess':task_classess}
    return dataloaders


def generate_merge_task(task_ids, num_list, cat_list, X, categories, info, config, args):
    # !!! not X_ = X.copy(), dict of tensors must use clone() to copy each items, directly copy dict() still can lead to change X_ will change X, a view of level form up to down
    X_ = {k: v.clone() for k, v in X.items()}
    y_ = {k: v[:, task_ids].clone() for k, v in X.items()}

    for k, v in X_.items():
        for task_id in task_ids:
            if task_id in cat_list:
                # v[:, task_id] = v[:, task_id] - v[:, task_id] + sum(categories)
                v[:, task_id] = v[:, task_id] - v[:, task_id] + categories[task_id-len(num_list)]-1
            elif task_id in num_list:
                v[:, task_id] = v[:, task_id] - v[:, task_id]

    task_type = 'multiclass'

    y_merge = torch.cat([y_['train'], y_['val']], dim=0).data.cpu().numpy()
    kmeans = KMeans(n_clusters=args.task_classes)
    kmeans.fit(y_merge)
    y_merge = kmeans.labels_.astype('int64')
    
    y_merge = torch.tensor(y_merge).cuda()
    y_['train'] = y_merge[:y_['train'].shape[0]]
    y_['val'] = y_merge[y_['train'].shape[0]:]

    return X_, y_, task_type

def generate_tasks_merge(task_ids, num_list, cat_list, X, categories, info, config, args):
    n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
    _train, _val, task_types, task_classess = [], [], [], []
    X_, y_, task_type = generate_merge_task(task_ids, num_list, cat_list, X, categories, info, config, args)

    _train.append(X_['train'][:,:n_num_features])
    _train.append(X_['train'][:,n_num_features:] if n_cat_features>0 else torch.empty(X_['train'].shape[0], X_['train'].shape[1]).cuda())
    _train.append(y_['train'])

    _val.append(X_['val'][:,:n_num_features])
    _val.append(X_['val'][:,n_num_features:] if n_cat_features>0 else torch.empty(X_['val'].shape[0], X_['val'].shape[1]).cuda())
    _val.append(y_['val'])

    task_types.append(task_type)
    task_classess.append(args.task_classes)

    # for unlimited feature lists, i.e. the length of X_train is unsure
    _train, _val = TensorDataset(*_train), TensorDataset(*_val)
    _trainloader, _valloader = build_data_loader(_train, config['training']['batch_size'], False), build_data_loader(_val, config['training']['batch_size'], False)
    dataloaders = {'_train':_trainloader, '_val':_valloader, 'task_types':task_types, 'task_classess':task_classess}
    return dataloaders


def run_one_epoch_ssl(encoder, heads, ssl_dataloaders, optimizer=None):
    running_loss = 0.0

    if optimizer is not None:
        _loaders = ssl_dataloaders['_train']
    else:
        _loaders = ssl_dataloaders['_val']

    task_types = ssl_dataloaders['task_types']

    # for batch, each batch contains n subtasks  
    for bid, _all in enumerate(_loaders):
        # traverse each subtask
        loss = 0.0
        for i in range(len(task_types)):
            X_n, X_c, y = _all[i*3], _all[i*3+1], _all[i*3+2]
            head = heads[i]
            task_type = task_types[i]

            pred = head(encoder(X_n, X_c))
            loss_func = get_loss_function(task_type)
            # equally weight
            if loss_func == F.cross_entropy:
                loss += loss_func(pred, y)
            else:
                loss += loss_func(pred, y.reshape(-1,1))

        running_loss += loss.item()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return running_loss / len(_loaders)

def fit_ssl(encoder, heads, ssl_dataloaders):
    best_val_loss = 1e30
    best_encoder = None
    best_heads = None
    # combine encoder and heads to one module
    optimizer = optim.AdamW(list(encoder.parameters())+list(heads.parameters()), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    early_stop = config['training']['patience']
    epochs = config['training']['n_epochs']

    patience = early_stop

    for eid in range(epochs):
        encoder.train()
        heads.train()
        train_loss = run_one_epoch_ssl(encoder, heads, ssl_dataloaders, optimizer)

        encoder.eval()
        heads.eval()
        val_loss = run_one_epoch_ssl(encoder, heads, ssl_dataloaders)

        print(f'Epoch {eid}, train loss {train_loss}, val loss {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # best_encoder = encoder
            # best_heads = heads
            best_encoder = copy.deepcopy(encoder)
            best_heads = copy.deepcopy(heads)
            patience = early_stop
        else:
            patience = patience - 1

        if patience == 0:
            break
    return best_encoder


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':

    ## add hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--pretrain', type=str)
    parser.add_argument('--apply_kmeans', type=str, default=False)
    parser.add_argument('--task_classes', type=int, default=10)
    parser.add_argument('--task_ids', type=str, default='[0,1]')
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--hyper', type=str, default='default')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    _set_seed(args.seed)

    config = toml.load(f'./hypers_{args.hyper}/{args.dataname}/{args.model_type}.toml')

    with open(f'./data/{args.dataname}/info.json') as f:
        info = json.load(f)

    gc.collect()
    torch.cuda.empty_cache()

    X, y, n_classes, y_mean, y_std, categories = load(args.dataname, info, config['data']['normalization'], args.ratio)

    task_type = info.get('task_type')
    print(task_type)

    n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
    num_list = np.arange(n_num_features)
    cat_list = np.arange(n_num_features, n_num_features + n_cat_features) if n_cat_features!=None else None
   
    task_ids = ast.literal_eval(args.task_ids)

    ## generating pretrain data
    if args.apply_kmeans == 'True':
        ssl_dataloaders = generate_tasks_merge(task_ids, num_list, cat_list, X, categories, info, config, args)
    else:
        ssl_dataloaders = generate_tasks(task_ids, num_list, cat_list, X, categories, info, config, args)

    ## initializing models
    model = Model(n_num_features, args.model_type, n_classes if task_type == 'multiclass' else 1, 
    info=info, config = config, categories = categories)
    model.cuda()
    encoder = model.encoder
    head_final = model.head

    ## initializing heads specific for pretrain
    heads = [nn.Linear(model.head_input, task_classes) for task_classes in ssl_dataloaders['task_classess']]
    heads = nn.ModuleList(heads)
    heads.cuda()
    
    ## pretrain
    if args.pretrain == 'True':
        encoder = fit_ssl(encoder, heads, ssl_dataloaders)
    else:
        None

    ## loading finetune data
    train_loader = build_data_loader(TensorDataset(X['train'][:,:n_num_features], X['train'][:,n_num_features:] if n_cat_features>0 else torch.empty(X['train'].shape[0], X['train'].shape[1]).cuda(), y['train']), config['training']['batch_size'], False)
    val_loader = build_data_loader(TensorDataset(X['val'][:,:n_num_features], X['val'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['val'].shape[0], X['val'].shape[1]).cuda(), y['val']), config['training']['batch_size'], False)
    test_loader = build_data_loader(TensorDataset(X['test'][:, :n_num_features], X['test'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['test'].shape[0], X['test'].shape[1]).cuda(), y['test']), config['training']['batch_size'], False)

    ## finetune
    loss_func = get_loss_function(task_type)

    best_encoder, best_head = fit(encoder, head_final, train_loader, val_loader, loss_func, args.model_type, config, task_type, y_std, args)
    # model.load_state_dict(torch.load(f'./results/{dataname}_{em_type}_{model_type}_{num}_model.pth'))

    test(best_encoder, best_head, test_loader, task_type, y_std, args, config)
    # save_in_out(model, X, y, model_type, num)



