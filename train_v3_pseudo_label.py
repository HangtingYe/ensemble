import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from data import load
from models_v3 import Model
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
import ipdb
import argparse
import pickle
import sklearn
import scipy
import gc
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from scipy.linalg import svd
import importlib
import toml
import os
import copy
import ot
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

def cosine_dist(a:torch.tensor,b:torch.tensor):
    assert a.shape == b.shape
    dot = torch.matmul(a,b.transpose(0,1))
    a_ = torch.norm(a,p=2,dim=1,keepdim=True)
    b_ = torch.norm(b,p=2,dim=1,keepdim=True)
    return 1 - dot/(torch.matmul(a_,b_.T))

def run_one_epoch(model, data_loader, loss_func, optimizer=None):
   
    running_loss = 0.0
    for bid, (X_n, X_c, y) in enumerate(data_loader):
        r_hid = model.estimator_encoder(X_n, X_c)
        r = model.estimator_head(r_hid)
        r = torch.softmax(r, dim=1)
        hid = torch.mm(r.float(), model.topics.float())

        pred = model.head_estimator(hid)
        # pred loss
        if loss_func == F.cross_entropy:
            loss = loss_func(pred, y)
        else:
            loss = loss_func(pred, y.reshape(-1,1))

        # independence loss
        if model.args.orthogonal_weight > 0.0:
            r_1 = torch.sqrt(torch.sum(model.topics.float()**2,dim=1,keepdim=True))
            topic_metrix = torch.mm(model.topics.float(), model.topics.T.float()) / torch.mm(r_1, r_1.T)
            topic_metrix = torch.clamp(topic_metrix.abs(), 0, 1)

            l1 = torch.sum(topic_metrix.abs())
            l2 = torch.sum(topic_metrix ** 2)

            loss_sparse = l1 / l2
            loss_constraint = torch.abs(l1 - topic_metrix.shape[0])

            r_loss = loss_sparse + 0.5*loss_constraint
            
            loss = loss + r_loss*model.args.orthogonal_weight

        running_loss += loss.item()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return running_loss / len(data_loader)
    
def fit(model, train_loader, val_loader, loss_func, model_type, config, task_type, y_std, args):
    best_val_loss = 1e30
    best_model = None

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    early_stop = config['training']['patience']
    epochs = config['training']['n_epochs']

    patience = early_stop

    for eid in range(epochs):
        model.train()
        train_loss = run_one_epoch(
            model, train_loader, loss_func, optimizer
        )

        model.eval()
        val_loss = run_one_epoch(
            model, val_loader, loss_func
        )

        print(f'Epoch {eid}, train loss {train_loss}, val loss {val_loss}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience = early_stop
        else:
            patience = patience - 1

        if patience == 0:
            break
    return best_model

def run_one_epoch_finetune(pretrain_model, model, data_loader, loss_func, optimizer=None):
   
    running_loss = 0.0

    for bid, (X_n, X_c, y) in enumerate(data_loader):
        
        pred = model(X_n, X_c)
        
        # pred loss
        if loss_func == F.cross_entropy:
            loss = loss_func(pred, y)
        else:
            loss = loss_func(pred, y.reshape(-1,1))

        if args.apply_attribute == 'True':
            r_hid = pretrain_model.estimator_encoder(X_n, X_c)
            r = pretrain_model.estimator_head(r_hid)
            r = torch.softmax(r, dim=1)
            pseudo = torch.mm(r.float(), pretrain_model.topics.float())

            hid = model.encoder(X_n, X_c)
            pred_pseudo = model.head_pseudo(hid)

            pseudo_loss = F.mse_loss(pseudo, pred_pseudo)
            loss += args.pseudo_loss_weight * pseudo_loss

            

        running_loss += loss.item()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return running_loss / len(data_loader)
    
def fit_fintune(pretrain_model,model, train_loader, val_loader, loss_func, model_type, config, task_type, y_std, args):
    best_val_loss = 1e30
    best_model = None

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    early_stop = config['training']['patience']
    epochs = config['training']['n_epochs']

    patience = early_stop

    for eid in range(epochs):
        if pretrain_model != None:
            pretrain_model.eval()
        model.train()
        train_loss = run_one_epoch_finetune(
            pretrain_model, model, train_loader, loss_func, optimizer
        )

        if pretrain_model != None:
            pretrain_model.eval()
        model.eval()
        val_loss = run_one_epoch_finetune(
            pretrain_model, model, val_loader, loss_func
        )

        print(f'Epoch {eid}, train loss {train_loss}, val loss {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience = early_stop
        else:
            patience = patience - 1

        if patience == 0:
            break
    return best_model

def test(model, test_loader, task_type, y_std, args, config,phase=""):
    assert phase in ["phase1","phase2"]
    model.eval()

    pred = []
    ground = []
    for bid, (X_n, X_c, y) in enumerate(test_loader):
        if phase=='phase2':
            pred.append(model(X_n, X_c).data.cpu().numpy())
        elif phase=='phase1':
            r_hid = model.estimator_encoder(X_n, X_c)
            r = model.estimator_head(r_hid)
            r = torch.softmax(r, dim=1)
            hid = torch.mm(r.float(), model.topics.float())
            pred.append(model.head_estimator(hid).data.cpu().numpy())
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
    np.save(open(f'./results/pseudo_label/{args.dataname}_{args.model_type}_{args.apply_attribute}_{args.n_clusters}_{args.pseudo_loss_weight}_{args.initial_method}_{args.distance_metric}_{args.ot_weight}_{args.orthogonal_weight}_{args.sample_ratio}_{args.mask_ratio}_{args.hyper}_{args.seed}_{phase}.npy','wb'), score.item())
    torch.save(model.state_dict(), f'./results/pseudo_label_models/{args.dataname}_{args.model_type}_{args.apply_attribute}_{args.n_clusters}_{args.pseudo_loss_weight}_{args.initial_method}_{args.distance_metric}_{args.ot_weight}_{args.orthogonal_weight}_{args.sample_ratio}_{args.mask_ratio}_{args.hyper}_{args.seed}_{phase}_model.pth')


def generate_topic(train_loader, val_loader, args, categories):
    if args.apply_attribute == 'False':
        return None

    data = []
    for bid, (X_n, X_c, y) in enumerate(train_loader):
        if categories is None:
            data.append(X_n)
        else:
            data.append(torch.cat([X_n, X_c], dim=1))

    for bid, (X_n, X_c, y) in enumerate(val_loader):
        if categories is None:
            data.append(X_n)
        else:
            data.append(torch.cat([X_n, X_c], dim=1))

    data = torch.cat(data, dim=0)
    if args.initial_method == 'kmeans':
        kmeans = KMeans(n_clusters=args.n_clusters)
        cluster_centers_ = kmeans.fit(data.data.cpu().numpy().astype('float64')).cluster_centers_
        cluster_centers_ = cluster_centers_.astype('float32')
    elif args.initial_method == 'random':
        # data_min = torch.min(data, dim=0)[0]
        # data_max = torch.max(data, dim=0)[0]
        # cluster_centers_ = np.random.uniform(data_min.cpu().numpy(), data_max.cpu().numpy(), size=(args.n_clusters, data.shape[1])).astype('float32')
        cluster_centers_ = np.random.randn(args.n_clusters, data.shape[1]).astype('float32')

    if args.model_type == 'MLP' or args.model_type == 'SNN':
        transformer = GaussianRandomProjection(n_components=config['model']['d_layers'][-1])
    elif args.model_type == 'FTTransformer':
        transformer = GaussianRandomProjection(n_components=config['model']['d_token'])
    elif args.model_type == 'AutoInt':
        transformer = GaussianRandomProjection(config['model']['d_token'] * (n_num_features + n_cat_features))
    elif args.model_type == 'ResNet':
        transformer = GaussianRandomProjection(n_components=config['model']['d'])
    elif args.model_type == 'DCN2':
        transformer = GaussianRandomProjection(n_components=config['model']['d'] if config['model']['stacked'] else 2 * config['model']['d'])

    cluster_centers_ = transformer.fit_transform(cluster_centers_)

    return cluster_centers_
def mask_data(X,ratio,categories, info):
    n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
    num_list = np.arange(n_num_features)
    cat_list = np.arange(n_num_features, n_num_features + n_cat_features) if n_cat_features!=None else None
    mask_dict = {}
    masked_X = {k:v.clone() for k,v in X.items()}
    # if mask ratio==0.0, then do not mask any data
    if ratio == 0.0:
        print("no mask")
        mask_dict = {k:torch.ones(v.shape).long().cuda() for k,v in masked_X.items()}
        return masked_X, mask_dict
    
    # generate mask matrix
    for k,v in masked_X.items():
        mask = torch.rand(v.shape).cuda()
        mask[mask<ratio]=0
        mask[mask>ratio]=1
        #v=v*mask not work, v is not reference of X_[k]
        #v*=mask
        mask_dict[k] = mask.long()
    
    # mask data of X
    for num_id in num_list:
        for k,v in masked_X.items():
            indices = torch.nonzero(mask_dict[k][:,num_id]==0).squeeze(dim=1)
            if indices.shape[0]!=0:
                v[indices,num_id] =  v[indices,num_id] -  v[indices,num_id]
    for cat_id in cat_list:
        for k,v in masked_X.items():
            indices = torch.nonzero(mask_dict[k][:,cat_id]==0).squeeze(dim=1)
            if indices.shape[0]!=0:
                v[indices,cat_id] = v[indices,cat_id] - v[indices,cat_id] + categories[cat_id-len(num_list)]-1
    return masked_X, mask_dict

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--apply_attribute', type=str)
    
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--pseudo_loss_weight', type=float, default=0.25)
    parser.add_argument('--initial_method', type=str, default='kmeans')
    parser.add_argument('--distance_metric', type=str, default='sqeuclidean')
    parser.add_argument('--orthogonal_weight', type=float, default=0.1)
    parser.add_argument('--ot_weight', type=float, default=0.01)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--mask_ratio', type=float, default=0.0)
    parser.add_argument('--hyper', type=str, default='default')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    _set_seed(args.seed)

    config = toml.load(f'../hypers_{args.hyper}/{args.dataname}/{args.model_type}.toml')

    with open(f'../data/{args.dataname}/info.json') as f:
        info = json.load(f)

    gc.collect()
    torch.cuda.empty_cache()

    # X, y, n_classes, y_mean, y_std, categories = load(args.dataname, info, config['data']['normalization'], args.sample_ratio)

    # task_type = info.get('task_type')
    # print(task_type)

    # n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
    # num_list = np.arange(n_num_features)
    # cat_list = np.arange(n_num_features, n_num_features + n_cat_features) if n_cat_features!=None else None
   
    # train_loader = build_data_loader(TensorDataset(X['train'][:,:n_num_features], X['train'][:,n_num_features:] if n_cat_features>0 else torch.empty(X['train'].shape[0], X['train'].shape[1]).cuda(), y['train']), config['training']['batch_size'], False)
    # val_loader = build_data_loader(TensorDataset(X['val'][:,:n_num_features], X['val'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['val'].shape[0], X['val'].shape[1]).cuda(), y['val']), config['training']['batch_size'], False)
    # test_loader = build_data_loader(TensorDataset(X['test'][:, :n_num_features], X['test'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['test'].shape[0], X['test'].shape[1]).cuda(), y['test']), config['training']['batch_size'], False)
    no_masked_X, y, n_classes, y_mean, y_std, categories = load(args.dataname, info, config['data']['normalization'], args.sample_ratio)
    masked_X,Mask_Mat=mask_data(no_masked_X, args.mask_ratio, categories, info)
    task_type = info.get('task_type')
    print(task_type)

    n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
    num_list = np.arange(n_num_features)
    cat_list = np.arange(n_num_features, n_num_features + n_cat_features) if n_cat_features!=None else None
   
    train_loader = build_data_loader(TensorDataset(masked_X['train'][:,:n_num_features], masked_X['train'][:,n_num_features:] if n_cat_features>0 else torch.empty(masked_X['train'].shape[0], masked_X['train'].shape[1]).cuda(), y['train']), config['training']['batch_size'], False)
    val_loader = build_data_loader(TensorDataset(masked_X['val'][:,:n_num_features], masked_X['val'][:, n_num_features:] if n_cat_features>0 else torch.empty(masked_X['val'].shape[0], masked_X['val'].shape[1]).cuda(), y['val']), config['training']['batch_size'], False)
    test_loader = build_data_loader(TensorDataset(masked_X['test'][:, :n_num_features], masked_X['test'][:, n_num_features:] if n_cat_features>0 else torch.empty(masked_X['test'].shape[0], masked_X['test'].shape[1]).cuda(), y['test']), config['training']['batch_size'], False)


    ## generating topcs
    topics = generate_topic(train_loader, val_loader, args, categories)

    ## model initialization
    _set_seed(args.seed)
    pretrain_model = Model(n_num_features, args.model_type, n_classes if task_type == 'multiclass' else 1, 
    info=info, config = config, categories = categories, topic_num = args.n_clusters, topics = topics, args = args)

    model = copy.deepcopy(pretrain_model)


    ## pre-training
    if args.apply_attribute == 'True':
        pretrain_model.cuda()
        loss_func = get_loss_function(task_type)
        best_pretrain_model = fit(pretrain_model, train_loader, val_loader, loss_func, args.model_type, config, task_type, y_std, args)
        test(best_pretrain_model, test_loader, task_type, y_std, args, config,"phase1")
    else:
        best_pretrain_model = None

    ## finetune
    model.cuda()
    loss_func = get_loss_function(task_type)
    best_model = fit_fintune(best_pretrain_model, model, train_loader, val_loader, loss_func, args.model_type, config, task_type, y_std, args)

    ## test
    test(best_model, test_loader, task_type, y_std, args, config,"phase2")




