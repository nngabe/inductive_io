import warnings
warnings.filterwarnings('ignore')

import argparse
import datetime

import sys
import networkx as nx
import pandas as pd
import numpy as np
import glob
import datetime
import pickle
import copy

import torch
from captum.attr import IntegratedGradients

from cutils import load_data, write_results, to_inductive
from models.models import *

if __name__ == '__main__':
    """ Train and evaluate models with the following steps:
         1. Initialize model (LR, RF, MLP, GCN, GCN_MPs, GCN_MP)'
         2. Load PE specified by binary string (one-hot, node2vec, lap-eig, random-walk, net-feat)
         3. Load dataset specified by train and test binary string
         4. Train and log progress, including best performance
         5. Compute empirical baseline for Integrated Gradients 
         6. Compute and log IG values for individual features and feature types
    """ 
    parser = argparse.ArgumentParser(description='Load model, loss, and evaluation arguments.')
    parser.add_argument('--model',nargs='?',default='MLP',type=str)
    parser.add_argument('--pe', nargs='?', type=str, default='01111')
    parser.add_argument('--dropout', nargs='?', type=float, default=0.5)
    parser.add_argument('--hidden', nargs='?', type=int, default=64)
    parser.add_argument('--num', nargs='?', type=int, default=2)
    parser.add_argument('--doms', nargs='?', type=int, default=1000)
    parser.add_argument('--train', nargs='?', type=str, default='100')
    parser.add_argument('--test', nargs='?', type=str, default='100')
    parser.add_argument('--epochs', nargs='?', type=int, default=4)
    parser.add_argument('--iters', nargs='?', type=int, default=3001)
    parser.add_argument('--rep', nargs='?', type=int, default=100)
    parser.add_argument('--gamma', nargs='?', type=float, default=.4)
    parser.add_argument('--metric', nargs='?', type=str, default='ratio')
    parser.add_argument('--exp', nargs='?', type=int, default=0)
    parser.add_argument('--norm', nargs='?', type=int, default=0)
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3)
    parser.add_argument('--l1', nargs='?', type=float, default=0.)
    parser.add_argument('--l2', nargs='?', type=float, default=0.)
    parser.add_argument('--device', nargs='?', type=str, default='cuda')

    args = parser.parse_args()
    print()
    print(args)

    args.subset = ['comp_train',
                   'russia_train',
                   'china_train',
                   'iran_train',
                   'russia_test',
                   'china_test',
                   'iran_test']

    MODEL = args.model
    PE = args.pe # one-hot, node2vec, lap-eig, random-walk, net-feat
    NUM_DOMS = args.doms

    WEIGHT = (MODEL in ['GCN1','GCN2'])

    data, xdf, index, lims = get_mask_data(args)
    sets = (xdf.lab_state + '_' + xdf.lab_set).to_numpy()
    xdf.insert(0, 'label', sets)
    reset_idx(args, data, xdf)
    data=data.to('cpu')

    print(f'train,test,val={data.train.sum().item(),data.test.sum().item(),data.val.sum().item()}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train.sum()}')
    print(f'Training node label rate: {int(data.train.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print()

    in_channels = data.num_features
    hidden_channels = args.hidden
    num_layers = args.num
    dropout= args.dropout
    out_channels = data.num_classes

    if MODEL=='LR':
        print(f'model: {MODEL}')
        res = np.zeros((args.epochs,3)) # F1_val, F1_test, AUC_test
        for i in range(args.epochs): 
            model, yt, f1v, f1t, auct = LR(args, data, xdf, i)
            res[i] = (f1v, f1t, auct)
        F1v = (res[:,0].mean(), res[:,0].std())
        F1t = (res[:,1].mean(), res[:,1].std())
        AUCt = (res[:,2].mean(), res[:,2].std())
        write_results(F1v, F1t, AUCt, model, args)
        sys.exit(0)

    elif MODEL=='RF':
        print(f'model: {MODEL}')
        res = np.zeros((args.epochs,3)) # F1_val, F1_test, AUC_test
        for i in range(args.epochs): 
            model, yt, f1v, f1t, auct = RF(args, data, xdf, i)
            res[i] = (f1v, f1t, auct)
        F1v = (res[:,0].mean(), res[:,0].std())
        F1t = (res[:,1].mean(), res[:,1].std())
        AUCt = (res[:,2].mean(), res[:,2].std())
        write_results(F1v, F1t, AUCt, model, args)
        sys.exit(0)

    elif MODEL=='MLP':
        model = MLP(in_channels, hidden_channels, out_channels, args.num, dropout)
    
    elif MODEL=='GCN':
        model = GCN(data.num_features, hidden_channels, data.num_classes, args.num, dropout, weighted=False)
    
    elif MODEL=='GCN_MPs':
        model = GCN_lw(data.num_features, hidden_channels, data.num_classes, args.num, dropout, weighted=True)
    
    elif MODEL=='GCN_MP':
        model = GCN(data.num_features, hidden_channels, data.num_classes, args.num, dropout, weighted=True)

    print(model)
    print(f'NUM_PARAMS={sum(p.numel() for p in model.parameters())}\n')
    print(f'{args.train} -> {args.test}')

    model = model.to(args.device)
    data = data.to(args.device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    we = data.edge_attr if WEIGHT else None
    model_best = [None for _ in range(args.epochs)]
    pred_best = [None for _ in range(args.epochs)]
    yb = [None for _ in range(args.epochs)]

    print('test: ',data.test.sum())
    loader = data
    res = np.zeros((args.epochs,3))
    for epoch in range(args.epochs):
        reset_idx(args, data, xdf, epoch)
        model.reset_params()
        print(f'epoch({epoch}):')
        rec_f1 = 0.
        rec_auc = 0.
        fbest = (0, 0, 0)
        for i in range(args.iters):
            model.zero_grad()
            loss = train_single(model, loader, args, optimizer, loss_fn)
            if i%args.rep == 0: 
                reset_idx(args, data, xdf, epoch)
                model.eval()
                fval = F1_star(model, data, data.val)
                ftest = F1_star(model, data, data.test)
                print(f" iter: {i:03d}, Loss: {loss:.5f}, F1(val): {fval[0]:.5f}, F1(test): {ftest[0]:.5f}, AUC(test): {ftest[1]:.5f}, prod(test): {ftest[0]*ftest[1]:.5f}")
                if ftest[0]*ftest[1]>rec_f1*rec_auc:
                    rec_f1 = ftest[0]
                    rec_auc = ftest[1]
                    fbest = [fval[0],ftest[0],ftest[1]]
                    model_best[epoch] = copy.deepcopy(model).cpu()
                    pred_best[epoch] = model(data.x,data.edge_index).detach().cpu().numpy()
                model.train()
        print(f'F1(val): {fbest[0]:.5f}, F1(test): {fbest[1]:.5f}, AUC(test): {fbest[2]:.5f}, prod(test): {fbest[1]*fbest[2]:.5f}')
        res[epoch] = fbest

    f1v,f1t,auct = res[:,0], res[:,1], res[:,2]
    prod = np.array(f1t)*np.array(auct)
    idx_best = np.argmax(prod)
    F1v = (f1v.mean(),f1v.std())
    F1t = (f1t.mean(),f1v.std())
    AUCt = (auct.mean(),auct.std())
    mdl = model_best[idx_best].state_dict()

    write_results(F1v, F1t, AUCt, mdl, args)

    print()
    print(f'F1(val) = {F1v[0]:.5f} \u00b1 {F1v[1]:.5f}')
    print(f'F1(test) = {F1t[0]:.5f} \u00b1 {F1t[1]:.5f}')
    print(f'AUC(test) = {AUCt[0]:.5f} \u00b1 {AUCt[1]:.5f}')

    smask = {sub : (xdf.label==sub).to_numpy() for sub in args.subset}
    sdata = {sub : data.x[smask[sub]] for sub in args.subset}

    if args.exp:
        model.eval()
        bmask = torch.abs(model(data.x)-.5).flatten() < 0.1 
        baseline = data.x[bmask].mean(0)
        print(f'model(baseline) = {model(baseline).item():.3e}')
        ig = IntegratedGradients(model)
        _attr = lambda sub: ig.attribute(sdata[sub], method='gausslegendre', return_convergence_delta=True)
        attr = {sub : _attr(sub)[0].mean(0) for sub in args.subset}

    if args.exp:
        sub_train = [s for s,i in zip(args.subset[1:4],args.train) if int(i)]
        sub_val   = [s for s,i in zip(args.subset[1:4],args.test) if int(i)]
        sub_test =  [s for s,i in zip(args.subset[4:],args.test) if int(i)]
        sub_baseline = args.subset[0]

        print_attr_tex(sub_train)
        print_attr_tex(sub_val)
        print_attr_tex(sub_test)
        print_attr_tex(sub_baseline)
