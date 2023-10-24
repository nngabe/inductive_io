import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import TransformerConv
import torch_geometric.transforms as T

import numpy as np
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
eta = 1e-2
gamma=30.
max_depth= 40 
subsample = 1
colsample_bytree = 1
min_child_weight=1
random_state = 0
params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "rmsle",
        "eta": eta,
        "gamma": gamma,
        "tree_method": "exact",
        "max_depth": max_depth,
        "subsample": .7,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "seed": random_state,
    }
num_boost_round = 500
early_stopping_rounds = 5

def data_train(data):
    data = data.clone()
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None, relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data

def reset_idx(data, sample=False):
    train = torch.where(data.train)[0]
    val = torch.where(data.val)[0]
    comp = torch.where(data.comp)[0]

    comp_train = comp[torch.randperm(comp.shape[0])[:train.shape[0]]]
    comp_val = comp[torch.randperm(comp.shape[0])[:val.shape[0]]]
    
    data.train_idx = torch.cat([train,comp_train])
    data.val_idx = torch.cat([val,comp_val])
    
    data.train_mask = torch.ones_like(data.train)*False
    data.val_mask = torch.ones_like(data.val)*False
    
    data.train_mask[data.train_idx] = True
    data.val_mask[data.val_idx] = True

def LR(data):
    set_idx(data)
    nr = np.random.randint(0,100)
    model = LogisticRegression(penalty='l2', C=.5, solver='liblinear', random_state=nr)
    model.fit(data.x[data.train_idx].numpy(), data.y[data.train_idx].numpy())
    yv = model.predict(data.x[data.val_idx].numpy())
    yt = model.predict(data.x[data.test_idx].numpy())
    f1s = f1_score(yv,data.y[data.val_idx].numpy(),average='macro')
    f1ts = f1_score(yt,data.y[data.test_idx].numpy(),average='macro')
    y = data.y[data.test_idx]
    roc = roc_auc_score(y,yt)
    print(f" Epoch: F1(val): {f1s:.4f}, F1(test): {f1ts:.4f}, AUC(test): {roc}")  
    return model, yt, f1s, f1ts, roc
def RF(data):
    set_idx(data)
    model = RandomForestClassifier(n_estimators=500,max_depth=50,verbose=0,max_features='sqrt')
    model.fit(data.x[data.train_idx].numpy(), data.y[data.train_idx].numpy())
    yv = model.predict(data.x[data.val_idx].numpy())
    yt = model.predict(data.x[data.test_idx].numpy())
    f1s = f1_score(yv,data.y[data.val_idx].numpy(),average='macro')
    f1ts = f1_score(yt,data.y[data.test_idx].numpy(),average='macro')
    y = data.y[data.test_idx]
    roc = roc_auc_score(y,yt)
    print(f" Epoch: F1(val): {f1s:.4f}, F1(test): {f1ts:.4f}, AUC(test): {roc}")  
    return model, yt, f1s, f1ts, roc
def XGB(data,num_boost_rounds=500,early_stopping_rounds=20):
    set_idx(data)
    dm = xgb.DMatrix
    dtrain = dm(data.x[data.train_idx].numpy(), data.y[data.train_idx].numpy())
    dvalid = dm(data.x[data.val_idx].numpy(), data.y[data.val_idx].numpy())
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    yv = model.predict( dm(data.x[data.val_idx].numpy()) ) 
    yt = model.predict( dm(data.x[data.test_idx].numpy()) )
    f1s = f1_score(yv.round().astype(int),data.y[data.val_idx].numpy().astype(int),average='macro')
    f1ts = f1_score(yt.round().astype(int),data.y[data.test_idx].numpy().astype(int),average='macro')
    y = data.y[data.test_idx].numpy().astype(int)
    roc = roc_auc_score(y,yt.round().astype(int))
    print(f" Epoch: F1(val): {f1s:.4f}, F1(test): {f1ts:.4f}, AUC(test): {roc}")  
    return model, yt, f1s, f1ts, roc

def train(model, loader, args, optimizer, loss_fn):
    model.train()
    model = model.to(args.device)
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(args.device)
        if hasattr(batch,'edge_attr'):
            out = model.forward(batch.x, batch.edge_index, batch.edge_attr)  
        else:
            out = model.forward(batch.x, batch.edge_index)
        loss = loss_fn(out, batch.y.unsqueeze(-1))  
        if args.l1>0.: loss += args.l1 * torch.abs(torch.cat([p.view(-1) for p in model.parameters()])).sum()
        if args.l2>0.: loss += args.l2 * torch.square(torch.cat([p.view(-1) for p in model.parameters()])).sum()
        loss.backward()
        optimizer.step()  
        total_loss += loss.item()
        
    return total_loss / len(loader)

@torch.no_grad()
def F1_star(model, data, mask, alpha=0.5):
    model.eval()
    model = model.to('cpu')
    out = model.forward(data.x,data.edge_index)
    idx = np.where(data.comp.cpu().numpy())[0]
    order = out[data.comp].flatten().argsort().detach().cpu().numpy()
    n = mask.sum()
    idy = copy.copy(np.flip(idx[order])[:n])
    mask = copy.copy(mask)
    mask[idy] = True
    def predict(alpha): return (out>alpha).cpu().numpy().astype(np.int32)
    true = data.y.cpu().numpy()
    if hasattr(mask,'device'): mask = mask.cpu()
    score = f1_score(predict(alpha)[mask], true[mask], average='macro') 
    roc = roc_auc_score(true[mask],out.cpu()[mask].detach())
    return  (score,roc)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(Linear(hidden_channels,hidden_channels))
        self.convs.append(Linear(hidden_channels, out_channels))
        self.dropout = dropout
        
    def reset_params(self):
        def rp(module):
            op = getattr(module, 'reset_parameters', None)
            if callable(op): op()
            elif (hasattr(module,'children')):
                for m in module.children(): 
                    rp(m)
        rp(self)

    def forward(self, x, edge_index=None, w=None):
        for conv in self.convs[:-1]:
            x = conv(x)
            x = F.relu(x)
            #x = x/x.norm(dim=1).view(-1,1)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return torch.sigmoid(x)

class RNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, bidirectional=False):
        super(RNN,self).__init__()
        self.rnn = torch.nn.LSTM(in_channels,hidden_channels,num_layers,dropout=dropout,bidirectional=bidirectional,proj_size=1)
        self.bi = bidirectional       
        if self.bi:
            self.L = torch.nn.Linear(2,1)
            
    def reset_params(self):
        def rp(module):
            op = getattr(module, 'reset_parameters', None)
            if callable(op): op()
            elif (hasattr(module,'children')):
                for m in module.children(): 
                    rp(m)
        rp(self)

    def forward(self, x, edge_index=None):
        if self.bi:
            return torch.sigmoid(self.L(self.rnn(x)[0]))
        else:
            return torch.sigmoid(self.rnn(x)[0])
    
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
        
class MP(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super().__init__(aggr=aggr) 
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):

        tmp = torch.cat([x_i, x_i - x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
    
    
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, in_dim, dropout=0.5, aggr=['mean'], skip=None, conv=SAGEConv):
        super().__init__()
        
        self.dropout = dropout
        self.skip = skip
        self.aggr = aggr
        self.num_aggr = len(aggr)
        
        self.enc = torch.nn.ModuleList()
        self.enc.append(conv(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers-1):
            self.enc.append(conv(hidden_channels, hidden_channels, aggr=copy.deepcopy(aggr)))
        
        if self.skip=='cat': out_dim = hidden_channels + hidden_channels*num_layers
        elif self.skip=='sum': out_dim = hidden_channels
        else: out_dim = hidden_channels
        
        self.lin_skip = torch.nn.Linear(in_dim, hidden_channels)
        self.lin1 = torch.nn.Linear(out_dim, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
    def reset_params(self):
        def rp(module):
            op = getattr(module, 'reset_parameters', None)
            if callable(op): op()
            elif (hasattr(module,'children')):
                for m in module.children(): 
                    rp(m)
        rp(self)

    def forward(self, x, edge_index, edge_attr=None):
        xi = []
        mask = (x==1.).sum(0)!=1 # mask out ones if present
        xs = self.lin_skip(x[:,mask]).relu()
        #print('xs=',xs.shape)
        xi.append(xs)
        
        for enc in self.enc[:-1]:
            x = enc(x, edge_index, edge_attr).view(self.num_aggr,x.shape[0],-1).sum(0)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            #print('x=',x.shape)
            xi.append(x)
        x = self.enc[-1](x, edge_index, edge_attr).view(self.num_aggr,x.shape[0],-1).sum(0)
        #print('x=',x.shape)
        xi.append(x)
        
        if self.skip=='cat': x = torch.cat(xi,axis=1)
        elif self.skip=='sum': x = sum(xi)
        
        x = self.lin1(x).relu()
        x = self.lin2(x).sigmoid()
        
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, in_dim, dropout=0.5, aggr='mean', skip=None, conv=GATConv, dec='mlp'):
        super().__init__()
        
        self.dropout = dropout
        self.skip = skip
        
        self.enc = torch.nn.ModuleList()
        self.dec = dec
        self.heads = 1
        self.enc.append(conv(in_channels, hidden_channels, aggr=aggr, heads=self.heads))
        for _ in range(num_layers-1):
            self.enc.append(conv(self.heads*hidden_channels, hidden_channels, aggr=copy.deepcopy(aggr), heads=self.heads))
        
        if self.skip=='cat': out_dim = hidden_channels + hidden_channels*num_layers
        elif self.skip=='sum': out_dim = hidden_channels
        else: out_dim = hidden_channels
        
        self.lin_skip = torch.nn.Linear(in_dim, hidden_channels)
        if dec=='lstm':
            self.rnn = torch.nn.LSTM(in_channels,hidden_channels,num_layers,dropout=dropout,bidirectional=True,proj_size=1)
            self.L = torch.nn.Linear(2,1)
        elif dec=='mlp':
            self.lin1 = torch.nn.Linear(out_dim, hidden_channels)
            self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
    def reset_params(self):
        def rp(module):
            op = getattr(module, 'reset_parameters', None)
            if callable(op): op()
            elif (hasattr(module,'children')):
                for m in module.children(): 
                    rp(m)
        rp(self)

    def forward(self, x, edge_index, edge_attr=None):
        xi = []
        mask = (x==1.).sum(0)!=1 # mask out ones if present
        xs = self.lin_skip(x[:,mask]).relu()
        #print('xs=',xs.shape)
        xi.append(xs)
        
        for enc in self.enc[:-1]:
            x = enc(x, edge_index, edge_attr).view(3,x.shape[0],-1).sum(0)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            #print('x=',x.shape)
            xi.append(x)
        x = self.enc[-1](x, edge_index, edge_attr).view(3,x.shape[0],-1).sum(0)
        #print('x=',x.shape)
        xi.append(x)
        
        if self.skip=='cat': x = torch.cat(xi,axis=1)
        elif self.skip=='sum': x = sum(xi)
    
        if self.dec=='lstm':
           pass 
    
    
        if self.dec=='mlp': 
            x = self.lin1(x).relu()
            x = self.lin2(x).sigmoid()
        
        return x
    

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, weighted=False):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        self.dropout = dropout
        
        self.weighted=weighted
        if self.weighted:
            inc = 100
            hidc = 16
            self.wconv = torch.nn.ModuleList()
            self.wconv.append(Linear(inc,hidc))
            self.wconv.append(Linear(hidc,1))
            
    def reset_params(self):
        def rp(module):
            op = getattr(module, 'reset_parameters', None)
            if callable(op): op()
            elif (hasattr(module,'children')):
                for m in module.children(): 
                    rp(m)
        rp(self)
            
    def forward(self, x, edge_index, w=None):
        if self.weighted and torch.is_tensor(w):
            w = w[:-20616]
            for conv in self.wconv[:-1]:
                w = conv(w)
                w = F.relu(w)
                w = F.dropout(w, p=.1, training=self.training)
            w = self.wconv[-1](w).sigmoid()
            w = torch.cat((w,torch.ones((x.shape[0],1),device=x.device)),0)
            
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, w)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return torch.sigmoid(x)

    
class GCN_lw(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, weighted=False):
        super(GCN_lw, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        self.dropout = dropout
        
        self.weighted=weighted
        if self.weighted:
            inc = 99
            self.wconv = torch.nn.ModuleList()
            self.wconv.append(Linear(inc,1))
            
    def reset_params(self):
        def rp(module):
            op = getattr(module, 'reset_parameters', None)
            if callable(op): op()
            elif (hasattr(module,'children')):
                for m in mod.children(): 
                    rp(m)
        rp(self)

    def forward(self, x, edge_index, w=None):
        if torch.is_tensor(w):
            for conv in self.wconv[:-1]:
                w = conv(w)
                w = F.relu(w)
                w = F.dropout(w, p=.2, training=self.training)
            w = self.wconv[-1](w) 
            #w = torch.sigmoid(w)
            w = torch.cat((w,torch.ones((x.shape[0],1),device='cuda')),0)
            
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, w)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return torch.sigmoid(x)

class GAT0(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads=3, weighted=False):
        super().__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels,heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(GATConv(heads * hidden_channels, out_channels, heads, concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(torch.nn.Linear(in_channels, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(torch.nn.Linear(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(torch.nn.Linear(hidden_channels * heads, out_channels))
        self.dropout = dropout

        self.weighted=weighted
        if self.weighted:
            inc = 100
            hidc = 16
            self.wconv = torch.nn.ModuleList()
            self.wconv.append(Linear(inc,hidc))
            self.wconv.append(Linear(hidc,1))
            
    def reset_params(self):
        def rp(module):
            op = getattr(module, 'reset_parameters', None)
            if callable(op): op()
            elif (hasattr(module,'children')):
                for m in mod.children(): 
                    rp(m)
        rp(self)
            
    def forward(self, x, edge_index, w=None):
        if self.weighted and torch.is_tensor(w):
            for conv in self.wconv[:-1]:
                w = conv(w)
                w = F.relu(w)
                w = F.dropout(w, p=.1, training=self.training)
            w = self.wconv[-1](w) 
            w = torch.cat((w,torch.ones((x.shape[0],1),device=x.device)),0)
            
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, w)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return torch.sigmoid(x)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_params(self):
        def rp(module):
            op = getattr(module, 'reset_parameters', None)
            if callable(op): op()
            elif (hasattr(module,'children')):
                for m in mod.children(): 
                    rp(m)
        rp(self)

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1)
    
