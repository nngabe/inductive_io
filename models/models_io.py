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

def reset_idx(args, data, xdf, seed=0):
    sub_train = [s for s,i in zip(args.subset[1:4],args.train) if int(i)]
    sub_val = [s for s,i in zip(args.subset[1:4],args.test) if int(i)]
    sub_test = [s for s,i in zip(args.subset[4:],args.test) if int(i)]
    sub_base = ['comp_train']
    
    np.random.seed(seed)
    val_c = xdf.label.apply(lambda x: x in sub_val)
    mask = (np.random.rand(val_c.shape[0])<.2)
    if args.train==args.test:
        _train = val_c * ~mask
        _val = val_c * mask 
        _test = xdf.label.apply(lambda x: x in sub_test)
        _base = xdf.label.apply(lambda x: x in sub_base)
    else:
        _train = xdf.label.apply(lambda x: x in sub_train)
        _val = val_c * mask
        _test = xdf.label.apply(lambda x: x in sub_test)
        _base = xdf.label.apply(lambda x: x in sub_base)

    dev = data.x.device
    data.train = torch.tensor(_train.to_numpy(), device=dev)
    data.val = torch.tensor(_val.to_numpy(), device=dev)
    data.test = torch.tensor(_test.to_numpy(), device=dev)
    data.base = torch.tensor(_base.to_numpy(), device=dev)

    train = torch.where(data.train)[0]
    val = torch.where(data.val)[0]
    test = torch.where(data.test)[0]
    base = torch.where(data.base)[0]

    base_train = base[torch.randperm(base.shape[0])[:train.shape[0]]]
    base_val = base[torch.randperm(base.shape[0])[:val.shape[0]]]
    base_test = base[torch.randperm(base.shape[0])[:test.shape[0]]]

    data.train_idx = torch.cat([train, base_train])
    data.val_idx = torch.cat([val, base_val])
    data.test_idx = torch.cat([test, base_val])
    
    data.train_mask = torch.ones_like(data.train)*False
    data.val_mask = torch.ones_like(data.val)*False
    data.test_mask = torch.ones_like(data.test)*False
    
    data.train_mask[data.train_idx] = True
    data.val_mask[data.val_idx] = True
    data.test_mask[data.test_idx] = True

def LR(args, data, xdf, seed=0):
    reset_idx(args, data, xdf, seed)
    nr = np.random.randint(0,100)
    model = LogisticRegression(penalty='l2', C=.5, solver='liblinear', random_state=nr)
    model.fit(data.x[data.train_idx].numpy(), data.y[data.train_idx].numpy())
    yv = model.predict(data.x[data.val_idx].numpy())
    yt = model.predict(data.x[data.test_idx].numpy())
    f1s = f1_score(yv,data.y[data.val_idx].numpy(),average='macro')
    f1ts = f1_score(yt,data.y[data.test_idx].numpy(),average='macro')
    y = data.y[data.test_idx]
    roc = roc_auc_score(y,yt)
    print(f" Epoch: F1(val): {f1s:.4f}, F1(test): {f1ts:.4f}, AUC(test): {roc:.4f}")  
    return model, yt, f1s, f1ts, roc
def RF(args, data, xdf, seed=0):
    reset_idx(args, data, xdf, seed)
    model = RandomForestClassifier(n_estimators=500, max_depth=50, verbose=0, max_features='sqrt')
    model.fit(data.x[data.train_idx].numpy(), data.y[data.train_idx].numpy())
    yv = model.predict(data.x[data.val_idx].numpy())
    yt = model.predict(data.x[data.test_idx].numpy())
    f1s = f1_score(yv,data.y[data.val_idx].numpy(),average='macro')
    f1ts = f1_score(yt,data.y[data.test_idx].numpy(),average='macro')
    y = data.y[data.test_idx]
    roc = roc_auc_score(y,yt)
    print(f" Epoch: F1(val): {f1s:.4f}, F1(test): {f1ts:.4f}, AUC(test): {roc:.4f}")  
    return model, yt, f1s, f1ts, roc

def train_single(model, data, args, optimizer, loss_fn):
    model.train()
    #model = model.to(args.device)
    #total_loss = 0
    #for batch in loader:
    batch = data
    optimizer.zero_grad()
    batch = batch.to(args.device)
    if hasattr(batch,'edge_attr'):
        out = model.forward(batch.x, batch.edge_index, batch.edge_attr)  
    else:
        out = model.forward(batch.x, batch.edge_index)
    loss = loss_fn(out, batch.y.unsqueeze(-1))  
    loss.backward()
    optimizer.step()  
        
    return loss

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
def F1_test(model, data, mask, alpha=0.5):
    model.eval()
    out = model.forward(data.x,data.edge_index)
    def predict(alpha): return (out>alpha).cpu().numpy().astype(np.int32)
    true = data.y.cpu().numpy()
    if hasattr(mask,'device'): mask = mask.cpu()
    score = f1_score(predict(alpha)[mask], true[mask], average='macro') 
    roc = roc_auc_score(true[mask],out.cpu()[mask].detach())
    return  (score,roc)

@torch.no_grad()
def F1_star(model, data, mask, alpha=0.5):
    model.eval()
    out = model.forward(data.x,data.edge_index)
    idx = np.where(data.base.cpu().numpy())[0]
    order = out[data.base].flatten().argsort().detach().cpu().numpy()
    n = mask.sum()
    idy = np.sort(np.flip(idx[order])[:n])
    mask[idy] = True
    def predict(alpha): return (out>alpha).cpu().numpy().astype(np.int32)
    true = data.y.cpu().numpy()
    pred = out.cpu().flatten().numpy()
    if hasattr(mask,'device'): mask = mask.cpu()
    score = f1_score(predict(alpha)[mask], true[mask], average='macro') 
    roc = roc_auc_score(true[mask], pred[mask])
    return  (score, roc, true, pred)

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

        self.lin_enc = Linear(in_channels,hidden_channels)
        self.lin_dec = Linear(hidden_channels,out_channels)
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
                w = F.dropout(w, p=.0, training=self.training)
            w = self.wconv[-1](w).sigmoid()
            w = torch.cat((w,torch.ones((x.shape[0],1),device=x.device)),0)
           
        x0 = F.dropout(self.lin_enc(x).relu(), p=self.dropout, training=self.training) 
        _x = x
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, w)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if _x.shape[1]==x.shape[1]:
                x += _x
            _x = x
        x += x0
        x = self.lin_dec(x)
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
    
