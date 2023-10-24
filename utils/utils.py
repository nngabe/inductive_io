import pandas as pd
import numpy as np
import time

import datetime
import copy
from ast import literal_eval
import urllib3
import tldextract
import requests
import matplotlib.pyplot as plt
import itertools
import tldextract

from torch_geometric.data import Data
from torch_geometric.utils import subgraph

import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from ast import literal_eval as lit_eval


# shortened urls not in publicly available lists
alias = {'dailym.ai': 'dailymail.co.uk',
 'amzn.to': 'amazon.com',
 'apne.ws': 'apnews.com',
 'abcn.ws': 'abcnews.com',
 'bloom.bg': 'bloomberg.com',
 'politi.co': 'politico.com',
 'fxn.ws': 'foxnews.com',
 'usat.ly': 'usatoday.com',
 'n.pr': 'npr.org',
 'cbsloc.al': 'cbsnews.com',
 'es.pn': 'espn.com',
 'spoti.fi': 'spotify.com',
 'nyp.st': 'nypost.com',
 'washex.am': 'washingtonexaminer.com',
 'etsy.me': 'etsy.com',
 'read.bi': 'businessinsider.com',
 'ti.me': 'time.com',
 'lat.ms': 'latimes.com',
 'wpo.st': 'washingtonpost.com',
 'econ.st': 'economist.com',
 'thebea.st': 'thedailybeast.com',
 'cnb.cx': 'cnbc.com',
 'bzfd.it': 'buzzfeed.com',
 'crwd.fr': 'crowdfireapp.com',
 'appsto.re': 'apple.com',
 'fb.com': 'facebook.com',
 'owl.li': 'other_dom',
 'commun.it': 'other_dom',
 'paper.li': 'other_dom',
 'fw.to': 'other_dom',
 'cle.clinic': 'clevelandclinic.com',
 'mailchi.mp': 'mailchimp.com',
 'untp.beer': 'untappd.com',
 'cle.ac': 'bleacherreport.com',
 'spr.ly': 'other_dom',
 'thr.cm': 'other_dom',
 'eonli.ne': 'eonline.com',
 'pdora.co': 'pandora.com',
 'peoplem.ag': 'people.com',
 'lnk.to': 'linkfire.com',
 'chng.it': 'change.org',
 'ed.gr': 'meetedgar.com',
 'goo.gle': 'google.com',
 'shorturl.at': 'other_dom',
 'forms.gle': 'other_dom',
 'shr.lc': 'shareaholic.com',
 'rviv.ly': 'revive.social',
 'flip.it': 'flipit.com',
 'itun.es': 'apple.com',
 'chn.ge': 'change.org',
 'tnw.to': 'thenextweb.com',
 'thkpr.gs': 'thinkprogress.org',
 'huffp.st': 'huffpost.com',
 'huffingtonpost.com': 'huffpost.com',
 'cbsn.ws': 'cbsnews.com',
 'ia.cr': 'iacr.org',
 'flic.kr': 'flickr.com',
 'msft.social': 'microsoft.com',
 'interc.pt': 'theintercept.com',
 'aka.ms': 'microsoft.com',
 'tcrn.ch': 'techchrunch.com',
 'ebay.to': 'ebay.com'
}

def censor_domains(df,args):
    """ Censor domains by frequencies in the training dataset.
    """
    if args.metric=='ratio':
        xdf = pd.read_csv('../comp_nodes/xdf_rci.csv',index_col=0)
        mask = np.where(xdf.isna().sum(1)>0)
        xdf.lab_io.iloc[mask] = '0'
        xdf.lab_state.iloc[mask] = 'comp'
        xdf.lab_set.iloc[mask] = 'train'

        df = xdf.reset_index(drop=True)
        edges = pd.read_csv('../edge_data/C_15_rci.csv',index_col=0).to_numpy(dtype=np.int32)
        ew = pd.read_csv('../edge_data/edge_counts_rci.csv',index_col=0)

        n1 = df[df.lab_io==1].iloc[:,4:].sum()
        n0 = df[df.lab_io==0].iloc[:,4:].sum()
        ratio = n1/n0
        p = args.gamma
        max_ratio = p/(1.-p)
        print(f'max_ratio = {max_ratio:.2f}')
        strip = np.logical_or(ratio>max_ratio,ratio.isna()) # strip domains with ratios which are obvious give aways and nan.
        df = df.drop(columns=strip[strip].index.to_list())
        df = df.reset_index(drop=True)
        cnew = [alias[c].lower() if c in list(alias.keys()) else c.lower() for c in list(df.columns)]
        df.columns = cnew
        df = df.sum(axis=1, level=0)
        index = df.columns
    elif args.metric=='score':
        stats = pd.read_csv('domain_stats_.csv',index_col=0)
        if len(stats.columns)!=len(df.columns[4:]): 
            stats = score_domains(args,res=True)
        m1 = stats.loc['n0_count']>2
        m2 = stats.loc['n1_count']>2
        m3 = stats.loc['n0_num']>1
        m4 = stats.loc['n1_num']>1
        m12 = np.logical_and(m1,m2)
        m34 = np.logical_and(m3,m4)
        mask = np.logical_and(m12,m34)
        stats = stats.loc[:,mask].T
        
        ls = stats.applymap(np.log)
        power = ls.iloc[:,4:6].prod(1)**args.alpha * ls.iloc[:,6:8].prod(1)**args.beta
        score = power * (stats.mutual_info_score+1e-12)**args.gamma
        rank = score.sort_values()[::-1]
        index = df.columns[:4].to_list() + rank.index.to_list()
        df = df.loc[:,index]
    else:
        print('args.metric not in [ratio,score]')
    
    return df,(index,strip)
    

def load_data(args,early=False):
    """ Get full data set and mask depending on training/testing campaign(s):
        1. read csv files with edge-wise and node-wise features
        2. select positional encoding spcified by arguments
        3. normalize data as specified by arguments
        4. 
    """ 
    MODEL = args.model
    PE = args.pe # one-hot, node2vec, lap-eig, random-walk, net-feat
    NUM_DOMS = args.doms

    WEIGHT = (MODEL in ['GCN1','GCN2','GNN','MPNN','GAT']) 
    
    xdf = pd.read_csv('../comp_nodes/xdf_rci.csv',index_col=0)
    mask = np.where(xdf.isna().sum(1)>0)
    xdf.lab_io.iloc[mask] = '0'
    xdf.lab_state.iloc[mask] = 'comp'
    xdf.lab_set.iloc[mask] = 'train'
    if early==1: return xdf

    df = xdf.reset_index(drop=True)
    edges = pd.read_csv('../edge_data/C_15_rci.csv',index_col=0).to_numpy(dtype=np.int32)
    ew = pd.read_csv('../edge_data/edge_counts_rci.csv',index_col=0)

    dm, index = censor_domains(df,args)
    if early==2: return dm,index
    
    X = dm.iloc[:,4:].to_numpy(dtype=np.float32)
    X = X[:,1:NUM_DOMS]
    
    if args.norm==0: norm = lambda x: x
    if args.norm==1: norm = lambda x: F.normalize(x,dim=0)
    if args.norm==2: norm = lambda x: F.normalize(x,dim=1)
    if args.norm==3: norm = lambda x: (x-x.mean(0))/x.std(0)
    if args.norm==4: norm = lambda x: (x-x.min(0))/(x.max(0) - x.min(0))
    
    X = torch.tensor(X)
    X = norm(X)
    
    lims = [X.shape[1]] # start and stop indices of each positional encoding.
    if PE[0]=='1':
        print('ones: ',X.shape[1],'>>> ',end='')
        X = torch.cat((X,torch.eye(X.shape[0])),axis=1)
        print(X.shape[1])
    lims.append(X.shape[1])
    if PE[1]=='1':
        print('n2v: ',X.shape[1],'>>> ',end='')
        pe = pd.read_csv('../comp_nodes/n2v.csv',index_col=0).to_numpy()
        pe = torch.tensor(pe)
        X = torch.cat((X,norm(pe)),axis=1)
        print(X.shape[1])
    lims.append(X.shape[1])
    if PE[2]=='1':
        print('le: ',X.shape[1],'>>> ',end='')
        pe = pd.read_csv('../comp_nodes/pe_le.csv',index_col=0).to_numpy()
        pe = torch.tensor(pe)
        X = torch.cat((X,norm(pe)),axis=1)
        print(X.shape[1])
    lims.append(X.shape[1])
    if PE[3]=='1':
        print('rw: ',X.shape[1],'>>> ',end='')
        pe = pd.read_csv('../comp_nodes/pe_rw.csv',index_col=0).to_numpy()
        pe = torch.tensor(pe)
        X = torch.cat((X,norm(pe)),axis=1)
        print(X.shape[1])
    lims.append(X.shape[1])
    if PE[4]=='1':
        print('nf: ',X.shape[1],'>>> ',end='')
        nf = pd.read_csv('../comp_nodes/nf.csv',index_col=0).to_numpy()
        nf = torch.tensor(nf)
        X = torch.cat((X,norm(nf)),axis=1)
        print(X.shape[1])
    lims.append(X.shape[1])
 
    y = df.lab_io.to_numpy(dtype=np.int32)

    if WEIGHT:
        e = np.array(ew.index.to_series().apply(lit_eval).to_list(),dtype=np.int32)
        w = ew.to_numpy(dtype=np.float32)
    else:
        cut = edges[:,2]>0
        w = edges[cut][:,2]
        e = edges[cut][:,[0,1]]

    self_loops = [[i,i] for i in range(X.shape[0])]
    e = np.concatenate((e,e[:,[1,0]]),axis=0)
    if args.model in ['GCN1','GCN2']: 
        e = np.concatenate((e,self_loops),axis=0)
    w = np.concatenate((w,w),axis=0)

    data = Data(x=torch.tensor(X,dtype=torch.float), edge_index=torch.tensor(e,dtype=torch.long).T, y=torch.tensor(y,dtype=torch.float), edge_attr=w)
    data.edge_attr = torch.tensor(w,dtype=torch.float,requires_grad=True)
    if args.model in ['GCN1','GCN2']: 
        data.edge_attr = torch.cat((data.edge_attr,-1*torch.ones((X.shape[0],w.shape[1]))),axis=0)
    data.num_classes = 1
    e = data.edge_index<0
    data.edge_index[e] = data.edge_index[e] % data.x.shape[0]
        
    return data, xdf, index, lims

def write_results(F1v, F1t, AUCt, model_dict, args, file_score='results_.txt'):
    res = f'prod(test): {F1t[0]*AUCt[0]:.5f}, F1(val): {F1v[0]:.5f}+/-{F1v[1]:.5f}, F1(test): {F1t[0]:.5f}+/-{F1t[1]:.5f}, AUC(test): {AUCt[0]:.5f}+/-{AUCt[1]:.5f}'
    out_file = open("results.txt", "a")  # append mode
    res_str = res + ', ' + ', '.join([str(k)+'_'+str(v) for k,v in vars(args).items()]) + ', '+str(datetime.datetime.now()).split('.')[0]+'\n'
    out_file.write(res_str)
    out_file.close()

    if args.model in ['MLP','GCN0','GCN1','GCN2']:
        model_file = 'params/' + str(int(time.time())) + '.pt'
        torch.save(model_dict,model_file)

def evaluate(x):
    try:
        res = literal_eval(x)
    except:
        res = x[1:-1].split(',')
    
    return res

col = ['screen_name', 'id', 'created_at', 'is_retweet', 'retweeted', 'mentions', 'url_exp', 'url', 'tweet_text','RT']

def ast_try(x):
    try:
        res = ast.literal_eval(x)
    except :
        res = None
    return res

def clean_io_data(data,col=col):
    try:
        data = data.loc[:,['userid', 'tweet_time', 'tweet_text', 'is_retweet','urls', 'user_mentions']]
        data[col[5]] = data.user_mentions.apply(lambda x: ast.literal_eval(x)) # mentions
        data[col[4]] = data.mentions.apply(lambda x: x[0] if len(x)>0 else []) # retweeted
    except:
        print('no user mentions...')
        data = data.loc[:,['userid', 'tweet_time', 'tweet_text', 'is_retweet','urls']] 

    data[col[0]] = data.userid # screen_name
    data[col[1]] = data.userid # id
    data[col[2]] = data.tweet_time # created_at
    data[col[3]] = data.is_retweet # is_retweet
    data[col[6]] = None # url_exp
    if type(data.urls.iloc[0])==str:
        data[col[7]] = data.urls.apply(ast_try) # url
    else:
        data[col[7]] = data.urls
    data = data[~data.url.isna()]
    data[col[8]] = data.tweet_text # tweet_text
    data['RT'] = data.tweet_text.apply(lambda x: str(x)[:2]=='RT' if len(str(x))>1 else None)
    data = expand(data,'url')
    col = [c for c in col if c in list(data.columns)]
    return data.loc[:,col]
    #return expand(data.loc[:,col],'url')
    
def dom(x):
    ext = tldextract.extract(x)
    return ext.registered_domain

def expand(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    # create expanded DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0.)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
