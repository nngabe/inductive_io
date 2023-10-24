import itertools
import sys


# take the cartesian product of all opts and write 
# arg strings to file for training:
# e.g. python train.py --model MLP --dropout 0.6 --pe 01111 

print('options: gnn,base,var,all,test')
batch = sys.argv[1]
OPTS = []

if batch=='exp':
    OUT_FILE = 'exp.txt'
    opts = {'model':['MLP'], 'pe':['01111'], 'doms': [2500], 'train':['100','011'], 'test':['100'], 'gamma': [.2,.25,.3,.35,.4,.45,.5,.55,.6]}
    OPTS.append(opts)
    opts = {'model':['MLP'], 'pe':['01111'], 'doms': [2500], 'train':['010','101'], 'test':['010'], 'gamma': [.2,.25,.3,.35,.4,.45,.5,.55,.6]}
    OPTS.append(opts)
    opts = {'model':['MLP'], 'pe':['01111'], 'doms': [2500], 'train':['001','110'], 'test':['001'], 'gamma': [.2,.25,.3,.35,.4,.45,.5,.55,.6]}
    OPTS.append(opts)

elif batch=='gnn':
    OUT_FILE = 'gnn.txt'
    opts = {'model': ['GCN0','GCN1','GCN2'], 'pe': ['01111','11111'], 'dropout':[.6], 'hidden': [64], 'train': ['001'], 'test': ['001']}
elif batch=='base':
    OUT_FILE = 'base.txt'
    opts = {'model': ['LR','RF','XGB','MLP'], 'pe': ['01111','01001','00111','00000'], 'train': ['010'], 'test': ['010']}
        
print(f'out file: {OUT_FILE}')

if __name__ == '__main__':
   
    OUT_FILE = sys.argv[1] if len(sys.argv)>1 else 'args.txt'
    if len(OPTS)==0:
        vals = list(itertools.product(*opts.values()))
        args = [''.join([f'--{k} {str(v[i])} ' for i,k in enumerate(opts)]) for v in vals]
    elif len(OPTS)>0:
        vals = []
        args = []
        for opts in OPTS: 
            vals = list(itertools.product(*opts.values()))
            args += [''.join([f'--{k} {str(v[i])} ' for i,k in enumerate(opts)]) for v in vals]
    with open(OUT_FILE,'w') as fp: fp.write('\n'.join(args))
