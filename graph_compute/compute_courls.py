#!/usr/bin/env python

import os
import pandas as pd
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
import networkx as nx
import gc
from cutils import *

cimport cython
cimport numpy as np
import numpy as np

cpdef int[:] cython_where(int[:] data, int val):
    
    cdef int xmax = data.shape[0]
    cdef int i
    cdef int count = 0
    cdef int[::1] xind = np.zeros(xmax,dtype=np.int32)
    for i in range(xmax):
        if(data[i] == val):
            xind[count] = i
            count += 1
    return xind[0:count]

cpdef int[:,::1] calculate_C(int num_uids, int[:,:] d, int tmax):

    cdef int p = num_uids 
    cdef int[:,::1] C = np.zeros((p,p),dtype=np.int32)
    cdef int[::1] n = np.unique(d[:,1])
    cdef int u,m,i,j
    cdef int[:] di,dj
    cdef int delta
    cdef int[:] idx
    for u in n:
        idx = cython_where(d[:,1],u)
        m = idx.shape[0]
        for i in range(m):
            di = d[idx[i]]
            for j in range(i,m):
                dj = d[idx[j]]
                if di[0]==dj[0]:
                    continue
                delta = di[2] - dj[2]
                if abs(delta)<tmax:
                    if delta>0:
                        C[dj[0],di[0]] += 1
                    else:
                        C[di[0],dj[0]] += 1
    return C




# In[ ]:


def C_batch(d,num_uids,M,cmap):
    num_uids = np.int32(num_uids)
    k = d.shape[0]-1
    ds = [d[np.where(np.logical_and(d[:,1]>=d[k*i//M,1],d[:,1]<d[(k*(i+1))//M,1]))] for i in range(M)]
    
    pool = mp.Pool(M)
    res = pool.map(cmap,ds)
    
    return res


if __name__ == '__main__':

    num_uids = np.int32(4352)
    d = pd.read_csv('d.csv',index_col=0).to_numpy(dtype=np.int32)
    uids = pd.read_csv('uids.csv',index_col=0).to_numpy()
    M = 16
    T = [i for i in range(46,100)]
    for t in T:
        print(f't={t}...',end='')
        def cmap(df): return np.array(calculate_C(num_uids,df,t))
        Cb = C_batch(d,num_uids,M,cmap)
        C = pd.DataFrame(np.array(Cb,dtype=np.int32).sum(0),index=uids,columns=uids)
        C.to_csv(f'cotweets_T{t}.csv')
        print(' Done.')
        del C
        del Cb
        gc.collect()

