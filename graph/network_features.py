
#%%time
ec = data.edge_index.cpu().detach().numpy()
G = nx.from_edgelist(ec.T)
degree_dict = dict(G.degree())
print(datetime.datetime.now().time())
print('computing clustering coefficient...',end='')
cc = nx.clustering(G)
print('Done.\n')
print(datetime.datetime.now().time())
print('computing betweenness centrality...',end='')
bc = nx.betweenness_centrality(G, k=1000)
print('Done.\n')
print(datetime.datetime.now().time())
print('computing pagerank...',end='')
pr = nx.pagerank(G)
print('Done.\n')
print(datetime.datetime.now().time())
print('computing HITS...',end='')
hits = nx.hits(G)
print('Done.')

bc3 = nx.betweenness_centrality(G, k=4000)

ld = list(degree_dict.keys())
lcc = list(cc.keys())
lbc = list(bc.keys())
lpr = list(pr.keys())
lh = list(hits[0].keys())
m = data.x.shape[0]
d_vec = [degree_dict[i] if i in ld else 0 for i in range(m)]
print('0',datetime.datetime.now().time())
cc_vec = [cc[i] if i in lcc else 0 for i in range(m)]
print('1',datetime.datetime.now().time())
bc_vec = [bc[i] if i in lbc else 0 for i in range(m)]
print('2',datetime.datetime.now().time())
pr_vec = [pr[i] if i in lpr else 0 for i in range(m)]
print('3',datetime.datetime.now().time())
hits1_vec = [hits[0][i] if i in lh else 0 for i in range(m)]
hits2_vec = [hits[1][i] if i in lh else 0 for i in range(m)]
print('4',datetime.datetime.now().time())

lbc = list(bc.keys())
print('5',datetime.datetime.now().time())
bc_vec = [bc3[i] if i in lbc else 0 for i in range(m)]
print('6',datetime.datetime.now().time())

vec = [cc_vec, bc_vec, pr_vec, hits1_vec]
nf = np.array([np.array(v) for v in vec]).T
#X = df.iloc[:,7:].to_numpy(dtype=np.int32)[:,:1000]
X_df = pd.DataFrame(nf)
Xnf = (X_df - X_df.min(0))/(X_df.max(0) - X_df.min(0))
#X = np.concatenate((Xnf.to_numpy()*100,X),axis=1)

ld = list(degree_dict.keys())
lcc = list(cc.keys())
lbc = list(bc.keys())
lpr = list(pr.keys())
lh = list(hits[0].keys())
m = data.x.shape[0]
d_vec = [degree_dict[i] if i in ld else 0 for i in range(m)]
print('0',datetime.datetime.now().time())
cc_vec = [cc[i] if i in lcc else 0 for i in range(m)]
print('1',datetime.datetime.now().time())
bc_vec = [bc[i] if i in lbc else 0 for i in range(m)]
print('2',datetime.datetime.now().time())
pr_vec = [pr[i] if i in lpr else 0 for i in range(m)]
print('3',datetime.datetime.now().time())
hits1_vec = [hits[0][i] if i in lh else 0 for i in range(m)]
hits2_vec = [hits[1][i] if i in lh else 0 for i in range(m)]
print('4',datetime.datetime.now().time())

lbc = list(bc.keys())
print('5',datetime.datetime.now().time())
bc_vec = [bc3[i] if i in lbc else 0 for i in range(m)]
print('6',datetime.datetime.now().time())

vec = [d_vec, cc_vec, bc_vec, pr_vec, hits1_vec]
vec = [np.array(v) for v in vec]
nf = np.array(vec)
nf[4] += - min(nf[4]) + 1e-20
nf[0] = np.log(nf[0])
nf[1] = np.log(nf[1]+1e-15)
nf[2] = np.log(nf[2]+1e-15)
nf[3] = np.log(nf[3]+1e-20)
nf[4] = np.log(nf[4])
#X = df.iloc[:,7:].to_numpy(dtype=np.int32)[:,:1000]
Xdf = pd.DataFrame(nf.T,columns=['degree','cc','bc','pr','hits'])
Xnf = (X_df - X_df.min(0))/(X_df.max(0) - X_df.min(0))
#X = np.concatenate((Xnf.to_numpy()*100,X),axis=1)