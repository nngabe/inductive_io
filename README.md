# inductive_io

Sample code for _``Inductive detection of Influence Operations via Graph Learning"_ [[arXiv](https://arxiv.org/abs/2305.16544)]

An overview of the [framework](##figure) can be found in the figure below,

## Contents:

**train.py**  
Train and evaluate models with the following steps:
1. Initialize model: LR, RF, MLP, GCN, GCN_MPs, or GCN_MP
2. Load PE specified by binary string (one-hot, node2vec, lap-eig, random-walk, net-feat)
3. Load dataset specified by train and test binary string
4. Train and log progress, including best performance
5. Compute empirical baseline for Integrated Gradients (IG) 
6. Compute and log IG values for individual features and feature types

**/models**

* models.py: Modules for LR, RF, MLP, GCN, GCN_MPs, GCN_MP, as well as training utilities.
  
**/utils**

* utils.py: utility functions for loading, saving, and plotting data.
* twitter_api.py: wrapper functions for making requests from the Twitter API by username or uid.

**/graph_compute**

* compute_courls.py: iterate over sorted list of integers representing unique URLs in 12M tweets. Uses Cython and multithreading. 
* node2vec.py: script for computing node2vec embeddings using graph data.
* positional_encoding.py: script for computing random walk positional encoding (RWPE) and laplacian eigenvectors positional encoding (LE) using graph data.
* network_features.py: script for computing network features (degree, clustering coefficient, betweenness, PageRank, HITS).

**batch.sh**
Iterates train.py script over command line arguments written to file by gen_args.py.

## Figure
<img width="953" alt="inductive_io" src="https://github.com/nngabe/inductive_io/assets/50005216/62a9e715-30f8-47da-98cc-f5e90f21ef85">


