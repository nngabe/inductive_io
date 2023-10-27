# inductive_io

Repository for _``Inductive detection of Influence Operations via Graph Learning"_ [[arXiv](https://arxiv.org/abs/2305.16544)]

An overview of the framework can be found in the [figure](#figure) below. 

## Contents:

**train.py**  

Train and evaluate models with the following steps:
1. Initialize model: LR, RF, MLP, GCN, GCN_MPs, or GCN_MP
2. Load up to five positional encodings specified by binary string 
3. Load train/val/test datasets specified by train/val/test binary strings
4. Train and log progress, including best performance
5. Compute empirical baseline for Integrated Gradients (IG)
6. Compute and log IG values for individual features and feature types

**/models**

* models.py: Modules for LR, RF, MLP, GCN, GCN_MPs, GCN_MP, as well as training utilities.
  
**/utils**

* utils.py: utility Functions for loading, saving, and plotting data.
* twitter_api.py: Wrapper functions for making requests from the Twitter API by username or uid.

**/graph_compute**

* compute_courls.py: Iterate over sorted list of integers representing unique URLs in 12M tweets. Uses Cython and multithreading. 
* node2vec.py: Computes node2vec embeddings and writes to file.
* positional_encoding.py: Computes random walk positional encoding (RWPE) and laplacian eigenvectors positional encoding (LE) and writes to file.
* network_features.py: Computes network features (degree, clustering coefficient, betweenness, PageRank, HITS) and writes to file.

## Figure
<img width="953" alt="inductive_io" src="https://github.com/nngabe/inductive_io/assets/50005216/62a9e715-30f8-47da-98cc-f5e90f21ef85">


