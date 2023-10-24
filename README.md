# inductive_io

Sample code for ``Inductive detection of Influence Operations via Graph Learning" [paper]: https://arxiv.org/abs/2305.16544

## Contents:

###train.py: 
  
  Train and evaluate models with the following steps:
    1. Initialize model (LR, RF, MLP, GCN, GCN_MPs, GCN_MP)'
    2. Load PE specified by binary string (one-hot, node2vec, lap-eig, random-walk, net-feat)
    3. Load dataset specified by train and test binary string
    4. Train and log progress, including best performance
    5. Compute empirical baseline for Integrated Gradients 
    6. Compute and log IG values for individual features and feature types

**models:**
  
**utils:**

**graph_compute:**
