+++ 
draft = false
date = 2025-02-16T14:48:13+01:00
title = "TLDR; HC-GAE The Hierarchical Cluster-based Graph Auto-Encoder for Graph Representation Learning"
description = ""
slug = ""
authors = []
tags = ["TLDR", "GNN", "subgraph", "hierarchical graph auto-encoder" ]
categories = ["TLDR", "subgraph", "GNN" , "hierarchical graph auto-encoder"]
externalLink = ""
series = ["TLDR", "subgraph", "GNN","hierarchical graph auto-encoder"  ]
+++

# Source
- Paper link: https://arxiv.org/abs/2405.14742
- Source Code: https://github.com/JonathanGXu/HC-GAE

# Abstract
Graph Representations learning is an essential topic in Graph ML, and it is all about compressing an whole Graph (arbitrary large) into a fixed representation. Usually these techniques leverage Graph Auto Encoders, whichh are train in an self-supervised fashions. This is all good however they usually focus on node feature reconstruction, and they tend to lose the topological information that input Graph encodes.

# **H**ierarchical **C**luster-based **G**raph **A**uto **E**ncoder (HC-GAE)
An approach that learns graph representations, by encoding node features but also the topology of the Graph. This is done by a encoder-decoder architecture, where encoder operates in multiple steps, in each step we compress the input graph into a collection of subgraphs. The goal of the decoder is to reverse this process and recover the original graph, with the correct topology and the correct node features.

![](/images/hc_gae.png)

## Encoder
Encoder consists of a bunch of layers, where each layer can be characterized by two processes:

1. Subgraph Assignment
2. Coarsening

### Subgraph Assignment
The idea is that we have an input graph which is compressed into an output graph, by assigning one or more nodes from the input graph, to a single node in the output graph.  The assignment works in two steps:

1. Soft Assignment
$$ S_{soft} =
    \begin{cases}
    softmax(GNN(X^{(l)}, A^{(l)})) & \text{if } l = 1 \\
    softmax(X^{(l)}) & \text{if } l > 1
    \end{cases}
$$
- $S_{soft} \in R^{n_{(l)} \times n_{(l+1)}}$

This just calculates the probability that a node $i$ from the input graphs will belong to node $j$ in the output graph

2. Hard Assignment
$$
S^{(l)}(i, j) =
\begin{cases}
1 & \text{if } S_{soft}(i, j) = \max_{\forall j \in n_{l+1}} [S_{soft}(i, :)] \\
0 & \text{otherwise}
\end{cases}
$$
Nothing fancy we just take the maximum, this enforces that the input graph is partitioned into a bunch of Subgraphs.

A another view on this problem is that we learn an mapping between two Graph Adjacency matrices:

$$A^{(l+1)} = S^{(l)^T} A^{(l)}S^{(l)}$$


### Croasening
Once we have the learned subgraph partitioning, we learn the node representations for croasened graph:

$$Z_j^{(l)} = A_j^{(l)}X_j^{(l)}W_j^{(l)} $$

This is just Graph Convolution Network (GCN), where we aggreagte information from a neighborhood, and since we operate on Subgraphs we do not need to wory about over-smoothing. Given the the learned representations we derive the node features:

$$X^{(l+1)} = Reorder[\underset{j=1}{\overset{n_{l+1}}{\parallel}} s_j^{(l)^\top}] Z^{(l)}$$
- $s_j^{(l)} = softmax(A_j^{(l)} X_j^{(l)} D_j^{(l)})$ this are just mixing weights
- we need to REORDER so we use the correct weight with the correct embedding

### Final Graph Representation

## Decoder
- this reverses the graph compression in multiple layers, here we will use soft assignment, we learn an re-assignment matrix
$$  \bar{S}^{l'} \in R^{n_{(l')} \times n_{(l' +1)}}$$
- in this case $n_{(l')} < n_{(l'+1)}$

$$ \bar{S}^{(l')} = softmax(GNN_{l', re}(X'^{(l')}, A'^{(l')})) $$

$$ \bar{Z}^{(l')} = GNN_{l', emb}(X'^{(l')}, A'^{(l')}) $$

- $GNN_{re}, GNN_{emb}$ are two GNN decoder that do not share parameterss
- we compute $A^{(l')}$ same as in the encoder, but here we increase the dimensionss with later layer

$$ X'^{(l'+1)} = \bar{S}^{(l')^\top} \bar{Z}^{(l')} $$

$$ A'^{(l'+1)} = \bar{S}^{(l')^\top} A'^{(l')} \bar{S}^{(l')} $$
## Loss



