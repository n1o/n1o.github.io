+++ 
draft = false
date = 2024-12-16T09:17:05+01:00
title = "Graph Neural Networks meet Large Language Models"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

# Abstract
I am a huge fan of Graph Neural Networks (GNNs), and I am (bit less) a fan of Large Language Models (LLMs), however they are hard to ignore. Booth have different strengths, while GNNs excel when it comes to problems that have a inherent structure, LLMs strive in cases where we treat everything as a sequence of tokens (maybe Bytes in the future). An natural question arises, what if we can combine these two? It turns out yes, initially I encountered the [Graph Neural Prompting](https://arxiv.org/abs/2309.15427) paper during my [T5](https://n1o.github.io/awesome-t5/) journey (still not finished, just making an couple of detours) whichh is a cool idea on how to merge knowledge graphs and T5. As it turns out there is already comprehensive research done on how to merge GNNs and LLMs in the following paper: [A Survey of Graph Meets Large Language Model: Progress and Future Directions](https://arxiv.org/abs/2311.12399) and its companion github repo: [Awesome-LLMs-in-Graph-tasks](https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks) and I decided to crate this blog post to clarify how the merging of these two technologies is done.


# Why to combine GNNs and LLMs (or LLMs and GNNs)

![](/images/llms-in-graphs.png)

To answer the this questions we start by asking two additionall ones:

1. How can LLM help with a Graph problem?
2. How can GNN help an LLM?

To answer those questions lets us first analyze the cons/pros of booth GNNs and LLMs. GNNs excel when it comes to capturing structural information, this can be the structure of a document, images on the side, text divided into different sections, or source code, which has an inherit graph/tree like structure (I already did write about [GALa](https://codebreakers.re/articles/llm-and-security/galla-graph-aligned-llm) which merges GNN and LLM for Source code understanding and Generation), however they need an way to capture semantic information into their initial node embedings. On the other hand, LLMs excel at capturing semantic information of text (or source code) , however they fail (where not designed to is an better word) to capture complex hierarchical dependencies. 

Just from description above, we can clearly see an simple, but efficient solution where we use LLMs to generate the initial node embeddings for GNNs. Indeed this is powerful, and it is widely used and somewhat proven approach. However it has a shortcoming, we only use the LLM to enhance the GNN, however with this we hardly use the LLM to its full potential.

# GNNs and Pretraining

With the rise of Pretrained Language Models (PLM) and their ease of adapting them to task specific problems (transfer learning) it is just natural that research tried to achieve something similar with GNNs, however this story is somewhat different and way more complicated to achieve. In GNNs we nearly always are interested in node embeddings, we start with some initial and after N rounds of training and given the actual graph structure and model layers we end up with some final emebedings. The final embeddings than can be used for node clarification, where we put an additional model on top of the node embeddings, for link prediction, where take a pair of nodes and their embeddings and use again an additional model to produce an binary variable if there is a link (if we would have different link types we can go with softmax (for mutualy exclusive)). For Graph classification as whole the standard solution is to pool all the node embeddings to generate one graph embedding, the pooling can be taking an average over all embeddings, to more complicated hierarchical embeddings. 

For pretraining GNNs there are two common approaches:

1. **Reconstruction**, this is an classical approach where we have an encoder that embbed the graph into a latent representation, and an decoder that learns to reconstruct the original graph from the embedding, here are some examples:
    - [GraphMAE](https://arxiv.org/abs/2205.10803), which uses an Masked Graph Autoencoder
    - [S2GAE](https://www.semanticscholar.org/paper/S2GAE%3A-Self-Supervised-Graph-Autoencoders-are-with-Tan-Liu/355cf4ef8c666898ceed76ea7950c3df176900fc), again leverages an Graph Autoencoder but uses it to mask and reconstruct edges
    - [DUPLEX](https://arxiv.org/abs/2406.05391) is an example for directed graphs, we learn an complex embedding, where the real part is used to encode the existence of an edge and the imaginary part the orientation of this edge
2. **Constrastive objective**, this is very much the same as any contrastive learning objective, we try to create graph embedding that for similar graphs are close, we can than later leverage the embeddings for downstream task

# GGNs + LLMs = Better Together

There are 3 main approaches how to combine LLMs and GNNs (Or GNNs with LLMs) and they are:

1. LLM as an Enhancer
2. LLM as a Predictor
3. GNN-LLM Alignment


## LLM as an Enhancer

![](/images/llm-as-an-enhancer.png)
We already explored this approach a bit, in the part where I talked about using LLMs genering the initial node embeddings, however there is more to it and we are going to explore it in more detail. In this setting we always work with Text Attributed Graphs (TAG), this are graphs that have text node features.

### Embedding based enhancement
Yes, you guess right, we use an LLM to generate an node embedding! However there are some extra details we need to explore. First we may not want to embed the whole text, but we may decide to extract only relevant information, this approach is explored in depth in the [G-PROMPT](https://arxiv.org/pdf/2309.02848) paper. Also we need to consider if we train the embedding model, this can be beneficial since it may learn to extract better features, on other hand, it is way more resource hungy.

### Explanation based enhancement
Technically this also is an embbeding based enhancement, but we do not embed the actual text itself but an expalantion of the text that has been produced by an language model. Why would you do that? It is simple we can use proprietary language models to generate the explanation. Is this actually an good approach? Well yes and now. Yes since the proprietary language models tend to be excellent and they may provide extra information, but no since it can balloon up the costs significantly, since most API based LLMs are expensive.


## LLM as a Predictor
![](/images/llm-as-predictor.png)

Here we want to use LLM to make predictions about graph related tasks. This is an very active are of research, and overall I believe it is one that makes most sense, especially the GNN-based predictions approach.

### GNN-based Predictions

Here we have an GNN model which produces node embeddings, we than take these node embedings and we embedd them as tokens for an LLM

### Flatten-based Predictions

