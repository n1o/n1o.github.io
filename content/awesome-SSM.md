---
title: "Awesome SSM"
date: 2024-09-05T10:39:29+01:00
draft: false
---

This series will cover a bunch of posts about State Space Models, their extensions and applications.

# Basics
- [Mamba, Mamba2]({{< relref "posts/from-mamba-to-mamba2.md" >}})

# Bidirectional
- [Hydra]({{< relref "posts/hydra-a-double-headed-mamba.md" >}})

# Attention Hybrids
- [SSM-Transformer Hybrids]({{< relref "posts/ssm-transformer-hybrids-guide.md" >}}) covers:
    - [An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887)
    - [SAMBA Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/html/2406.07522v1)
    - [Jamba A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887v1)
    - [Jamba 1.5 Hybrid Transformer-Mamba Models at Scale](https://arxiv.org/abs/2408.12570)
    - [Zamba A Compact 7B SSM Hybrid Model](https://arxiv.org/abs/2405.16712v1)
    - [Zamba2-small](https://www.zyphra.com/post/zamba2-small)
    - [Zamba2-mini](https://www.zyphra.com/post/zamba2-mini)
- [Hymba]({{< relref "posts/hymba-new-ssm-att-hybrid-breed.md" >}})
    - [Hymba: A Hybrid-head Architecture for Small Language Models](https://www.arxiv.org/abs/2411.13676)
    - instead of having Attention and Mamba running in alternating layers we run them in parallel in the same layer
    
# Theory and Limitations
- [Illusion of State in SSMs like Mamba]({{< relref "posts/ssm-the-illusion.md" >}}) covers:
    - [The Expressive Capacity of State Space Models: A Formal Language Perspective](https://www.semanticscholar.org/paper/The-Expressive-Capacity-of-State-Space-Models%3A-A-Sarrof-Veitsman/e7f47e8393c697696a3fccd9ff906dfdb49fe736)
    - [The Illusion of State in State-Space Models](https://www.semanticscholar.org/paper/The-Illusion-of-State-in-State-Space-Models-Merrill-Petty/917479a7a72ee7c1fb320c14d770e30ef322ef28)

# With Graphs
- [Graph Mamba: Towards Learning on Graphs with State Space Models](https://www.semanticscholar.org/paper/Graph-Mamba%3A-Towards-Learning-on-Graphs-with-State-Behrouz-Hashemi/2dda6da7375bf5e8bcf60f87b17ba10757f3bc57)
    - we leverage SSMs an alternative to Message Passing in Graph Neural Networks

# Distillation
- [Hohawk and Mamba in the Llama]({{< relref "posts/distilling-ssm-from-transformers.md" >}}) covers:

    - idea is to take an pretrained transformer and distill it into a SSM

# Reinforcement Learning
- [Decision Mamba: Reinforcement Learning via Sequence Modeling with Selective State Spaces](https://www.semanticscholar.org/paper/Decision-Mamba%3A-Reinforcement-Learning-via-Sequence-Ota/9b8130a2a5d3398f4993f540ddd01d440d99d62e)
    - apply SSMs to Sequential Decision Making
