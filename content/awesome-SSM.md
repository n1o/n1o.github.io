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

# Theory and Limitations
- [The Expressive Capacity of State Space Models: A Formal Language Perspective](https://www.semanticscholar.org/paper/The-Expressive-Capacity-of-State-Space-Models%3A-A-Sarrof-Veitsman/e7f47e8393c697696a3fccd9ff906dfdb49fe736)
    - look at SSMs from a lens of regular languages
- [The Illusion of State in State-Space Models](https://www.semanticscholar.org/paper/The-Illusion-of-State-in-State-Space-Models-Merrill-Petty/917479a7a72ee7c1fb320c14d770e30ef322ef28)
    - look at the limitations of SSMs, especially when it comes to tracking state in Chess, Code and other domains

# With Graphs
- [Graph Mamba: Towards Learning on Graphs with State Space Models](https://www.semanticscholar.org/paper/Graph-Mamba%3A-Towards-Learning-on-Graphs-with-State-Behrouz-Hashemi/2dda6da7375bf5e8bcf60f87b17ba10757f3bc57)
    - we leverage SSMs an alternative to Message Passing in Graph Neural Networks

# Distillation
- [Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models](https://browse.arxiv.org/abs/2408.10189v1)
    - idea is to take an pretrained transformer and distill it into a SSM

# Reinforcement Learning
- [Decision Mamba: Reinforcement Learning via Sequence Modeling with Selective State Spaces](https://www.semanticscholar.org/paper/Decision-Mamba%3A-Reinforcement-Learning-via-Sequence-Ota/9b8130a2a5d3398f4993f540ddd01d440d99d62e)
    - apply SSMs to Sequential Decision Making
