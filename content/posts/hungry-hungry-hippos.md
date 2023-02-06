+++ 
draft = true
date = 2023-02-06T09:39:03+01:00
title = "Paper overview: Hungry Hungry Hippos: Towards Language Modeling with State Space Models"
description = ""
slug = ""
authors = []
tags = ["NLP"]
categories = []
externalLink = ""
series = []
+++

# High level overview
We take State Space Models (SSM) and we combine them with Attention. As a result we get a model from which we can generate text more efficiently (roughly 1.6x times faster) and we save some space (Yay we can have bigger models on existing hardware).

# Language modeling requirements

During writing this document ChatGPT is riding high on the hype train. In theory it is nothing fancy it is just a [Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html). But hey, it is doing a great job, but why? In the following [paper](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) they argue that the mayority of in-context learning capacity of the Transformer architecture can be tested by the following two tests:

1. Induction Head Task Test. This tests how well a model can recall context after a special token. At the end of the sequence the model must recall the token that appeared immediatly afther the special token earlier in the sequence.
2. Active Recall Task Test. This is similar to induction head but requires the model to remember multiple key-value pairs

Attention, which is the core building block of Transformers, has booth Inductive Head and Active Recall capabilities. It can compare tokens by constructing the quadratic attention matrix $QK^T$ and it can recall tokens by directly copying, multiplying $\text{softmax}(QK^T)$ with $V$.

Thus if we can design a model that would scores high in these two test we would expect it to also perform well in language modeling.

# State Space Models
This is just a [Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model) where the hidden state is continuous. Most people know States Space Models (SSM) trough Kalman Filtering, which is an exact algorithm for Bayesian filtering in a Linear-Gaussian SSM.

Since we are living in a world full of neural networks,  we can express SSM as the following layer:
  
$$y = SSM_{A,B,C,D}(u)$$
- A,B,C,D are parameters that are learned using a gradient based optimizer

If we peek inside this layer we would find:

$$x_i = Ax_{i-1} + Bu_i $$
$$y_i = Cx_i + Du_i $$

- $x$ is our hidden state
- $u$ is an input from the user
- $y$ is an output

## Benefits
SSM allow for efficient generation since the next entry depends only on the current state, and we can extrapolate larger sequences than seen durring training.

## Downsides
The recurrent behaviour of SSM introduces a lot of IO overhead, and results in inefficient hardware utilization (Cache miss). However we can mitigate this by employing convolution.

## Convolution and Fast Fourier Transform (FFT)

For efficient traning we can write the entire sequence of input $u_1, \cdots, u_N$, the output sequence $y_1, \cdots, y_N$ as a confolution of the input with the filter f

  $$f = [CA^0B, CAB, CA^2B, \cdots, CA^{N-1}B]$$

Given an initial condition $x_0$ we get

  $$y_i = CA^iBx_0 + (f*u)_i + Du_i$$
  - $(f*u)$ is the linear convolution

More generally any linear time-invariant system (SSM is a special case) can be expressed as a convolution.

### FFT
Convolution is still pretty expensive $O(N^2)$ however we can speed it up using FFT. 

$$(f * u) = iFFT(FFT(f) \odot FFT(u))$$

Esentially we take the FFT of booth f, and u, multiply and take the inverse the FFT. This brings down the computational costs to $O(N \log N)$.

# H3 layer
We define an SSM + Attention hybrid. We start with projecting our input $u$ into $Q,V,K$ the output is defined as:

$$ Q \odot SSM_{diag}(SSM_{shift}(K) \odot V)$$

![H3](/images/h3_layer.png)

Shift and diagonal SSM are desinged to address the capacity to log tokens afther particular events.

## Shift SSM
We contrain $A \in R^{m\times m}$ to be a shift matrix

  $$A_{ij} = \begin{cases} 1 & \text{if }  i-1 = j \\ 0 & \text{otherwise} \end{cases} $$

By shfiting the hidden state down by one we create a memory of previous states.

## Diagonal SSM
Constrains A to be diagonal initialized from a diagonal version of [Hippo](https://arxiv.org/abs/2206.11893), this allows to remember tokens aftherwards for the rest of the sequence

## Efficiency
H3 scales $O(N \log N)$ for a sequence of length N where Attention requires $O(N^2)$ time and $O(N^2)$ space

# Disclaimer

I did read the original paper, and look at the code that was supplied with it. I did write this article on my own, however since I am not a native speaker, I did use ChatGPT for proof reading and improving my writing. If you are really interested how the original was worded. I will provide a link below.

# Sources
1. https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html
2. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
3. https://en.wikipedia.org/wiki/Hidden_Markov_model
4. https://arxiv.org/abs/2206.11893
5. https://arxiv.org/abs/2212.14052