---
title: "Longer Context for T5"
date: 2024-04-29T14:10:36+02:00
draft: true
---

# Why does T5 need a longer context?

In my previous post
[T5 the Old New Thing]({{< relref "posts/t5-the-old-new-thing.md" >}})
we already explored why T5 is awesome. But one downside is its limited context length of 512 tokens. Now technically this cannot be compared to the context length of an decoder-only model, since T5 is an encoder-decoder model, meaning the encoder can process input up to 512 tokens, and the decoder can generate output up to 512 tokens. Making the total context length 1024 tokens.  We will cover two extensions:
1. [LongT5](https://arxiv.org/abs/2112.07916)
2. [ColtT5](https://arxiv.org/abs/2303.09752)

Booth LongT5 and CoLT5 are eploring ways how to extend the context length of the encoder part of T5. This means that we are looking into how we are able to process longer input lengths and not necessarily generate longer texts. This approach is especially useful for problems like text summarization or document question answering.

# LongT5

Originally published in 2022 and it uses a new pretraining strategy called Pegasus and explores using Local Attention and a novel TGlobal attention in the encoder.

## Pegasus
Pegasus is a pretraining strategy pecially designed for abstract summarization, where we mask out key (principle) sentecnes from a document and we ask the model to reproduce them as a single string as if it would be a summary.

## Local Attention
This is just a sliding window attention, thus any given token is able to attend to a neighborhood of $l$ tokens.

## TGlobal
In TGlobal we partition the input tokens into chucks of length $k$, for each chuck we compute a global token by summing up the individual token embeddings. Now when we perform attention we take a Local Window $l$ and we apped to it all the global tokens.

![TGlobal](/images/t_global_attention.png)

### Cons
Since we do not perform full attention we experience some minor performance degradation, and we need an couple of extra parameters. Computation wise we compute the global tokens on the fly, but we can compute them only once per input tokens per layer and cache them.

### Pros
We are able to process way larger input lengths.

## Notes
It is worth noting that there is a variant of LongT5 that uses only Local Attention without Global Tokens. This wariant can be scaled up to even longer sequences unfortunately there is also an more profound performance drop.

# CoLT5
Paper from 2023, and it builds upon LongT5 by bring in ideas like [Mixture of Experts](https://arxiv.org/abs/2101.03961) and [Multi Query Attention](https://arxiv.org/abs/1911.02150).

## Conditional Computation

![ColT5 Attention](/images/colt5_transformer_layer.png)

The whole idea behind Conditional Computation is that not all tokens are equally important, and we do not need to spend the same amount of computation on all of them. In terms of CoLT5 we have two branches, a light brand and an heavy branch. The idea is that we apply the light branch to all the tokens on the heavy branch is applied only to important tokens. This branching happens at two places, first one is at the attention layer and the second one is at the feed forward layer.

### Attention
The light attention branch has fewer heads than the heavy branch, and the light branch uses only local attention while the heavy branch utilizes full attention.

### Feed Forward
As for the feed forward layer the light branch has an lower hidden size than the heavy branch.

### Routing
This is the meat of the Conditional Computation, how do we decide which token is important and which is not? Here we create an scoring function for each token, this scoring function takes the value of the token $X_i$ and maps it to an d-dimensional embedding. 

$$ s_i = X_i . u_d $$
- $s_i$ is the score for token $i$
- $u$ is the embedding function

Once we have the scores for every token we have to choose the top-k tokens, this is not easy since $s_i$ has a dimensionality of $d$, and we cannot just take the top-k values (and we also have to normalize this score). The authors use a iterative soft top-k algorithm from [lei2023conditional](https://arxiv.org/abs/2304.04947) and [qian2022multivector](https://arxiv.org/abs/2211.01267), TLDR this is an optimization problem where we solve the Dual Problem using Coordinate Descent.


#### Coordinate Descent
Coordinate Descent is a gradient free optimization algorithm (Yes it does not use Gradient Descent), the intuition is that if we have an function $f(x_1, \cdots, x_n)$ we can transform this problem into a single variable optimization problem by fixing all other variables except one. Than we use a root finding algorithm like [Brent Dekker](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html) (Thechinally this is a bracketing algorithm combined with secant method and quadratic interpolation) to find the optimal value for this variable. We repeat this process for all variables.

##### Least Absolute Shrinkage and Selection Operator (LASSO)
Most of you out there probably used this algorithm without even knowing it. If you ever used LASSO regression (This is just linear regression with L1 regularization) currently the fastest way to solve this problem is using Coordinate Descent. Why is this the case? Well first the L1 norm is convex but not smooth, thus we cannot use Gradient Based Optimizations, thus we can use Coordinate Descent. There are other ways to solve this problem like using Proximal Descent or Subgradient Descent (Yep L1 is not differentiable at 0 but the subgradient is) but Coordinate Descent is just faster.

### Computation

Since we have an optimization problem inside our routing mechanism we want this routing to able send signals, because of this we want the routing scores to be part of the computation graph. 

#### Feedforward
$$X_i = X_i + FFd_{Light}(X_i) + \tilde{s}_i . FFd_{Heavy}(X_i)$$
- $X_i$ is the model state at token $i$ 
- $\tilde{s}_i$ is the normalized routing score (this is 0 for non-routed tokens)

#### Attention
$$X_i = X_i = A_{Light}(X_i, X) + \tilde{s}^q_i . A_{Heavy}(X_i, \tilde{s}^{kv} . X)$$
- $X_i$ is the model state at token $i$ 
- $\tilde{s}^q_i$ are the normalized routing scores for the queries for token i set to 0 if not routed
- $\tilde{s}^{kv}$ are the normalized routing scores for the key-values for all tokens set to 0 if not routed

#### Performance 

Here we compare the performance between vanilla T5, LongT5 and CoLT5:
- T5  $12nd^2 + 2n^2d$
- LongT5 $12nd^2 + \frac{n^2}{8}d$
- ColT5 $7\frac{1}{4}nd^2 + \frac{n^2}{84}d$

## Decoder
During output generation long input setences cause an memory bandwidth bottle neck, by using Multi Query Attention (MQA) to speed up the decoding process. In MQA all the query heads share the same key-value pair.

![Grouped Attention Variants](/images/multi_head_grouped_multi_query_attention.png)

### Performance
Vanilla Multi Head Attention tends to have the highest accuracy but it requires the most memory and it is the slowest to generate, by allowing query heads to share key-value pairs we decrease the memory requirements and improve token generation speed. Unfortunately this speedup comes at a cost and we loose accuracy. 

## Cons
Where with LongT5 we have an open source implementation from [Hugging Face LongT5](https://huggingface.co/docs/transformers/model_doc/longt5) and [Google LongT5](https://github.com/google-research/longt5). Unfortunately with CoLT5 this gets tricky, I found the following repository [ColT5](https://github.com/lucidrains/CoLT5-attention) but the implementation is a best effor reproduction (and a great one at that) but as the author mentions there are some open questions about the implementation.

# Personal Thoughts

CoLT5 is a great extension to vanilla T5 and it shows a lot of promise, its biggest downside is the lack of an official implementation and the absence of a pretrained model. LongT5 on the other hand is a great extension to T5 but it start to show its limits when we start to scale up the input length to modern standards.  What I would like to see is a continuation of the CoLT5 with cool stuff as Mixtrure of Experts and Sliding Window Multi Query Attention in the decoder part. With this we would be able to efficiently process long input sequences and efficiently generate high quality outputs.

