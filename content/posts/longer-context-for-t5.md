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

### Routing

## Decoder
During output generation long input setences cause an memory bandwidth bottle neck, by using Multi Query Attention (MQA) to speed up the decoding process. In MQA all the query heads share the same key-value pair.

![Grouped Attention Variants](/images/klu-gqa-grouped-query-attention.webp)

### Performance
Vanilla Multi Head Attention tends to have the highest accuracy but it requires the most memory and it is the slowest to generate, by allowing query heads to share key-value pairs we decrease the memory requirements and improve token generation speed. Unfortunately this speedup comes at a cost and we loose accuracy. 


## Cons
Where with LongT5 we have an open source implementation at hug
