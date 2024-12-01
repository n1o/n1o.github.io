+++ 
draft = false
date = 2024-12-02T07:00:00+01:00
title = "Hymba, a new breed of SSM-Attention Hybrids"
description = ""
slug = ""
authors = []
tags = ["NLP", "SSM", "Transformers"]
categories = []
externalLink = ""
+++


# Abstract

State space models are really close to my heard, I even have a [dedicated page]({{< relref "awesome-ssm" >}}) about them. But when it comes to Language Models they lack some performance and that gave an rise to [SSM-Attention Hybrids]({{< relref "posts/ssm-transformer-hybrids-guide.md" >}}). Until now it was conventions that in hybrid layers you sequentially combine Mamba(2) with Attention layers. Nvidia introduced [Hymba](https://www.arxiv.org/abs/2411.13676) this paper changes the game, by using Attention and Mamba2 in the same layer, each of them processing the same input tokens. The authors also give an human memory like analogy why this approach is supperior, which they also show in the benchmarks. And since Nvidia pushed this further they also introduced special Meta tokens, which alleviate Attention sink issues and its over-focus on BOS token and also setting up Mambas hidden state before it even starts processing the data.


# Hymba

![](/images/hymba_model.png)

There are 3 important aspects to Hymba:
1. Mamba Attention Fusion
2. Meta Tokens
3. KV Cache Sharing

Here I oredered them by my subjective importance, lets look at each of them more in detail:

## Mamba Attention Fusion

A normal SSM-Attention hybrid alternates between Attention and Mamba(2) between didfferent layers, and in most cases we have multiple consecutive Mamba layers. The main reason for this Mamba is more efficient and we just need only a couple of Attention layers to achieve (even overcome) the performance of an pure Attention model. Hymba does this differently, they use Attention and Mamba2 at the same time on the same layer. With this booth process the same input tokens,  here we can view this as running multi head attention, but some heads are not processed with self attention but with SSM. 

What are the benefits? 

1. **Attention and SSM are complementary**, since Attention is stateless it provides an crystal clear recall however this stateless nature is inefficient since we need to cache previous tokens! Mamba2 has a global state, which serves as an compressed long term memory, however this memory fades with time and may forget details. 
2. **Increased efficiency**, previous research shown that replacing Attention with Mamba leads to increased throughputt and reduced memory requirements. This is also true if we replace a couple of attention heads with Mamba, instead of whole layers, however fusion gives an additionall benefit. We can replace full Attention with Sliding Window Attention (SWA). In SWA we constrain how far can attention look back, with this we risk that we may forget past information, but since we process the tokens with Mamba as well, we can leave the long range dependencies to it.

![](/images/hymba_attention_map.png)

The picture above makes this even more clear, it includes Meta tokens and we cover them in a minute, but for now it is enough to say that they are always before: the prompt, or a sliding window if we process long sequences.

If we look at Mamba we can clearly see that the importance of prior tokens diminish as we move father in the sequence, SWA takes into account only the most recent tokens, if we combine these two we get an excellent short term recall and a fading long range memory. And since meta tokens are always present we cannot forget those!

### Math
I am kind of a math nerd, so I have to include some equations, lets define the output of the fused layer:


$$Y = W_{\text{out}}(\beta_1 \text{norm}(M_{\text{att}}\tilde{X}) + \beta_2 \text{norm}(M_{\text{ssm}} \tilde{X})) $$
- $\beta_1, \beta_2$ are learnable vectors that rescale each channel from the SSM and Attention output
- $ \tilde{X} = [R, X] = [r_1, \cdots, r_{128}, x_1, x_2, \cdots, x_{n}] $, R are the Meta tokens, yep there is 128 of those!

- $M_{\text{att}} = \text{softmax}(QK^T)V$
- $M_{\text{ssm}} = G \odot \alpha(A,B,C,\Delta) W^{\text{SSM}} \tilde{X}$
- $G$ is the output gate
- $A,B,C,\Delta$ are the SSM parameters

One thing that stands out is the extra normalization of the SSM and Attention output, this is needed because the output of Mamba tends to be larger than the output from Attention especially in later layers.

![](/images/hymba_normalization.png)

### Fusion Strategy
There is a tradeof between how much Attention and Mamba to use, more Attention larger cache, slover to generate, with more Mamba we have a risk of not picking up the right information from a local context.

![](/images/hymba_attention_ssm_ratio.png)

This graph above shows what is the penalty we pay (sometimes we gain) by using less Mamba (more Attention heads) or less Attention (more Mamba heads). Mamba is more important in the initial layers, these first layers are theoretized are more important for language modeling with later layers are more useful for the next token prediction task.

I mentioned before that we use SWA, however that is not strictly true we also have full attention, and that exactly in 3 places: First, Middle and Final layers.

## Meta Tokens

As mentioned somewhere above, meta tokens are fixed tokens that are always prepended to prompt, and we can view it as a general models knowledge. Why is this useful? Gain as before meta tokens as any tokens are process at the same time by Attention and Mamba. For Mamba they are responsible to bootstrap the initial hidden state, for Attention they are responsible to mitigate Attention Sinks. What is that? Well attention tends put too much weight on tokens that are not too much important, lets look an example:

![](/images/hymba_attention_scores_distribution.png)

Lets start with Llama3, we can see that most of the attention weight is at the Beginning Of Sentence (BOS) token. Now this makes little sense, since semantically the first token in the prompt has nearly no value (maybe we should start all of our prompts Please, or My Life Depends on this, etc). In Hymba we can clearly see that we pay more attention to Cross tokens (below the main diagonal) and to Self tokens (hidden state) for Mamba2.

There is also an comparison to Jamba, whichh is an sequential SSM-Attention hybrid. We can clearly see that the Attention suffers from the same overfixation on BOS, which the the Mamba layer tries to ease out by focusing on the rest, but at the price of not really modeling long range dependencies well!

### Activations

Lets do an small breakdown how the meta tokens are actually used:

![](/images/hymba_meta_token_importance.png)

We can see the attention weight for different task, we can clearly see that there is an difference, indicating that different problems leverage different information stored in meta tokens.

#### Remark
Here I see an clear research direction, where we can provide different meta tokens for different tasks, technically this is already being researched in [Graph Neural Prompting](https://arxiv.org/abs/2309.15427) where we prepent special tokens that are constructed from an knowledge graph and in [GALLa](https://www.arxiv.org/abs/2409.04183) which encodes an graph created from an programming language snippet as special tokens at the beginning of the prompt.

## KV Cache Sharing
This is the last part, and I guess the least ground breaking. Long story short we share the KV cache between two adjacent layers.

What is however more important, the overall performance gain from fusion, SWA and cache sharing that the KV cache is nearly 20x smaller than in a pure Attention model and also we achieve nearly 3x higher throughput!

# Model

Here is the overall model:

![](/images/hymba_model_detailed.png)

## Benchmarks
Af for the performance benchmarks, we have a couple of them, first to asses the real performance we pretrain multiple models on 100B tokens from SmolLM-Corpus

![](/images/hymba_vs_other.png)

### Needle in the Haystack

Performance on retrieving a specific value "needle" from he input "haystack":

![](/images/hymba_needle_in_the_haystack.png)

### Instruction Tuning

By taking further and applying Directed Preference Optimization (DPO) we can compare its performance to current state of the art sub 2B language models, where Hymba is the clear winner.
![](./images/hymba_instruct.png)


# Recap

With fusing Mamba2 and Attention on the same layer we get a synergical effect. Mamba is responsible for long range recall, and Attention serves as a short term memory with perfect recall. Since Attention is not responsible to model long range dependencies we use Sliding Window Attention, except in the first, middle and last layer. By incorporating meta tokens we avoid Attention sinks in our attention layers, and make focus more actually important tokens, and we as well fortify the long range modelling behaviour of Mamba. By introducing an shared KV cache between 2 adjacentSWA layers we achieve an 20x times reduced cache size and the overall throughput is 3x higher than an comparable vanila Attention model.


# Final impressions

When I first found about this paper, I was super excited, and as took the paper and dug into the details my excitement just kept growing. 
