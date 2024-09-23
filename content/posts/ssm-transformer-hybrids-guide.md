+++ 
draft = false
date = 2024-09-18T11:26:25+02:00
title = "Mamba(2) and Transformer Hybrids: An Overview"
description = ""
slug = ""
authors = []
tags = ["NLP", "SSM", "Transformers"]
categories = []
externalLink = ""
series = []
+++

# Abstract
We already looked into [Mamba and Mamba2]({{< relref "posts/from-mamba-to-mamba2.md" >}}). In turns of efficiency, with their linear complexity and the absence of Key-Value cache they are an large improvement over Attention based models, in terms of throughput and memory usage. However not everything is perfect. Transformers have an certain advantage when it comes to in context learning.  In context learning is the ability to adapt the model without retraining it. This is done by providing relevant (or not) context to the model, in form of a prompt. 

In this post we cover the following papers and blog posts:

1. [An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887)
2. [SAMBA Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/html/2406.07522v1)
3. [Jamba A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887v1)
4. [Jamba 1.5 Hybrid Transformer-Mamba Models at Scale](https://arxiv.org/abs/2408.12570)
5. [Zamba A Compact 7B SSM Hybrid Model](https://arxiv.org/abs/2405.16712v1)
6. [Zamba2-small](https://www.zyphra.com/post/zamba2-small)
7. [Zamba2-mini](https://www.zyphra.com/post/zamba2-mini)

# An Empirical Study of Mamba-based Language Models

I already mentioned that the biggest shortcoming of Mamba and Mamba2 is their lack of in context learning capabilities. To showcase this lets look at the following table:

![Mambas vs Transformers](/images/mamba_transformer_benchamrks.png)

Overall I am not an huge fan of this kind of academic benchmarks. All 3 models have been pretrained on the same data of 1.1T of tokens and have 8B parameters. If you look close at the table, we can see in most cases the models perform similarly except for the MMLU benchmark. 

The MMLU (Massive Multitask Language Understanding) benchmark aims to measure the general language understanding and reasoning capabilities of models by testing them on questions that require knowledge and reasoning across different fields. (STEM, History, Literature, etc).

From the table it is obvious that Mamba and Mamba2 perform more than 10 points below Transformers (Under transformers we can imagine an Llama like model). But what is more interesting is the 5-Shot setting. In the 5-Shot setting we provide the model with 5 examples of the same task and ask it to perform the same task on a new example. We can see that for Transformers there is an noticeable improvement, but for Mamba and Mamba2 the performance is almost the same.

## MMLU in detail

The accuracy of MMLU is calculated by prompting the model with a question that includes four answer choices A,B,C,D and the model should choose one. In this case what happens to Mamba(2) that it gets confused and provides an wrong answer. If we look at an different formulation called "cloze", where the model is asked to answer the question instead of choosing and answer, the performance of Mamba(2) drastically improves to the same level as Transformers.

![Clozed MMLU](/images/mamba_2_transformers_mlu.png)

### Conclusion

Since the performance of the cloze task variant is comparable for Transformer and Mamba(2) models. Because of this we can actually conclude that booth Mamba(2) and Transformers contain the same knowledge and reasoning capabilities. The main difference is that Mamba is struggling with the precise incontext formulation of the MMLU task.

Why does Transformers perform so good? Transformers are stateless and they can easily copy the answer from the context. Where SSMs are stateful and they may struggle to determine what to keep and what to discard from their hidden state.

### More Data

There is an laternative approach to improve the performance of Mamba-2 model, and that is to pretrain it on more data. The following table compares Mamba-2 (only 2 since it is more efficient to pretrain) with Transformers booth pretrained on 3.5T tokens and booth have 8B parameters.

![MMLU More Data](/images/transformer_mamba_mmlu_more_data.png)

We can see that we came close in terms of 0-shot performance, but there is still an gap in the 5-shot performance. This just fortifies the point that SSMs are not as good as Transformers in in-context learning.

## Hybrid Models

Just from the numbers above we can see that an pure SSM model does perform really good, but it lacks in-context learning capabilities of Transformers. An hybrid model would retain the best of booth world. If we design an hybrid model we have a couple of options each influencing the performance of the model:

1. The ratio of SSM to Attention
    - the more SSM layers we have the faster the model will be faster and more memory efficient
    - with more Attention layers the memory will need an larger KV cache and there is the quadratic complexity of the Attention mechanism
2. The type of Attention
    - Full Attention is the most accurate however has quadratic complexity 
    - Sliding Window Attention gives us linear complexity but for long context it is less accurate than Full Attention
    - Choice between Grouped Query/Multi Head Attention etc
3. The type of SSM
    - we can choose between Mamba and Mamba2
    - Mamba2 is way more compute efficient and more Expressive given its larger hidden state, however we may not need the extra expressiveness
4. Positional Encoding
    - SSMs do not use positional encoding since they do not need it
    - Transformers need positional encoding since Attention is permutation invariant
5. The number of MLP layers
    - Transformers use MLPs for mixing information across channels, SSMs do not need this since booth sequence and channel mixing is done by the SSM
    - MLPs are supper efficient to compute

## Sample 8B Mamba2-Attention Hybrid Model

![](/images/ssm_attention_hybrid_sample_arch.png)

This is an sample architecture where:
- we 24 (Mamba2 + Attention) layers with 4 Layers of Attention and 20 Layers of Mamba2
- we 28 of MLP layers
- we use Grouped Query Attention with 8 groups on 32 heads
- Mamba2 has State Dimension of 128
- we use no positional encoding
- GeLU activations
- Skip connections and RMSNorm before Mamba2 and Attention layers

### Pretraining
Long story short the pretraining speed (FLOP utilizatoin) is equal to the pretraining speed of an 8B pure Transformer model.

### Efficiency
The hybrid can generate tokens 8x faster, compared to pure Transformer model. Since we use only 4 Attention layers, the KV cache should be significantly smaller than in an pure Transformer model. Unfortunately there are no concrete numbers, however we going to see for later models that there tend to by an 10x reduction in the size of KV cache.

### Context Extension
In general the accuracy for pure Transformer models tend to drop as we approach the edge of the context window. With the hybrid model we can go slightly beyod the context window (for a 4k context window we can go up to 5.5K tokens) and still retain near perfect accuracy.
And since we do not use any positional encoding we do not have to worry about theta scaling, NTK scaling, LongRoPE or any other positional encoding tricks.

### Benchmarks
One of the main reasons for hybrid models is to improve the incontext capabilities of SSMs. Lets look at some benchmarks:

![Hybrid MMLU](/images/hybrid_mmlu.png)

Long story short the hybrid model performs better than the pure Transformer model across all benchmarks. We can already see that by just adding 4 Attention layers we gave the model in-context learning capabilities.


#### Long Context

One of the main selling points of SSMs is their efficiency, which is especially important for long sequences.

![Hybrid Long Context](/images/ssm_hybrid_long_context.png)

Here we see that hybrid models lag behind pure SSM models. One reason for this is the actual formulation of the long context problem which is done by concatenating random sentences. This approach may confuse the hybrid model, since it sees random sentences and i may struggle to determine what to keep and what to discart from its hidden state. An simple reformulation of the problem, where we first ask a question and than we provide the context could help to address this issue, but the approach above is standard for transformer models.

## Conclusion

We showcased the shortcomingsof pure SSM models and explored the benefits of hybrid models. The general consensus is that hybrid models offer an nice uplift in performance to pure SSM or pure Transformer models. Especially in the in-context learning capabilities, where pure SSMs struggle with the precise formulation of the task. However the hybrid models are far from perfect and they strugge with long context problems, where there is a lot of random information that the model has to process. 

# Samba
Samba is an interesting approach to hybrid models, especially because it uses the same 3.5T pretraining data as [Phi3](https://arxiv.org/abs/2404.14219).
Just in hindsight, if we compare Samba to other Hybrids, it uses way more Attention layers than the remaining models. 

## Model

The core of the model is a Samba block. This is made of 4 parts:

![Samba](/images/samba_architecture.png)

1. Mamba
2. MLP 
3. Sliding Window Attention (2048 window size)
4. MLP

From this we already see that we have the same amount of Mamba layers as Attention layers. In total we have 12 blocks as above and 48 layers in total. There is an Residual connection between blocks, and an Pre-Norm layer before each block. The MLP blocks utilize SwiGLU activations, and we have RoPE positional encoding at the beginning of the model.

## Attention

Using Sliding Window Attention (SWA) instead of full attention is rather unique for a hybrid model. The main argument for SWA its stable perplexity during context extension, where with Full Attention the perplexity kept increasing. In my opinion the main culprit here is the RoPE positional encoding, which requires tricks for context extension. 

It is worth mentioning that when compared to pure Attention models, we can have SWA with 2x less query heads and still have the same performance.

### Remarks
SWA is a great approach how to reduce the quadratic complexity of Attention mechanism, to linear complexity. However since the model cannot look beyond the window size, it may struggle especially with long context problems. On the other hand, the Mamba layers are stateful, and its dynamics is determined by its hiddne state and the current token. By mixing Mamba with SWA, the Mamba layer has now access to more concrete information about the previously seen tokens.

## Performance
It is on par with Phi3 or better on all benchmarks.

![](/images/phi_3_vs_samba.png)

However it has greater throughput, which is most noticeable at context beyond 8K and by having half as many attentoin layers, with fewer query heads than Phi3 the size of KV cache should be smaller.

### Long Context
The last two benchmarks in the table above relate to Long Context understanding, in booth we are on par with Phi3. 
On the synthetic PassKey retrieval benchmark Samba achieves near perfect accuracy up to context length of 256K tokens. 

## Final Thoughts

The model by itself looks fine, just getting an model comparable to Phi3 is an great achievement. However the downside is that the model is not [publicly available](https://github.com/microsoft/Samba/issues/4).

# Jamba and Jamba 1.5

Booth Jamba and Jamba 1.5 bring Mixture of Experts (MoE) into the mix. What is MoE? The basic idea is that we trade extra memory for faster token generation without loosing quality. This is done by introducing extra parameters (experts) in the MLP layers. Each time we perform channel mixing with these MLP layers we choose an subset (top-k experts) of these parameters to generate the next token. Mixture of Experts where pupularized with LLMs with [Mixtral (2023)](https://mistral.ai/news/mixtral-of-experts/) however their roots with Transformers started (At least what I know of) with [Switch Transformers (2021)](https://arxiv.org/abs/2101.03961) and the concept dates back to [Adaptive Mixture of Experts (1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)



## Model
![](/images/jamba_arch.png)

Model is centered around a Jamba block. That is a block of 8 layers, from which 7 are Mamba layers and 1 s an Full Attention (Grouped Query Attention) layer. Each Mamba and Attention Layer start with an RMSNorm followed by Mamba or Attention, with an RMSNorm again and at last an MLP (SwiGLU activation), with Residual connections in the mix. In an Jamba block we replace the MLP with MoE (top 2 from 16 experts) every two layers.

### Positional Encoding

Since the model starts with a Mamba layer, we do not use explicit positional encoding. It is implicitly done by the first Mamba layer.

## Context Length

The officially released model support context length up to 256K tokens, however the paper showed that the model can be scaled up to 1M tokens. What is more important that Jamba that has 52B parameters from which 12B are active can be used on a single 80GB GPU with context length up to 128K tokens.

## Efficiency

### KV Cache
Since the whole model contains just a few Attention blocks the KV cache for 256K context length with 16 bit precision is just 4GB, where Mistral or Mixtral requires 32GB and Llama2 7B requires 128GB.

### Throughput
The throughput is comparable to Llama2 or Mixtral for a short context but it gets much higher after we pass 32K context length threshold.

## Benchmarks

![](./images/jamba_benchmarks.png)

In most cases it is on par with other Open Source Models. However it lags in GSM8K (Math) behind.
For the synthetic Needle-in-a-haystack we have nearly perfect accuracy up to 256K tokens. 

## Stability

The authors report that when scaling Mamba based model abve 1.3B parameters they experienced Loss Spikes.

![Mamba loss Spikes](/images/mamba_loss_spikes.png)


to stabilize the training they introduced the RMSNorms inside the Mamba layer.

## Jamba 1.5

Jamba 1.5 comes in two parts. The first is the original Jamba but with additional instruction finetunnig, pushingits benchmark results higher. And Jamba 1.5 Large is just Jamba scaled up to 398B parameters from which is 52B active at a time, with larger hidden state of 8192 and 64 Query Head for 8 Key-Value head.

### Mamba2

Jamba 1.5 was released after the release of Mamba2, and overall Mamba2 is more power than Mamba and we would expect they will use it. However the authors claim, that Mamba2 is more expressive due its larger hidden state, this extra expressivity is less important when we have Full Attention in the system.  Personally I look forward for future research from different AI labs if they confirm this viewpoint.

### Quantization
Having an 398B model is not cheap to run, because of this Jamba 1.5 Large comes with an custom Quantization Technique that stores the MoE weights in Int8. These weights that just in time dequantized to BF16 during infference, which happens in a fused kernel with minimal overhead.

### Training 
Here we have multiple training stages, each stage trying to teach the model some different behaviour.

#### Stage 1
The first stage uses publicly available data with cutoff March 2024 and it contains multiple languages, and the goal is to teach the model general knowledge.

#### Stage 2
In this stage we pretrain on long documents giving the model long range capabilities.

#### Stage 3
They heavily lean into synthetic data, with the assumption that it high quality, and the idea is to teach the model specific skills like: TableQA, DocumentQA, Tool Use and Steerability (model should be helpful to draft documents)


#### Remarks
During training they do not use Reinforced Learning (Proximal Policy optimization) nor Direct preference optimization just supervised finetunning. Also the model shows remarkable multilingual capabilities even trough only the first stage contained data in other than English.

#### Stability
The model was trained in BF16 for which the maximum value is 64L. However the authors experienced explosing of some activation losses which where way beyond the maximum value. Because of this they introduced an Activation Loss Penalty (L2 norm) to keep the activations reasonably small.

### Performance
![Jamba 1.5 Benchmarks](/images/jamba_1_5_benchmarks.png)

Again as with Jamba we see strong performance comparable with leading open source models but with significantly smaller memory footprint and higher throughput especially at large context lengths.

## Final Thoughts

Jamba 1.5 mini is an great example of an Small Language Model (SLM) that is small enough to run on a Single GPU with 128K context. This is especially suited to extract information, (structured Generation) from large documents.

I just there would be a Nano version that could be run on consumer grade hardware as well :(.

# Zamba and Zamba2

Zamba is somewhat similar to Samba, but with the distinction that it tries to use Attention as little as possible. Also Zamba(1) was pretrained on a budget or roughly 200K which is copared to models like LLama, Mistral or Phi is extremely cheap. Currently there are 3 versions, the original Zamba which is a 7B bilion model leveraging Mamba1. Zamba2 which leverages Mamba2 and comes in two sizes a 2.7B and a 1.2B mini version. There are some distinction between the models they have an unique feature, sharing of attention layers.

## Model

![](/images/zamba_model.png)

This model is quite different form what have we seen before. A Zamba block contains two parts.
1. Part, we start with a Mamba Block that is made of 6 consecutive Mamba layers, after the Mamba block we have a residual connection where we concatenate the output of the block with the original input, which is fed into the shared Full attention layer followed by an MLP. 
2. Part, here we take the output from the previous part, and we elementwise combine them with the output form the previous Mamba block and we feed it into a new Mamba block. The output of the Mamba block is concatenated with the original input and fed into the same shared attention layer as in part 1 followed by an MLP. And I forgot that there is an Linear Layer aftereach Attention layer.

Overall we have 7B parameters, with 13 Attention blocks, each of them having separate KV caches, and with RMSNorm layers (The paper does not state it but I would expect them to be after the Attention and MLP layer).

## Training
Before we go into detail, I would like to praise the research team, they gave away a lot of details on the actual pretraining, especially the details on the anealing phase of the training.

### Dataset
Nothing groundbreaking, mix of owever heavily deduplicated, and synthetic data.

### Phase 1
Here they use 950B of publicly available data (C4, Pile, Arxiv and PeS2o). We use Cosine learning rate decay, however the learning rate decay significantly faster than pure Transformer models. Pretraining is done on 128 H100 GPUs for 30 days, (this roughly costs 200K USD which is extremely cheap compared to other models), using ZeRO-1 optimizer. ZeRO-1 partitions the optimizer states across multiple GPUs, and updates only an subset of these states, while keeping a full copy of the model on each GPU.

### Phase 2
This phase uses high quality Curicullum data, consisting of instruction and synthetic data, in total 50B tokens. What is worth mentioning that the authors restarted the learning rate decay, with exponential decay, which is way faster than in Phase 1. The argument of using fast decay is that we have high quality data we force the model to pay more attention to it. (This seems like an common approach with continuous pretraining, where we repeat the pretraining on new data)

## Performance

![](/images/zamba_performance.png)

The model lags the current state of the art models (Mistral and Llama3) but it is on par with Llama2. The main argument is that the new model have been pretrained on wastly more data (Llama3 on 15T tokens) potentially higher quality. This seems an common theme among new models, from smaller AI labs that just do not have the access to the high quality data as the Big and established.

## Zamba2

Zamba2 is an extension of Zamba which leverages Mamba2 instead of Mamba. There are also additional changes, depending on the size of the model.

### Zamba2-Small (2.7B)

![](/images/zamba2_model.png)

The main differences besides Mamba2 and the model size are the following:

- We have two shared attention layers
- The MLP blocks also include an LoRA projection speciallization layers. LoRA is an efficient way of finetuning models on a budget. The idea here is somewhat similar, by including a projection layers we can specialize the MLP layers (remember they are shared) at the cost of only a few extra parameter per layer.

## Zamba2-Mini (1.2B)

This is really an small model, especially suited for edge computing. Compared to Zamba2 Small there are a couple of differences:

- Trained on wastly more tokens (3T + 100B ofr annealing)
- It has only 1 shared attention block but still keeps the LoRA projections
- Uses RoPE positional encoding before the Attention Blocks. This is particularly interesting since Mamba should be enough to handle the positional encoding, and it is not applied at the beginning of the model, but before the attention blocks.

### Remarks on Small and Mini

These two models have huge potential since they can run on consumer grade hardware and on edge with unparallel price performance ratio. They are publicly available, and I can't wait to take them for a spin.

## Final Thoughts
The idea behind the Zamba family can be sumarized as: "One Attention layer is all you need". The model is an great example of pushing Mamba(2) to its limits and unique application of LoRA and shared attention to keep the extra requirements on top of SSMs at bay.

# Conclusion
Over hybridizing Mamba(2) with Attention is an great approach to make the model more efficient by retaining the same accuracy. However there is no free lunch and it may require to rething the prompting strategies a bit, formulating the goal at the beginning of prompt (maybe even mentioning it multiple times). The obvious improvements are gained by running hybrid at settings where we need to process an large context but we have constrained resources.
