+++ 
draft = true
date = 2024-10-28T10:12:27+01:00
title = "Distilling State Space Models from Transformers"
description = ""
slug = ""
authors = []
tags = ["NLP", "SSM", "Transformers"]
categories = []
externalLink = ""
series = ["SSM"]
+++

# Abstract

It is notoriously expensive to train a Language model from scratch, this makes any independent research impossibleand trying out new architectures extremely risky. Because of this costs, Transformer++ models like LLaMa, based on Rotary Embedding, SwiGLU, MLP, RMSNorm, without linear bias, sometimes with grouped query attention and/or sliding window attention, are the defacto standard, because they work! In previous posts I covered [SSM-Transformer Hybrids]({{< relref "posts/ssm-transformer-hybrids-guide.md" >}}). The biggest benefit of hybridization is their reduced inference const, minimal memory overhead due reduced KV cache. The biggest obstacle of these models, is that they need to be pretrained from scratch. For example [Zamba2 7B](https://www.zyphra.com/post/zamba2-7b), is a 7B model, that was trained on 128 H100 GPUs for 50 days, bringing its pretraining costs around to 600K US dolars. This makes it one of the cheapest (but by a far not weakest) SSM-Attention hybrid, however if we look into detail the model was trained only on 3T (+100B high quaility for annealing) tokens. In comparison Llama3 was trained on 15T tokens, if we would apply the same number of tokens to Zamba we would end up with costs around 3M US dolar. It is not hard to see that these kinds of budgets are out of scope for any individual, but also out of scope for many medium sized research organizations and academia.

In this post we look into the details of two distillation techniques that take an already pretrained Transformer model, and they replace some of its parts with an Mamba like State Space model. The biggest benefit benefit of this approach is that we can drastically decrease the infference costs wile avoiding the whole pretraining procedure on trilions of tokens, and still maintain the same performance level by finetuning on a few bilion of tokens.

- [(Mohawk) Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models](https://arxiv.org/abs/2408.10189)
- [Mamba in the Llama](https://arxiv.org/abs/2408.15237)

# Sequence and Channel mixers!

Lets revisit some key concepts of Transformers and Self attention. So what is self-attention, remember, for a cusal model, every output token it is just an weighted sum of the input tokens. Yes you can have more heads, but that will just give you multiple weighted sum, preferably somewhat different (each head should capture different variation in the data). If you read, which I hope you did, my post about  [Monarch Matrices]({{< relref "posts/butterflies-monarchs-hyenas-and-lightning-fast-bert.md" >}}), there I did a breakdow a Attention Block into two parts, first part is the Sequence Mixer, and the second the Channel Mixer. Self-Attention is a Sequence Mixer, it may not be the most efficient in terms of computation, quadratic in computation and need a KV cache so we do not recompute the token representations over and over each time we sample a new token. Where the MLP parts of an Attention block serve as Channel Mixers. MLP plays also an another crucial role, sice it is theoretized that it holds most of the LLMs knowledge! 

# Matrix Orientation, Hidden-State Alignment, Weight-Transfer and Knowledge Distilation (Mohawk)
The goal of Mohawk is to take a Transformer model (In our context this is a Teacher Model), and replace some (or all) self-attention parts with Mamba2 (This we will call Student Model). The knowledge transfer happens in 3 stages, where in each consecutive stage we train more parameters, and is responsible of transferring different information.

## Stage 1

In the first stage we focus on aligning the Sequence Mixers between the Student Model and the Teacher model. For example if we would distil from Llama3, we would align the output from [here](https://github.com/meta-llama/llama3/blob/main/llama/model.py#L90C7-L90C16). 

**Objective**

$$\min_{\phi}||\text{TeacherMixer}(u) - \text{StudentMixer}_{\phi}(u) ||_F$$

- $u$ is the output of the preceding layer of the Teacher

### Remark!
This optimization can run in parallel, and we can precompute the Teacher part in advance. It is crucial that booth teacher and student mixer have the same preceding transformations, which means that take input and produce outputs of the same shape.

### Modifications to Mamba

Mamba and Mamba2 are richer than the self-attention operation, since it can also act as a Channel Mixer. Because of this the authors did one modification to Mamba, which is replacing the local convolution with the identity operatoin. This nullifies its effect, and according the following [research]({{< relref "posts/ssm-the-illusion.md" >}}) it is not really needed since Mamba itself is expressivenoughh to capture this information.

## Stage 2

As mentioned in stage 1, the role of self-atttention in an attention block is to perform sequence mixing. In stage 2 we go a step further and we align the whole Attention block.

**Objective**
$$ \min_{\phi}||\text{AttentionBlock}(u) - \text{StudnetMixtureBlock}_{\phi} (u)||_2$$

Here it is worth noting, that Mamba(2) is also the channel Mixer, thus the student block essentially stays the same between stages 1 and 2. We can view Stage 2 as a correction to Stage 1, which by itself pushes the attention weight to a slightly wrong direction. As with Stage 1, we can train all the blocks in parallel, on materialized data.

### Modifications to Mamba

The authors decided to remove the Normalization layers, and they set the gate to 1, to cancel out its initial effect.

![Here sigma is the gate, by forcing it to 1 we open it](./images/mamba_layer.png)

## Stage 3
Here we are going to train the studen model as a whole, but first we transfer the remaining parts from the teach model. There remaining parts are:
- MLPs, we will freeze these parts, sice they should contain most of the models learned knowledge
- initial embedding layer
- final layer normalization
- language modelling heads
- input normalizatoin of each block

**Objective**

$$ \min_{\phi}L_{CE}(\text{TeacherModel}(x), \text{StudentModel}_{\phi}(x)) $$

This is a very common approach were the student tries to mimic the distribution of the teacher.

## Phi Mamba
To showcase the effectiveness of this approach the authors distilled two hybrid models, since they used the stellar Phi-1.5 as the teacher model, they coined the models named Phi-Mamba.


### Architecture

Phi-Mamba retains 4 attention layers, the rest is replace by Mamba2. However as mentioned above, there are some modificatoins to the original architecture, these modifications are:

- remove post convolution activation
- sets convolution to identity, this disables this feature
- remove pre-output normalizaton
- normally Mamba2 is multi value, but we change it to multi head to match the behaviour of attention
- also drops the discretization by making A purely input dependent

These moddifications are not to major, and they do not require changes to the SSD algorithm.

### Results
The actual training was done on 3 Bilion tokens! Yes bilions, not trillions!  At the beginning of the post we saw, that Zamba2 7B was pretrained on 3T tokens with the estimated budged of around 600k, by naively extrapolating we could say that Mohawk enables distilation at 0.1% of the cost, making it around 600 US Dolars. Yes sure, there are wast difference in the architecture, but this is again one of the first steps of exploring new architectures with shortcuts in pretraining.

![](./images/phi_1_5_mamb_2_results.png)

The actual token distribution was 80M for Stage 1, 160M for Stage 2 and 2.78B for Stage 3.

#### Stage Imprtance

To better gasp the importance of various stages we the authors decided to train 3 model:
- Phi-Mamba is a pure Mamba2 model
- H-Phi-Mamba is the hybrid 
- Phi is a pure transformer model, this is done by randomly reinitializing the students attention weights.

All the models are distilled with Mohawk on 5B tokens instead of 3B. 

![](./images/phi_1_5_mamb_2_stage_comparison.png)

We can see that most of the training budget was spent on Stage 3, and we spent relative little in the first 2 stages, however we can see that there are massive gains. Even wen we just apply stage 2 and 3, we get improvements to performing vanila knowledge distilation. And at last even a bit of Stage 1, is the key for the student to retain the teachers performance.

## Remarks
The idea of mixer vise alignment for knowledge distilation is still in its infance, however it is an extremely cool concept, and it enables to create new models without the need of expensive pretraining. However as we can see, there are still strick constraints, how a studen model can look like, where Mamba2 is used to mimic the behaviour of self-attention.
