+++ 
draft = true
date = 2024-05-29T08:46:43+02:00
title = "CodeT5 and CodeT5+"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

# Abstract
We already toucheed the surface of CodeT5 and CodeT5+ the previous post [T5 the Old New Thing]({{< relref "posts/t5-the-old-new-thing.md" >}}). Now we will dive much deeper into the topic.

I already explored [CodeBERT and GraphCodeBERT](https://codebreakers.re/articles/detail/bert-codebert-and-graphcodebert/). These models are based on [BERT](https://arxiv.org/abs/1810.04805) and [RoBERTa](https://arxiv.org/abs/1907.11692) architectures, they are great for code understanding and retrieval task, but they fall short on code generation tasks. However it is wort mentioning that they share common ideas, which is to use unique pretraining objectives, that are tailored for source code.

# CodeT5
CodeT5 was introduced by [Wang et al.(2021)](https://arxiv.org/abs/2109.00859), and it tries to bridge the gap between Encoder-Only models like CodeBERT which excel at code understanding, but fail short at code generation and Decoder-Only models like GPT which are great at code generation but perform worse at code understanding tasks. Because of this CodeT5 is a full Encoder-Decoder model, which is pretrained on a huge corpus of unimodal purse Source Code and bimodal Source Code and Natural Language pairs, using a variety of pretraining objectives, some of which are tailored for source code.

## Intuition
The implementation of CodeT5 is based on the T5 model, which is a sequence-to-sequence model, thus we formulate all our problems as sequence-to-sequence tasks

![T5](/images/codeT5Architecture.png)

Since we pretrain on a large corpus of code, we can leverage developer assigned identifier, such as comments, variable types and their names, function names, signatures and return types, and many more. To capture the semantics of these identifiers, it introduces an identifier-aware pretraining objective, which trains the model to distinguish which tokens are identifiers and how to recover them when they are masked. This is beneficial for code understanding and generation tasks, as it allows the model to learn the alignment between code and comments, and to recover identifiers when they are masked.

## Tokenization
The tokenization of the input sequence is done using a custom trained Byte Pair Encoding (BPE) tokenizer, which is a subword tokenizer that is used to split words into subwords, and it is used to build the vocabulary of 32k tokens. We also introduce a couple of special tokens ([PAD], [CLS], [SEP], [MASK0], [MASK1], ... [MASK99]) that are used to represent the input sequence. It is worth mentioning that this differs from the original T5 model, which used a SentencePiece tokenizer. The reason is simple, the behaviour of our custom trained BE tokenizer behaves better on source code, whereas the original T5 tokenizer was encoding common code tokens like  [,],{,} as uknnown tokens.

## Input/Output representation
The input representation of CodeT5 is a sequence of tokens, which is a concatenation of the Natural Language (NL) and Programming Language (PL) tokens.

$$x = ([CLS], w_1, \cdots, w_n, [SEP], c_1, \cdots, c_m, [SEP]) $$
- $[CLS]$ is a special token at the beginning of the sequence, this token as in BERT is used to represent the entire sequence, that can be later used for classification or retrieval tasks
- $[SEP]$ is a special token that separates the NL and PL tokens
- $n$ is the number of NL word tokens
- $m$ is the number of PL code tokens

### Abstract Syntax Tree (AST)
Since we want to learn code-specific features, we leverage the AST of the source code. We extract node types for each node and introduce a sequence of binary labels $y \{0,1\}^m$ that represents if code segment $c_i$ is an identifier or not.

## Pretraining
As a corpus we use the CodeSearchNet dataset, and a collection of C/C# open source Github Repositories. As for the pretraining objectives we use the following:
- Masked Span Prediction 
- Identifier Tagging
- Masked Identifier Prediction

What is worth mentioning is that Identifier Tagging and Masked Identifier Prediction use information from the AST, and it is the source of its power.

![CodeT5 pretraining](/images/code_t5_pretraining.png)

### Masked Span Prediction (MSP)
This is a well known pretraining objective that is employed by T5, where we corrupt a span of tokens and we require the decoder to recover the original tokens including sentinel tokens. The corruption rate is 15% and the span length is sampled from Uniform(1,5) and we employ whole word masking to avoid partial masking.

#### Loss
$$ L_{MSP}(\theta) = \sum_{t=1}^k - \log P_{\theta}(x_t^{mask}| x^{\backslash mask}, x_{<t}^{mask})$$
- $x^{\backslash mask}$ masked input
- $x^{mask}$ is the masked sentence to predict from the decoder 
- k is the number of tokens in $x^{mask}$
- $x^{mask}_{<t}$ span sequence generated so far

### Identifier Tagging (IT)
This pretraining objective is used to train the model to distinguish between identifiers and non-identifiers in the code, we can view it as a kind of syntax highlighting. This objective only activates the encoder, and it uses the contextual representation of the PL tokens right before they are passed to the decoder. We take these contextual representations we map them to a sequence of probabilities, unfortunately the paper does not provide the exact details how this is done, nor does the [source code](https://github.com/salesforce/CodeT5/blob/e78a61a17f6dc2f3cbb968447d3e2d065b426e7b/CodeT5/models.py). My assumptions is that they add an additional projection layer with some sort of normalization (Like L2), the output of the projection layer will be a single scalar that is than passed trough a sigmoid function, which will yield the probability that the token is an identifier.

#### Loss
$$ L_{IT}(\theta_e) = - \sum_{i=1}^m y_i \log p_i + (1 - y_i) \log (1 - p_i)$$
- $\theta_e$ are the encoder parameters

Again the paper does not mention, byt I expect they may not even choose to send the output from the encoder to the decoder, as the decoder is not used in this task.

### Masked Identifier Prediction (MIP)
This is a pretraining objective that can be viewed as a sort of de-obfuscation task, and it is based on the paper [DOBF](https://arxiv.org/abs/2102.07492). The idea is that we mask specific identifiers in the codej. The role of the decoder is than to produce the output key value pairs, where the key is the masked identifier ([MASK0, MASK1, ...]) and the value is the masked value.

#### Loss
$$L_{MIP}(\theta) = \sum_{j=1}^{|I|} - \log P_{\theta}(I_j| x^{\backslash I}, I_{<j}) $$
- $x^{\backslash I}$ is the masked input

### Bimodal Generation
During pretraining the decoder only observes masked identifiers, but we want our model to be able to generate code from Natural Language, or generate Natural Language from code. In this step we feed NL into the encoder and we expect the decoder to generate the PL, and vice versa. Since we expect to handle multiple programming languages, we prepend the input with a special token that represents the programming language (<java>, <python>, ..., or natural language <en>).

# Fine-Tuning
We expect our model to be able to handle the following tasks:

## Code summarization
Here we expect an english summary of the code snippet, and we evaluate on BLEU-4 score, and we perform better than CodeBERT and PLBART, where the later was trained on a significantly larger dataset.
## Code generation
We except the model to generate code snippets from Natural Language descriptions, we evaluate on CodeBLEU (CodeBLEU is more suitable for comparing code generation models and it takes into account lnaguage syntax, semantics and structure, jop it is Language Specific). CodeT5 performs better than GPT-2.
## Code translation
The idea is to take one programming language and translate it into an another. We evaluate on Exact Match and CodeT5 manages to generate functions that are more readalbe and maintain the same functionality as the original code. This is an indicator of strong generalization capabilities of the model. Here the performance is better than for GraphCodeBERT. 

## Code refinement

The idea is to remove bungs, here the performance is comparable to GraphCodeBERT but it lags a bit behind.

## Defect detection

The idea is to detect if a code snippet is vulnerable or not, and performance wise we beat all models including GraphCodeBERT and PLBART which where the SoTA at the time the article was published.

## Clonde detection

Here we measure the similarty of two code snippets, determining if they are the same or not.  As with Defect detection CodeT5 achieved the current SoTA score.

## Remarks
Of course performance wise we cannot really compare this model to current ones, because CodeT5 came into two variants Small (120M) and Base (220M), these sizes are already dwarfis to current 7B CodeLlama or even 22B CodeStral. 

# CodeT+
Lets talk about CodeT5+ [Wang et al. (2023)](https://arxiv.org/abs/2305.07922). The model is from the same research team and it builds on top of CodeT5 and it scales it size up to 16B parameters. There are a couple of things worth mentioning. The first is the authors are starting to get biased more towards Decoder-only models, because of this they adopt an shallow encoder and a deep decoder architecture (Decoder has more attention stacks than the Encoder). Second they note that pretraining an LLM from scratch is expensive and time consuming, they bootstrap the encoder and the decoder from existing models and they connect them with a cross-attention layer at the last layer of the decoder, and during pretraining they freeze the decoder and train only the encoder and the cross-attention layers. And at last they introduce a couple of new pretraining objectives like Text-Code Contrastive Learning and Text-Code Matching. This is quite a bit lets dive deeper into the model.

## Model

![CodeT5Plus Model](/images/code_t5_plus_model.png)

It is not that dissimilar to plain T5, boostraping it from existing models is again not that strange, what makes this a bit unique is that it can work in different modes. These modes are encoder only, decoder only or full encoder-decoder mode. Why is this? Lets imagine that we just need the embedding of the encoder to perform some code similarty search, there is no reason to use the decoder for that, similarly lets assume we just need to perform code completion, this task can be done just by the decoder. Running it in full encoder-decoder model can be extremely useful for an Retrieval Augmented Generation (RAG) task, where the contextual representation of the encoder is used for the retrieval and the decoder is used for the generation.

### Bootstrapping
For the decoder the authors choose [CodeGen-Mono](https://arxiv.org/abs/2203.13474) 350m and for the Decoder CodeGen-Mono 2B, 6B and 16B. 

## Pretraining

Since we pointed out that CodeT5+ may work in different settings, because of this we have multiple stages of pretraining, with different objectives and types of data (Uni-modal code only and Bi-modal code and natural language pairs).

## 1 Stage: Uni-modal
Here we leverage only code data, the code however may contain comments, but we do not treat them as a natural language programming language pair. Lets dive into the objectives.
### Span Denoising Objective

This is similar to the Masked Span Prediction, where we randomly replace 15% of the tokens with [MASK] tokens in the encoder input, and we require the decoder to recover them via generating a combination of these spans. The difference is that we concatenate different code files into sequences and chunk them into fixed length sequences. This is a task that activates booth the encoder and the decoder.

### Causal Language Modeling
This is an split encoder-decoder and decoder only task. What is important that in booth variants we select an random pivot location, where all the tokens before the pivot are treated as the source sequence (context) and the tokens after the pivot are the target we want to predict.

1. Seq2Seq CausalLM variant, we prepend the source sequence with a special [CLM] token and we feed it to the encoder, the rest of the sequence is the target for the decoder to generate. The pivot is somewhere between 10% and 90% of the sequence length.
2. Extreme Seq2Seq CausalLM variant, we prepend the source sequence with a special [CLM] token and we require the decoder to generate the entire sequence. Here we train the decoder kind of independently from the encoder.

## 2 Stage: Bi-modal
Here we have code-snippet natural language pairs, this improves the alignment between code and natural language improving cross modal generation and understanding tasks. Here we have 3 pretraining objectives:

### Text-Code Contrastive Learning
Here we are going to require some patience (At least I did need multiple source to get an holistic picture). The idea is that we want to learn representations of code, what we want is that code that is similar should have similar representations, and code that is different should have different representations. The goal is to align the feature space of text and code representations by pulling together positive text-code pairs and pushing apart negative pairs. This only activates the encoder, which encodes a text or code snippet into a continuous representation. An [CLS] token is prependend to the input sequence and it is used as for the final embedding of the sequence. We pass the [CLS] representation trough a linear layer and use L2 norm to map it to an 256-dimensional embedding. 

#### Momentum Encoder

We already said that we are going to need positive and negative pairs, the question is how to get them. Well we are going to use an [Momentum Encoder (MoCo)](https://arxiv.org/abs/1911.05722). Technically this is just an another encoder that is 1 to 1 to the encoder we already use. The goal of this encoder is to generate our negative samples, and it by maintaining an queue of embeddings of samples from previous mini-batches, when there are new samples it enques them and when the queue is full it dequeues the oldest samples. The parameters of the momentum encoder are updated by linear interpolation of the original encoder and the momentum encoder, this ensure the consistency of the representations across training steps. 

The goal of momentum encoder is to have access to negative samples, (samples form previous batches) but since it evolves slowly it improves the stability of the training process.

#### Remarks
I am already familiar with negative sampling, but in general to construct the negative sample we just took a random sample from the training dataset. Since the momentum encoder evoles slower than the original encoder, we force the embeddings to evolve slower as well, thus the embeddings for the negative samples won't drastically change from one batch to another. This is a nice trick to stabilize the training process.

### Text-Code Matching
Text-Code Contrastive Learning was an objective that activated only the encoder, in contrast with Text-Code Matching this is an decoder only objective.

The idea is that we want the decoder to learn to determine if two snippets share the same semantics and better align the text and code modalities. We prepend a task specific [Match] token to the code input sequence to inform the decoder of the text-code matching functionality and we append [EOS] token to the end of the code input. We take the [EOS] representation at the last decoder layer as the text-code cross-modal alignment representation. We pass this representation trough a linear layer and use it for binary matching tasks predictiv if the text-code pair is a match or not. To get the negatives we employ [negative mining strategy](https://arxiv.org/abs/1911.05722).

#### Remarks
Why is this a decoder only task? Well hard to answer since the paper does not provide an explanation, naturally this task would be a fit for our encoder, but since we already have the encoder used in the Text-Code Contrastive Learning, we can use the decoder for this task, and force them booth to learn kind of the same thing, but capturing different aspects of the problem. Also the authors argue that the model can be used in decoder-only setting, and by forcing the decoder to learn this task, it can improve its performance in this setting.

### Text-Code Causal LM
This activates booth encoder and decoder and focuses on code-to-text and text-to-code generation. If the input is text we prepend [CDec] token to the input sequence to the decoder forcing the decoder to operate in code generation mode. If the input is code we prepend [TDec] token to the input sequence to the decoder forcing the decoder to operate in text generation mode. This type of causal LM closes the gap between pretraining and finetuning for generative downstream tasks.

## Instruction Tuning
Since this is an Post ChatGPT paper we are also interested in giving our model the ability to follow instructions, for this we pretrain it using the [CodeAlcapa](https://github.com/sahil280114/codealpaca) dataset, which is a dataset of code snippets and their corresponding instructions. 

## Performance
Here things get way more complicated, the evaluation was on similar tasks as CodeT5, and in general it exceeds the performance of CodeT5, when compared in equal settings (220M versions), and the improvements are nice. However most comparisons where done to other Open Source Language Models like CodeBERT, GraphCodeBERT, PLBART which are all smaller models. In pure code generation CodeT5+ outperforms LLaMa even StarCoder thus making it the SoTA model for code generation tasks. Unfortunately the gap to proprietary models like ChatGPT is still large.

### Remarks
This whole boostraping from existing LLMs is nice, however it makes a lot of comparisons more troublesome. I would really like to see some ablation studies, on the different pretraining objectives, and how they affect the performance of the model. 

# Remarks
The models gives a nice performance improvement over CodeT5 however, CodeT5 has seen an wide adoption in the filed of Cybersecurity and it have been applied for [Automated Vulnerability Repair](https://arxiv.org/abs/2401.15468), and it is the base for [BinT5](https://arxiv.org/abs/2301.01701) and [HexT5](https://ieeexplore.ieee.org/document/10298504)

The lack of adoption of CodeT5+ may be due of its relative new age, or its permisive licencing for its 16B instruct version (Or maybe 16B is just too large), and also when compared to CodeT5 it is better, but maybe this improvement is not enough. However by scaling it up to 16B it is a really nice step forward.
