---
title: "T5 the Old New Thing"
date: 2024-03-06T12:58:32+01:00
draft: true
---

# Why T5

A couple of weeks ago I run into the following paper [Tiny Titans](https://arxiv.org/abs/2402.00841). It compares multiple smallish (up to 1B parameters) open source LLMs with bigger proprietary ones on meeting summarization. TLDR; the small models tend to perform worse in zero-shot setting as well after fine-tunnig than big ones. Except for FLAN-T5-Large which after finetuning performs way beyond its league, beating even the biggest proprietary models (GPT-3.5).

This is not the first time I read an research paper that used T5 or build upon it. Some examples are:
- [CodeT5](https://arxiv.org/abs/2109.00859)
- [CodeT5+](https://arxiv.org/abs/2305.07922)
- [AST-T5](https://arxiv.org/abs/2401.03003)
- [LENS](https://arxiv.org/abs/2402.03646)

All this research tickled my curiosity and I decided to learn more about T5 and its applications. In this blog post I will cover the following papers:

- [Original T5 Paper](https://arxiv.org/abs/1910.10683)
- [FLAN](https://arxiv.org/abs/2210.11416)
- [UL2](https://arxiv.org/abs/2205.05131)

# Introducing T5
Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer is a paper introduced by Google in October 2019. The paper introduces a new approach to transfer learning in NLP called "text-to-text" transfer learning. 

![T5](/images/text-to-text.png)

The core idea of T5 is to pretrain a model on a large corpus of text, and then fine-tune the model on a wide range of NLP tasks by converting the task into a text-to-text problem. This novel approach achieves state-of-the-art results on a wide range of NLP tasks.

In comparison the previous state of the art in NLP was dominated by models that are trained on a single task and then fine-tuned on the target task. This approach has several drawbacks, such as the need to collect and label a large dataset for each task, and the need to fine-tune the model for each task. 

Lets given an example, and take a look an hypothetical text classification task. In the traditional approach, we would take an model like BERT, pretrain it on a large corpus of text, and then introduce an classification head on top of the model. In the text-to-text approach, we pretrain the model on a large corpus of text, and then fine-tune the model on the text classification task by converting the task into a text-to-text problem. For example, we could convert the text classification task into a text generation task by adding a prefix to the input text that specifies the task, and then training the model to generate the label. The downside of the later is that the label may be made up by the model (Halucinations where a thing 5 years ago as well). 

There is an obvious benefit to the text-to-text approach as it does not require introducing a new architecture (or additional layers) for each task. This makes the model more flexible and easier to train.

## Model
As for the actual model, T5 is an encoder-decoder transformer model that is more or less the same as introduced by [vaswani2023attention](https://arxiv.org/abs/1706.03762). 

The encoder uses a fully visible attention mask, which means that each token can attend to all other tokens in the input sequence. The decoder uses a causal attention mask, this is known from GPT like models, and means that each token can only attend to tokens that come before it in the output sequence, and has an additional cross-attention layer that allows the decoder to attend to the encoder output. As for positional encoding, T5 uses an relative positional encoding, where encode the relative position between queries and keys, and translate them into buckets. As for Layer Normalization we scale to unit variance but do not shift to zero mean. As for the Feed Feed Forward Network, it is a 2-layer ReLU network with an hidden size greater than the input size, with dropout before the activation and a dropout after the last projection.
Booth attention and feed forward layers have a residual connection and are followed by layer normalizationddddd.

For an mockup of the model, you can check the following colaboratory notebook: [T5 Mockup](https://colab.research.google.com/drive/1ynn1R5FWEZWxGkSDbKlymtFYmIR-Yj-h?usp=sharing)
Here I do not implement the Positional encoding (you can find it here: [Positional Buckets](https://github.com/huggingface/transformers/blob/9acce7de1cb8229304a467938ebb47727d60cdb2/src/transformers/models/t5/modeling_t5.py#L390C1-L390C108)) but the rest should be more or less be the same.

## Objective
The training objective is Span Corruption, where the model is trained to predict the original text given a corrupted version of the text. The corrupted version is created by replacing a span of text with a special token, and the model is trained to predict the original text given the corrupted text. This training objective is used to pretrain the model on a large corpus of text, and then fine-tune the model on a wide range of NLP tasks.

## Performance
Obviosly the state-of-the art results are no longer state-of-the art, but one important thing to note is the runtime requirements of T5, or encoder-decoder models compared to autoregressive models. 

Technically if we have a encoder-decoder model with B parameters (encoder + decoder parameters) and an autoregressive model with B parameters, the runtime of the decoder-encoder model will be B/2 compared to the autoregressive model. This is because half of the model parameters in the encoder-decoder model belong to the encoder, and it is used to process the input sequence, and the other half of the model parameters belong to the decoder, which are then used to generate the output. In contrast, all of the model parameters in the autoregressive model are used to generate the output, thus the runtime of an autoregressive model will be twice as long as the runtime of an encoder-decoder model with the same number of parameters.

If we analyze it further we see that the encoder processes only the input and it is enough to do it once. The decoder in an Encoder-Decoder model has 1/2 the amount of parameters than in an Encoder-only setting, thus the generation will be twice as fast.

Note: There is no rule that encoder and decoder have to be of same size, for example CodeT5 employs an shallow encoder and and deep decoder. In this case it is obvious that performance will be similar to an Autoregressive model.

# FLAN
FLAN takes T5 and introduces multiple finetuning tasks, including Chain of Though instruction tuning (it is worth noting that this paper was first published in October 2022 before ChatGPT was introduced to the public in November 2022). The idea is to have this promnt guided general intelligence model that can be used in a wide range of tasks.

## Back to Tiny Titans

Since T5 uses an fully visible encoder where tokens can attend to all other tokens, it is no surprise that the model works well for text summarization, since it has access to the whole input text. FLAN also introduced CoT finetuning which further improves the performance of all other downstream tasks.

# UL2

UL2 follows on T5 but extends it by introducing multiple pretraining objectives.

## Mixture of Denoisers
We introduce a mixture of denoising objectives, each of the objectives is designed to force the model to learn different types of knowledge.

### R-Denosing
This is the pain old denosing objective introduced by T5, where we corrupt a "shortish" span of the input text and the model is trained to predict the original text given the corrupted text. We corrupt spans of 2 to 5 tokens, around 15% of the text. This objective forces the language model to learn useful knowledge, not necessarily forcing it to generate fluent text.

### X-Denosing
This is a new denosing objective introduced by UL2, where we corrupt a "longish" span of the input text and the model is trained to predict the original text given the corrupted text. The idea is that weforce the model to be able to generate long text given an limited information. We corrupt spans of 12 or more tokens, around 50% of the text.

### S-Denosing
This is essentially the autoregressive LLM objective, where we corrupt the input text by removing all the tokens that come after a certain token, we then train the model to reconstruct the original text given the corrupted text. This objective forces the model to generate fluent text.

# To other relating papers
We already saw that UL2 introduced multiple pretraining objectives, we can push this further by introducing additional task specific pretrainig objectives. 

## CodeT5 and CodeT5+
In CodeT5 we 
## AST-T5
Here we have a special pretraining objective that includes the abstract syntax tree of the code, this forces the model to learn information about the structure of the code

## LENS
Introduces multiple pretrainig objectives that are helpful for network traffic analysis.

# Takeaway
T5 is a powerful base model, that is easily extendable by introducing additional pretraining objectives. These additional pretraining objectives can force our model to learn additional context, that can be leverged in more specific tasks. We can teoretically apply additional pretraining objectives also in Causal LLMs, however I havent seen any research in this direction (yet, but again there is a lot of going on in the field of LLMs and I am hardly an expert)

Its runtime requirements are also 1/2 of an autoregressive model with the same parameter.
