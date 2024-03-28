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

All this research tickled my curiosity and I decided to learn more about T5 and its applications. In this blog post I will cover (to some extent) the following papers:

- [Original T5 Paper](https://arxiv.org/abs/1910.10683)
- [FLAN](https://arxiv.org/abs/2210.11416)
- [UL2](https://arxiv.org/abs/2205.05131)

I will also mention other papers that are related to T5, but I will not go into details about them.

# Introducing T5
Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer is a paper introduced by Google in October 2019. The paper introduces a new approach to transfer learning in NLP called "text-to-text" transfer learning. 

![T5](/images/text-to-text.png)

The core idea of T5 is to pretrain a model on a large corpus of text, and then fine-tune the model on a wide range of NLP tasks by converting the task into a text-to-text problem. This novel approach achieves state-of-the-art results on a wide range of NLP tasks.

In comparison the previous state of the art in NLP was dominated by models that are trained on a single task and then fine-tuned on the target task. This approach has several drawbacks, such as the need to collect and label a large dataset for each task, and the need to fine-tune the model for each task. 

Lets given an example, and take a look an hypothetical text classification task. In the traditional approach, we would take an model like BERT, pretrain it on a large corpus of text, and then introduce an classification head on top of the model. In the text-to-text approach, we pretrain the model on a large corpus of text, and then fine-tune the model on the text classification task by converting the task into a text-to-text problem. For example, we could convert the text classification task into a text generation task by adding a prefix to the input text that specifies the task, and then training the model to generate the label. The downside of the later is that the label may be made up by the model (Halucinations where a thing 5 years ago as well). 

There is an obvious benefit to the text-to-text approach as it does not require introducing a new architecture (or additional layers) for each task. This makes the model more flexible and easier to train. And also an obvious downside that it is hard to control the output of the model.

## Model
As for the actual model, T5 is an encoder-decoder transformer model that is more or less the same as introduced by [vaswani2023attention](https://arxiv.org/abs/1706.03762). 

The encoder uses a fully visible attention mask, which means that each token can attend to all other tokens in the input sequence. The decoder uses a causal attention mask, this is known from GPT like models, and means that each token can only attend to tokens that come before it in the output sequence, and has an additional cross-attention layer that allows the decoder to attend to the encoder output. 
As for positional encoding, T5 uses an relative positional encoding, where encode the relative position between queries and keys, and translate them into buckets. As for Layer Normalization we scale to unit variance but do not shift to zero mean. As for the Feed Feed Forward Network, it is a 2-layer ReLU network with an hidden size greater than the input size, with dropout before the activation and a dropout after the last projection.
Booth attention and feed forward layers have a residual connection and are followed by layer normalization.

For an mockup of the model, you can check the following colaboratory notebook: [T5 Mockup](https://colab.research.google.com/drive/1ynn1R5FWEZWxGkSDbKlymtFYmIR-Yj-h?usp=sharing)
Here I do not implement the Positional encoding (you can find it here: [Positional Buckets](https://github.com/huggingface/transformers/blob/9acce7de1cb8229304a467938ebb47727d60cdb2/src/transformers/models/t5/modeling_t5.py#L390C1-L390C108)) but the rest should be more or less be the same.

Note: T5 being an encoder-decoder model, it does not mean we have to use it as an encoder-decoder model. We can allway just use the encoder by chopping of the decoder part, and introduce an additional classification head on top of the encoder output.

## Objective
The training objective is Span Corruption, where the model is trained to predict the original text given a corrupted version of the text. The corrupted version is created by replacing a span of text with a special token, and the model is trained to predict the original text given the corrupted text. This training objective is used to pretrain the model on a large corpus of text, and then fine-tune the model on a wide range of NLP tasks.

## Performance
Obviosly the state-of-the art results are no longer state-of-the art, but one important thing to note is the runtime requirements of T5, or more specifically encoder-decoder models compared to autoregressive models. 

Lets assume we have an balanced encoder-decoder model, where the encoder and decoder have the same number of parameters. The role of the encoder is to process the input sequence (Prompt) and this is done only once, while the role of the decoder is to generate the output sequence, and this is done, token by token. If we compare this to an autoregressive model, this model has only a decoder, if this decoder has the same number of parameters as the whole encoder-decoder model, the runtime of the autoregressive model will be twice as long as the runtime of the encoder-decoder model. The reason is that the decoder in an autoregressive model is twice as big as the decoder in an encoder-decoder model.

Note: There is no rule that encoder and decoder have to be of same size, for example CodeT5+ employs an shallow encoder and and deep decoder. In this case it is obvious that performance will be similar to an Autoregressive model.

# FLAN

FLAN takes T5 and introduces multiple finetuning tasks, including Chain of Though (CoT) instruction tuning (it is worth noting that this paper was first published in October 2022 before ChatGPT was introduced to the public in November 2022). The idea is to have this prompt guided general intelligence model that can be used in a wide range of tasks, and with the help of CoT finetuning we get a step-by-step reasoning model, that improves the performance of all other downstream tasks.

## Back to Tiny Titans

Lets get back to the Tiny Titans paper, FLAN-T5 is the only encoder-decoder model they tested. Since the encoder uses a fully visible attention mask, it is no surprise that the model works well for text summarization task. And since FLAN finetuning dataset involves a lot of summarization tasks, it is no surprise that the model works well for summarization tasks.

# UL2
UL2 follows on T5 but extends it by introducing a new Mixture of Denoisers pretraining objective.

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

## CodeT5 CodeT5+
CodeT5 introduces two additional pretraining objectives: Indenifier Tagging and Masked Identifier Prediction.
### Identifier Tagging
The goal is to teach the model knowledge of whether if a code token is a identifier or not, and it can be viewed as sort of syntax hihglihting.

### Masked Identifier Prediction
Here we mask a random identifier and replace all its occurrences by a sentinel token. It can be viewed as a sort of code obfuscation, where if we change a name of a identifier it has no impact on the code. Technically this should teach the model to perform deobfuscation.

## CodeT5
CodeT5+ builds up on top of CodeT5 and it also involves instruction tuning and employs an shallow encoder and and deep decoder. 

We also have specific pretraining objectives depending if we do unimodal or bimodal pretraining. Bimodal pretraining involves both text and code and it was also used in CodeT5 to give the model ability to generate code from text (and vice versa). However for CodeT5 this reduced the performance of the model on code-to-code tasks like code translation from one programming language to another or code defect detection. 

As for the bimodal pretraining objectives for CodeT5+ we have the following objectives:

### Text-Code Contrastive Learning
Here we have positive and negative code text pairs, the idea is that for the positive samples the code and text should be close together in the representation space. This task is only activates the encoder which is used to encode the text-code snippets into an continuous representation space.
### Text-Code Matching
This actives only the decoder and it should predict whether the code and text share the same semantics. It should enable the model capture fine-grained semantic information between code and text.
### Text-Code Causal LM
This activates booth the encoder and decoder and it should help the model generate code from text and vice versa.

## AST-T5
This is an another extension of T5 for code where we leverage the Abstract Syntax Tree of the code. A new AST-Aware Subtree Corruption objective is introduced, where corrupt tokens with respect to the corresponding subtree of the AST for the code snippet.

## LENS
This paper explores application of T5 in network security and it is a beast (I am planning to make an full blog post about it!). But long story short the paper introduces multiple embedding strategies and pretraining objectives. We wont go into details about the embedding strategies (It involves Payload Header Embedding and Packet Segment Embedding), but we will cover the pretraining objectives.

### Packet Order Prediction
This pretrains only the Encoder, where we try to teach the model to learn the natural order of the packets in the network traffic. This is done by corrupting the order of the packets and the model is trained to predict the original order of the packets.
### Homologous Traffic Prediction
As with Packet Order Prediction this objective is applied only to the Encoder and it is inspired by [ET-BERT](https://arxiv.org/abs/2202.06335). The idea is to give the model the ability to capture the difference between different types of network traffic. 

Note: This paper is kind of an opposite to CodeT5+ in the sense they push way more emphasis on the encoder than the decoder.

# Takeaway
T5 is a powerful base model, that is easily extendable by introducing additional pretraining objectives. These additional pretraining objectives can force our model to learn additional context, that can be leverged in more specific tasks. We can teoretically apply additional pretraining objectives also in Causal LLMs, however I haven't seen any research in this direction (yet, but again there is a lot of going on in the field of LLMs and I am hardly an expert)

Its runtime requirements are also 1/2 of an autoregressive model (If we have an balanced encoder decoder pair) with the same parameter.
