+++ 
raft = false
date = 2024-11-21T10:44:33+01:00
title = "Transform LLMs to Encoders"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

# Abstract

In the last two years there has been a surge of Large Language Models. This is understandable, since LLMs are amazing at generating, well text, and a lot of things can be viewed as text. However generation is not always all we need, sometimes we want to have semantically rich representations. In general LLMs are not the best tool to get semantically rich representation, and even today models like BERT are used to achieve state of the art text embedings. A natural question is to ask why is BERT so much better at embeddings than LLMs? Here are the two main reason:

1. **Causal vs bidirectional attetion**, in causal attention the contextual representation of a tokens i determined by the current token and all (or a window with sliding window attention) previous tokens. With bidirectional attention, we also allow influence from future tokens. This makes intuitive sense, that if we want to see importance, or meaning of a word in a sequence, it makes sense to take into account words that come before and after.
2. **Next token prediction vs Masked Token Prediction**, decoder models are trained to predict the next token given the previous. This is obviously very efficient if we want to generate semantically valid text. Encoder models are trained on Masked Token prediction, which means we replace a random sample (usually 15%) of tokens with a special [MASK] token. Its goal is to recover the masked token, and it is done by leveraging information from the neighborhood of the masked token.

It is not hard to see that causal models are meantt to be good at predicting future tokens, while encoder models are designed to capture information about their surrounding. A natural question to ask is: *"If Bidirectional models are so good why do we not stick to it, and train them to get better embedding models?"*. Again there are a couple of reasons:

1. **Sample efficiency of LLMs**, lets start here with an example sentence: "I think this article is really cool.". It does not matter if this is true or not, we can show who an causal LLM is trained:
useful
$$ p(., "I") $$
$$ p(.| "think", "I")$$
$$ p(.| "this", "think", "I")$$
$$ p(.| "article", "this", "think", "I")$$
$$ p(.| "is", "article", "this" ,"think", "I")$$
$$ p(.| "really" ,"is", "article", "this" ,"think", "I")$$

We can see that a single sentence produces multiple training examples, here $p(.|)$ says we want to observe the next word.  (Sure LLMs operate on tokens not words but the itunition still applies). 

For Masked Token Prediction the examples would be:

$$ p([MASK]| "I", "think", "this", [MASK], "is", "really", "cool") $$
$$ p([MASK]| "I", "think", "this", "article", "is", "really", [MASK]) $$

Here we want to predict [MASK], given the surrounding, and usually we use a sentence only a couple of times. Technically we could reuse the setence as many times as we want, but it is usually not done.

2. **LLMs are more useful**, here we may argue a bit, but maybe not. ChatGPT, Clause, or other Generative models, made an impact on knowledge workers, and have potential to transform a lot of aspects of it. On the other hand embedding models are useful but they are constrained to certain subjects like similarity detection, information retrieval and classification, all useful but the overall impact is somewhat less profound.

Lets recap the text above: LLMs are more efficient to train and are more flexible ultimately yielding them more useful, but for some special cases we need an encoder model to get rich representations. A natural question is: "Can we train an LLM and somehow make it so that can produce rich semantical representations?". The answer is of course we can, and we will cover the current state of the art how to do it:

1. [LLM2Vec: Large Language Models are Secretly Powerful Text Encoders](https://arxiv.org/abs/2404.05961)
2. NV-Embed
3. Large Language Models Are Overparameterized Text Encoders

## Remark
There is also a family of Encoder-Decoder models like [T5](). These models model are excellent at providing rich embeddings and they are great at generating text. So are they perfect? Yes they are, but (of course there is a but) they are way less efficient if you use them as an conversational agent. In a decoder only language model, you can cache the previous token representation, since they do not change if we keep adding tokens to the end. With an encoder-decoder models, we need to always reevaluate the entire input before we pass to the decoder, even if we only add a single token at the end. Ok so this is not 100% true, we could append the extra tokens just to the decoder, but that will the expressivity of the model, 

## Sequence Embeddings
Before we look into the main research papers, we have to clarify what an sequence embedding is. It is a function that takes an arbitrary long input and transforms it into a fixed length output.

$$ f: X \in R^{l \times d} \rightarrow Y \in R^d $$
- $l$ is the length of the original sequence
- $d$ is the dimension of this sequence

In human language, we take an arbitrary long text and we transform it to a fixed sized vector. A common way to get an sequence embbeding is just to average individual entries out, however with tokens this is not very fortunate, since different tokens can contribute more to an overall representation. A better ways is to take a weighted average, where we use an learnable weight function.

# LLM2Vec
LLM2Vec is an unsupervised approach that can transform any decoder-only LLMs into a strong text encoder. This is done without any expensive synthetic data in a parameter efficient way, and it works in 3 steps.

## Steps

![](/images/llm_2_vec_steps.png)

1. Enable bidirectional attention
2. Masked next token prediction
3. Unsupervised [[Contrastive Training|Contrastive Learning]]

### Enable Bidirectional Attention

At the beginning of this article we estabilished why causal attention mask is not the best option if we need a rich embedding, because of this the first step in LLM2Vec is replace the attention mask with a bidirectional (all ones) mask. If we use [FlashAttention](https://github.com/Dao-AILab/flash-attention) the following will make the trick:


```python
flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
```

Laten int the benchmarks, we are going to see that by changing from causal to bidirectional attention, wont really help, actually it will worsen the models performance. The reason is that the model does not really know what to do with future information. 

### Masked Next Token Prediction (MNTP)

As mentioned above, we need to teach the model to leverage information from future tokens. To do that we train it on [Masked Next Token Prediction](https://www.semanticscholar.org/paper/An-Analysis-and-Mitigation-of-the-Reversal-Curse-Lv-Zhang/175ccc099f2a91005f53d752c09a74b8b91fdc38) objective. MNTP is similar to Masked Language Modeling, with one key distinction, that to predict the masked token at position $i$, we take the representation at the previous token $i-1$ and not the masked position itself.

### Unsupervised Contrastive Learning
In the previous step we healed the model, it is no longer confused what these new tokens are about, and we have a good word level model. But since LLMs were not trained to capture the semantics of a whole sentence, we fill this gap by applying Constrastive learning. The key idea behind constrastive learning is that we want to pull the semantic representations of similar samples, sentence closer together while pusing dissimilar father apart. Here we follow the recipe introduce by [SimCE](https://arxiv.org/abs/2104.08821), and we need positive and negative samples. **Positive samples** are generated from the sentence itself, by applying an random dopout mask with dropout probability 0.3 and we generate exactly two samples. Dropout probability 0.3 is rather high, usually researchers leverage 0.1 when training bidirectional models with contrastive objective. **Negative samples** are simply other examples in a given mini-batch.

## Training
We have our 3 steps, from this Masked Netxt Token Prediction and Unsupervised Contrastive learning introduce objectives that require training. As a dataset it is chosen English Wikipedia, mainly because the underlying model should be already trained on it. For booth objectives we leverage [LoRA](https://huggingface.co/docs/peft/main/conceptual_guides/lora), which we merge back into the model. As for compute it requires a single 80GB A100 GPU for around 5 hours, which makes LLM2Vec compeling for individuals and academic institutions.

## Results

The researchers applied LLM2Vec to Mistral-7B, S-LLama-1.3B and Llama-2-7B and here are the results:

### Word Level
![](/images/llm_2_vec_word_level_tasks.png)

We have two baselines, the first is deBERTa-v3-large (dashed lines, and it is a 418M parameter model, however 131M are the initial embeddings) and the embedding produced by a plain causal model (full line).

In general swpaing causal attention for bidirectional hurts the performance, but not in case of Mistral. Here it is theoretised that it may have been trained in some of its pretrainig phases with bidirectional attention.

If we include MNTP we observed increased performance since the model was taught to leverage the bidirectional information. Contrastive learning actually hurts the performance a bit, this makes sense since its contribution is to get a better senetence model and it is not needed for a word model.

### Sentence Level
Here the baseline is a BERT model that was trained in the SimCSE paper and we also explore various pooling methods over individual token representations.

![](/images/llm_2_vec_sentence_level_tasks.png)

The results follow similar trend as in Word Level tasks, bidirectional attention without finetunning huts the performance, but not for Mistral. The best results are achieved by applying all 3 steps with weighted mean pooling.


## Further Finetuning for MTEB

To further evaluate the performance the authors use the [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) here is the [Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) from Hugging face where you can check the latest and greatest.

![](/images/llm_2_vec_mteb.png)

# NV-Embed
If you checked the lader board, well depends when you are reading this post :), you noticed that NV-embed was on the first place. Because of this we cannot really ignore it.

## Latent Attention
NV-Embed shares a lot of similar design decisions to LLM2Vec, booth swap causal fro bidirectional attention, booth finetune the resulting model, however NV-Embed does only Contrastive learning, but the biggest difference is how they pool individual token representations into a single embedding.


# Large Language Models are Overparametrized Text Encoder

Maybe you already noticed, if we derive an encoder model from an LLM, they tend to have a lot of parameters especially if we compare it to existing state of the art, they are even 20x larger. Their size makes them significantlyslower and more expensive to run.
