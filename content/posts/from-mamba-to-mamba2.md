+++ 
draft = false
date = 2024-08-08T09:57:32+02:00
title = "From Mamba to Mamba-2"
description = ""
slug = ""
authors = []
tags = ["NLP", "SSM"]
categories = []
externalLink = ""
series = ["Awesome State Space Models"]
+++

## Abstract
This is not my first gig where I write about State Space Models. I already mentioned them [here](o Transformers, we may never find anything better.

Before we dive into the details of Mamba-1 and Mamba-2, let me give you a brief summary:

**Mamba-1**: The idea behind Mamba is to make the updating mechanics of its hidden state input-dependent. This may sound intuitive, but previous SSM variants like H3 were Linear Time Invariant, which means that their updating mechanism did not change with time. I the remainder of the post I will use the term Mamba and Mamba-1 interchangeably.

**Mamba-2**: Mamba-2 is generally just a simplification of Mamba, with stronger constraints on the structure of the hidden space update matrix and moving some projections to the beginning of the layer. This enables the usage of common scaling strategies used in transformers, like tensor and sequence parallelism, and the ability to split input sequences across multiple GPUs. Also, the authors build a solid theoretical foundation behind SSMs and Semi-Separable Matrices and prove they have a primal-dual relationship with Linear Attention.

## Mamba
### Structured State Space Models (S4)

Structured State Space model is defined as a one-dimensional function of sequences {% katex inline %} x(t) \in R \rightarrow y(t) \in R {% endkatex %} mapped trough a hidden state {% katex inline %} h(t) \in R^N {% endkatex %}.

The actual model consist of four parameters {% katex inline %} \Delta, A, B, C {% endkatex %} and we can express it as:

{% katex %} h(t) = Ah(t) + Bx(t) {% endkatex %}
{% katex %} y(t) = Ch(t) {% endkatex %}

- {% katex inline %} A \in R^{N \times N} {% endkatex %} is contained to be diagonal
- {% katex inline %} B \in R^{N \times 1} {% endkatex %}
- {% katex inline %} C \in R^{1 \times N} {% endkatex %}

Because of the constraints, we can represent all matrices with N numbers. To generalize to a multi-dimensional input, we apply the SSM independently to each channel, making the total memory requirements {% katex inline %} O(BLDN) {% endkatex %}.

Since we work with continuous time but process discrete data, we need to discretize the model:

{% katex %}  h_t = \bar{A} h_{t-1} + \bar{B}x_t  {% endkatex %}
{% katex %}  y_t = Ch_t  {% endkatex %}

- {% katex inline %} \bar{A} = f_A(\Delta, A) {% endkatex %}
- {% katex inline %} \bar{B} = f_B(\Delta, A, B) {% endkatex %}
- with {% katex inline %} f_A, f_B {% endkatex %} being discretization rules. For example we can use [Zero-Order hold](https://en.wikipedia.org/wiki/Zero-order_hold)

To actually compute this model, we use global convolution:

{% katex %}  y = x * \bar{K}  {% endkatex %}
- K is our kernel that is implicitly parametrized by an SSM

{% katex %}  \bar{K} = (C\bar{B}, C\bar{AB}, \cdots, C\bar{A}^k\bar{C}, \cdots) {% endkatex %}

The benefit of this is that we can use Fast Fourier Transform to compute the convolution in {% katex inline %} O(N \log N){% endkatex %} time.

#### Linear Time Invariance (LTI)

Just from the definition above, we can see that the {% katex inline %} (\Delta, A, B, C){% endkatex %} do not depend on {% katex inline %} x{% endkatex %} nor {% katex inline %} t{% endkatex %}. This is one of the main drawbacks and the reason why State Space Models were struggling with in-context learning.

### Selective State Space Models (S6)

![S6](https://n1o.github.io/images/selective_state_space_models.png)

One easy fix is to take the same model as above but make the parameters {% katex inline %} \Delta, A, B{% endkatex %} functions of the input:

#### Algorithm

- we have input {% katex inline %} x: (B,L,D){% endkatex %} (Batch, Length, Dimension)
- output {% katex inline %} y: (B, L, D{% endkatex %})

1. {% katex inline %} A: (D,N) \leftarrow \text{Parameters} {% endkatex %}
2. {% katex inline %} B: (B,L,D) \leftarrow s_B(x) {% endkatex %}
3. {% katex inline %} C: (B,L,D) \leftarrow s_C(x) {% endkatex %}
4. {% katex inline %} \Delta: (B,L,N) \leftarrow \tau_{\Delta}(\text{Parameter} + s_{\Delta}(x)) {% endkatex %}
5. {% katex inline %} \bar{A}, \bar{B}: (B, L, D, N) \leftarrow \text{discretize}(A,B)  {% endkatex %}
6: {% katex inline %} y \leftarrow \text{SSM}(\bar{A}, \bar{B}, C)(x) {% endkatex %}

- {% katex inline %} A is still diagonal {% endkatex %}
- {% katex inline %} s_B(x) = \text{Linear}_N(x) {% endkatex %}
- {% katex inline %} s_C(x) = \text{Linear}_N(x) {% endkatex %}
- {% katex inline %} s_{\Delta} = \text{Broadcast}_D(\text{Linear}_1(x)){% endkatex %} (we choose this due to a connection to Recurrent Neural Networks) 
- {% katex inline %} \tau_{\Delta} = \text{softplus} {% endkatex %} (we choose this due to a connection to Recurrent Neural Networks) 
- {% katex inline %} \text{Linear}_d {% endkatex %} is parametrized projection to dimension d 

#### Selective Scan

Since the dynamics of the model are dynamic, we cannot use global convolution anymore. Because of this, we define selective scan, which is a hardware-aware algorithm. The actual implementation is rather [involved](https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/csrc/selective_scan/selective_scan_fwd_kernel.cuh). The main idea is that we load the parameters {% katex inline %} \Delta, A, B, C {% endkatex %} from HBM to SRAM, perform the discretization and recurrence in SRAM, and write the final output of size (B, L, D) back to main memory (HBM). To reduce memory requirements, the intermediate steps are not stored but recomputed during the backward pass.

#### Benefits of (Natural) Selection

Because of the selection mechanism, the model can choose what to store (or not) in its hidden state based on what it currently sees. It may also choose to reset its hidden state and start over. Selection enables the model to have strong in-context learning capabilities.

### Mamba Layer

The core of the Mamba architecture is the Mamba layer:

![Mamba](https://n1o.github.io/images/mamba_layer.png)

We are already familiar what is happening inside the SSM (Selective Scan) part of the Mamba. Prior to it we have two projections that expand the dimensionality of the input, than we perform short convolution as in M2 Bert with [torch.nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) on one branch on the other branch we apply just SiLu non-linearity (This is the same as the Gated approach found in other LLMs). After that we perform an additional projection, and we have all the inputs prepared for the SSM block. The output of the SSM block is than multiplied with the residual gate branch and finally we project the dimension back to match the input dimension.


## Mamba-2

Mamba is a cool innovation, and it has led to multiple cool models, especially attention-SSM hybrid models like [Samba](https://github.com/microsoft/Samba) and [Zamba](https://github.com/Zyphra/transformers_zamba). However, the authors recognize some of its shortcomings. Its biggest weak point compared to Transformers is the lack of research in terms of scaling. For Transformers, we have multiple system optimizations on how to split up a model or how to split up processing long sequences into more GPUs. Here is two of them:

1. **Tensor Parallelism**: This allows splitting each layer in a large Transformer model onto multiple GPUs on the same node.
2. **Sequence Parallelism**: This allows splitting a sequence into smaller parts, with each GPU holding one part.

Mamba-2 is designed in a way that allows for Sequence Parallelism by passing the recurrent state between multiple GPUs. Tensor Parallelism is possible because of independent parallel projections of A, B, C, and X inputs of its SSM part.

### Semi-Separable Matrices

This is a special structured matrix. We say that a lower triangular matrix {% katex inline %} M {% endkatex %} is N-semi separable if every submatrix contained in the lower triangular part has rank at most N.

Here, we are more interested in a special representation of N-semi separable called Sequentially Semi Separable (SSS).


#### Sequentially Semi Separable (N-SSS)

A lower triangular matrix {% katex inline %} M \in R^{(T,T)} {% endkatex %} has an N-sequentially semiseparable representation if we can write it as:

{% katex %}  M_{ij} = C_j^TA_j \cdots A_{i+1}B_i {% endkatex %}

- {% katex inline %} B_0, \cdots, B_{T - 1}, C_0, \cdots, C_{T-1} \in R^N {% endkatex %} are vectors
- {% katex inline %} A_0, \cdots, A_{T-1} \in R^{(N,N)} {% endkatex %} 

To express it in matrix form we define the SSS operator:

{% katex %}  M = SSS(A_{0:T}, B_{0:T}, C_{0:T}) {% endkatex %}

It turns out that every N-semiseparable matrix M is also an N-sequentially semiseparable matrix. The main const of N-SSS representation that we can compress down the parameters to {% katex inline %} O(NT) {% endkatex %}

### State Space Duality

Let's start by exploring a special case of 1-semiseparable (1-SS or just 1SS). This can be written in the Sequentially Semi-Separable form as:

{% katex %} SSS(a,b,c) = \text{diag}(c) \cdot M \cdot \text{diag}(b)  {% endkatex %}
- {% katex inline %} M_{ij} = \prod_{t=j}^i a_t = a_{j:i}^{\times} {% endkatex %}

M is an 1-SS

{% katex %} M = 1SS(a_{0:T}) = \begin{bmatrix} 1 \\\  a_1 && 1 \\\ a_{2}a_1 && a_2 && 1 \\\ \vdots && \vdots && \ddots && \ddots \\\ a_{T-1}\cdots a_1 && a_{T-1}a_2 && \cdots && a_{T-1} && 1 \end{bmatrix} {% endkatex %}

#### State Space Models are Separable Matrices

We make a special assumption that we have a State Space Model without projections (no B, C) and the state dimension {% katex inline %} N = 1 {% endkatex %}. Then we can express the multiplication {% katex inline %} y = Mx {% endkatex %} as a recurrence:

{% katex %} y_t = a_{t:0}x_0 + \cdots + a_{t:t}x_t  {% endkatex %}
{% katex %} y_t = a_t(a_{t-1:0}x_0 \cdots a_{t-1:t-1}x_{t-1} + a_{t:t}x_t  {% endkatex %}
{% katex %} y_t = a_t y_{t-1} + x_t {% endkatex %}

We can generalize this further by expressing any State Space Model as matrix multiplication by an N-semiseparable matrix in a sequentially semiseparable form:

{% katex %} y = SSM(A,B,C)(x) = SSS(A,B,C) \cdot x  {% endkatex %}


### Linear(Recurrent) and Dual(Quadratic) form

We already know we can express a State Space model as a matrix multiplication by an N-separable matrix in a sequentially semiseparable form:

{% katex %}  y = SSS(A,B,C) \cdot x  {% endkatex %}

However, if we naively first compute the {% katex inline %} SSS {% endkatex %} part and then multiply by {% katex inline %} x {% endkatex %}, we end up with an {% katex inline %} O(T^2) {% endkatex %} complexity. There is a more efficient recurrent way. However, let's break down the quadratic form first, since it has a tight connection to Attention.


#### Dual (Quadratic) Form

Here, we take a small detour from SSMs and look into Linear Attention. We can express the attention mechanism as:

{% katex %} Y = \text{softmax}(QK^T) V  {% endkatex %}

This is the most common form of attention, called Softmax Attention. By applying a causal mask, we get the following:

{% katex %} Y = (L \circ \text{softmax}(QK^T)) \cdot V  {% endkatex %}

- {% katex inline %} L {% endkatex %} is an lower triangular matrix with ones on and below the main diagonal 

In linear attention we drop the softmax to get:


{% katex %} Y = (L \circ (QK^T)) \cdot V  {% endkatex %}

This form is way nicer and we can rewrite it using einsum as:


{% katex %} Y  = \text{einsum}(TN,SN,SP, TS \rightarrow TP)(Q,K,V,L) {% endkatex %}

Or we can express it as pairwise matrix multiplication:

1. {% katex inline %} G = \text{einsum}(TN,SN \rightarrow TS)(Q,K) {% endkatex %} resulting shape (T,S)
2. {% katex inline %} M = \text{einsum}(TS,TS \rightarrow TS)(G,L) {% endkatex %} resulting shape (T,S)
3. {% katex inline %} Y = \text{einsum}(TS,SP \rightarrow TP)(M,V) {% endkatex %} resulting shape (T,P)
- T, S are the target source dimensions, for autoregressive self-attention they are the same
- P is the head dimensionality

#### Linear (Recurrent) Form 

Until now, we have just removed the softmax operation. However, we can go further by changing the order of matrix association, resulting in the following:

{% katex %} (QK^T)V = Q(K^TV)  {% endkatex %}

With this, we can re-express the definition of {% katex inline %} Y {% endkatex %} as:

{% katex %}  Y  = Q \cdot \text{cumsum}(K^TV) {% endkatex %}

- cumsum is just the cumulative sum

It may seem that we got rid of the causal mask. This is technically not true, since the cumsum operation is a causal operation, and we just hid it. To make this clearer, we can express the same equation using einsum:

1. {% katex inline %} Z = \text{einsum}(SP,SN \rightarrow SPN)(V,K) {% endkatex %} resulting shape (S,P,N)
2. {% katex inline %} H = \text{einsum}(TS,SPN \rightarrow TPN)(V,K) {% endkatex %} resulting shape (T,P,N) this being optimized with subquadratic matrix multiplication
3. {% katex inline %} Y = \text{einsum}(TN,TPN \rightarrow TP)(V,K) {% endkatex %} resulting shape (T,P)

Lets break down the equation:

1. Expands the dimensionality by a factor N
2. Uses the mask matrix L explicitly, we flatten the dimensions of (P,N) resulting in multiplying an lower triangular matrix with an vector. This just just an cumulative sum operation:

{% katex %}  y = \begin{bmatrix} 1 \\\ \cdots && \ddots \\\ 1 && \cdots && 1 \end{bmatrix}x \Leftrightarrow  \begin{matrix} y_0 = x_0 \\\ y_t = y_{t-1} + x_t\end{matrix}  {% endkatex %}

3. Contracts the dimensionality back to P

### State Space Models and Recurrent Linear Attention

The hints that there should be a connection between the recurrent form of Linear Attention and the State Space Model should be obvious.

Lets remind us about the definition of the State Space Model using SSS:


{% katex %}  Y = SSS(A,B,C) \cdot x  {% endkatex %}

The SSS matrix M is defined as:

- {% katex inline %} M_{ji} = C_j^TA_{j:i}B_i{% endkatex %}

By constraining the A matrix to be diagonal {% katex inline %} A = aI {% endkatex %} we can rearrange the terms a bit to get:

{% katex %}  M_{ji} = A_{j:i} \cdot (C_j^TB_i) {% endkatex %}

The equation for M in matrix form becomes:

{% katex %} L = 1SS(a) {% endkatex %}
{% katex %} M = L \circ (CB^T) {% endkatex %}

- {% katex inline %} B,C \in R^{(T,N) {% endkatex %}}

Now we can compute {% katex inline %} Y = MX {% endkatex %} using einsum as:

1. {% katex inline %} G = \text{einsum}(TN,SN \rightarrow TS)(C,B) {% endkatex %} resulting shape (T,S)
2. {% katex inline %} M = \text{einsum}(TS,TS \rightarrow TS)(G,L) {% endkatex %} resulting shape (T,S)
3. {% katex inline %} Y = \text{einsum}(TS,SP \rightarrow TP)(M,X) {% endkatex %} resulting shape (T,P)

If we assume that S = T, we end up with the same equations as in the Recurrent form of Linear Attention. And that is it, we have our duality.

### Mamba-2 Layer

At the beginning, I mentioned that there are few differences between Mamba and Mamba-2. One of them is a stronger constraint on the matrix A, for Mamba-2 it is {% katex inline %} A = aI {% endkatex %} in Mamba it was {% katex inline %} A = \text{diag}(a) {% endkatex %}. The reason to constrain to {% katex inline %} A = aI {% endkatex %} is that we can express the SSM as a matrix multiplication of an 1-SS matrix, which is more efficient to compute.

![S6](https://n1o.github.io/images/mamba_2_architecture.png)

In the image above, we can see the differences between Mamba and Mamba-2. While the idea of Mamba was to have a function {% katex inline %} X \rightarrow Y {% endkatex %}, in Mamba-2, we instead think of a mapping of {% katex inline %} A, B, C, X \rightarrow Y {% endkatex %}. Because of this, we can parallelize the computation of the projections at the beginning of the block. This enables tensor parallelism and reduces the number of parameters. This is also analogous to Attention, where {% katex inline %} X, B, C {% endkatex %} correspond to {% katex inline %} Q, K, V {% endkatex %}.

Additionally, Mamba-2 introduces a larger head dimension {% katex inline %} P {% endkatex %}. While Mamba leverages {% katex inline %} P =1  {% endkatex %}, Mamba-2 leverages {% katex inline %} P = \{64, 128\} {% endkatex %}. Again, this is similar to conventions in Transformer Architecture. What does this head dimension in Mamba mean? If we have a head dimension of 1, we are computing an SSM for each channel independently. By increasing the head dimension, we achieve a sort of weight-tying where we share SSMs across multiple channels.


Overall, it may seem that Mamba-2 is less expressive than Mamba. However, due to optimizations, we are able to train Mamba-2 models with much larger state dimensions (in Mamba-1 we had {% katex inline %} N=16 {% endkatex %}, whereas in Mamba-2 we can go up to {% katex inline %} N=256 {% endkatex %} or more), while also being much faster during training.

The model also adds an additional normalization layer, which improves the stability of larger models. There is nothing more to say about Mamba-2; it is simply a more efficient version of Mamba, incorporating many lessons learned from Transformers and the strong theoretical foundation behind SSMs and Semi-Separable Matrices.

#### Algorithm

As with Mamba-1, we cannot use Global Convolution. For Mamba-2, we need an efficient way to compute the matrix {% katex inline %} M {% endkatex %}. Luckily, the computation is much simpler than for Mamba-1, and we do not need to implement a low-level GPU kernel. The algorithm consists mostly of matrix multiplications.

![Mamba-2 Blocks](https://n1o.github.io/images/mamba_2_diagonal_off_diagonal_blocks.png)

This is an example for {% katex inline %} T=9 {% endkatex %} where we decompose it into chunks of length {% katex inline %} Q = 3 {% endkatex %},  we can generalize it as:


1. {% katex inline %} M^{(j,j)} = SSM(A_{jQ:(j+1)Q},B_{jQ(j+1)Q},C_{jQ:(j+1)Q} {% endkatex %} for the diagonal blocks
2. {% katex inline %} M^{(i,j)} = \begin{bmatrix}C_{jQ}^TA_{jQ:jQ-1} \\ \vdots \\ C^T_{(j+1)Q-1}A_{(j+1)Q-1:jQ-1}\end{bmatrix}A_{jQ-1:(i+1)Q-1} \begin{bmatrix}B_{iQ}^TA_{(i+1)Q-1:iQ} \\ \vdots \\ B_{(i+1)Q-1}^T A_{(i+1)Q-1:(i+1)Q-1}\end{bmatrix}^T {% endkatex %} for the off-diagonal low rank blocks

##### Diagonal Blocks

The general idea is that {% katex inline %} Q {% endkatex %} is rather small. Because of this, we can use the dual quadratic form of Structured Masked Attention (more on this later) and perform the computation for each block in parallel.

##### Low Rank Blocks

Here, we have three parts (the following example is the breakdown of the leftmost bottom block from the image above):

1. {% katex inline %} \begin{bmatrix} C_6^T A_{6:5} \\\ C_7^TA_{7:5} \\\ C_8^TA_{8:5} \end{bmatrix}^T {% endkatex %} this are the left factors (C-block factors)
2. {% katex inline %} A_{5:2} {% endkatex %} this are the center factors (A-block factors)
3. {% katex inline %} \begin{bmatrix} B_0^T A_{2:0} \\\ B_1^TA_{2:1} \\\ B_2^TA_{2:2} \end{bmatrix}^T {% endkatex %} this are the right factors (B-block factors)

##### Pytorch

Compared to Mamba-1's selective scan the implementation is way more straight forward:

```python
def segsum(x):
"""Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
which is equivalent to a scalar SSM."""
T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)

return x_segsum

def ssd(X, A, B, C, block_len=64, initial_states=None):
    """
    Arguments:
    X: (batch, length, n_heads, d_head)
    A: (batch, length, n_heads)
    B: (batch, length, n_heads, d_state)
     C: (batch, length, n_heads, d_state)
    Return:
    Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0
    ## Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)
    ## 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    ## 2. Compute the state for each intra-chunk
    ## (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    ## 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    ## (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    ## 4. Compute state -> output conversion per chunk
    ## (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    ## Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
```

##### Performance

Mamba-2, like Mamba-1, with hidden state size \( N \), has the same training speed \( O(TN^2) \) and inference speed \( O(N^2) \). However, the biggest improvement is the use of matrix multiplication in Mamba-2, which is much more efficient than the selective scan in Mamba-1.

## State Space Duality Additional Notes

Overall, the State Space Duality paper introduces many concepts; here are arguably the most important ones:

### Structured Masked Attention

This builds upon the notion of linear attention, where we expressed the causal mask matrix L as a cumulative sum. However, we can generalize the mask matrix L to any matrix that supports fast matrix multiplication.

![[https://n1o.github.io/images/structured_attention.png]]

In this case, we view the attention mechanism through the following equations (this is also the quadratic form mentioned earlier):

{% katex %} Y = MV {% endkatex %}

{% katex %} M = QK^T \circ L {% endkatex %}

Where L is our mask matrix, which we can choose as we like. In the context of State Space duality, we choose it as 1-semiseparable matrix. 

### Multi patterns for SSMs

Again, this builds upon analogies to Attention, where multihead attention involves applying self-attention multiple times and concatenating the results. We can achieve something similar by applying the SSD algorithm and broadcasting it across multiple dimensions.

#### Multi-Contract SSM
This is analogous to Multi-Query Attention, where we share K and V across all the heads of Q. For attention, this makes a lot of sense since we cache K and V pairs.

In SSMs, this is equivalent to sharing X and B across multiple heads of the SSM, and having C (parameters that control the contraction) be independent per head.


#### Multi-Expand SSM

Here, we share C and X across multiple heads, and B (controls expansion) is independent per head.

#### Multi-Input SSM

Here, we share B and C across multiple heads, and X is independent. For an SSM like Mamba, we consider X as the input. Because of this, it is a better fit to have a unique X per head.

Technically, we can view the S6 layer introduced in Mamba as having Head Dimension P = 1, which means that each channel has independent SSM dynamics A, and it is a Multi-Input SSM where we share B and C matrices across all channels.

## TLDR;

This was probably a lot to take in, so to sum it up, we introduced Mamba. Mamba is a State Space model whose dynamics are dependent on the input, which improves its ability for In-Context learning. Because we want efficient computation, we need to derive a hardware-efficient algorithm, and to do that, we need to enforce structure on the matrices used by Mamba. Mamba-2 tackles the efficiency problem by enforcing even more constraints, making the model more contained but easier to scale and allowing for larger hidden dimensions.
