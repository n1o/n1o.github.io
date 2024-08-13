+++ 
draft = true
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

# Abstract
This is not my first gig where I write about State Space Models. I already mentioned them [here]({{< relref "posts/hungry-hungry-hippos.md" >}}) and [here]({{< relref "posts/butterflies-monarchs-hyenas-and-lightning-fast-bert.md" >}}). Now what is the deal with this Mamba(2) thing? They are proving to be an alternative to the strong Transformer++ architecture. (Transformer++ are models like LLaMa based on Rotary Embedding, SwiGLU, MLP, RMSNorm, without linear bias, sometimes with grouped query attention and/or sliding window attention. Hold on, if this Transformer++ work well why do we need altneratives? There are multiple reason: 

1. **Performance**, self attention with an causal mask has an quadratic bottleneck, and as the sequence length is become longer this becomes an problem. Resolving this issue is an field of active research, one possible solution is to use Linear Attention, which we will cover since it is some of the basics Mamba-2 builds upon. Another possibility is to use Sliding Window Attention, which constraints the context for the next token generation up to the past last N tokens, where N is the window size. This alleviates the memory requirements, it we make the model less capable. Technically speaking Sate Space Models scale linearly in terms of the sequence length, (quadratically with the state size, but in general this is fixed).
2. **State**, attention is stateless, there is no hidden state that is sequentially updated. This is a good and a bad thing at the same time. It is good since if the model requires to look something up it will take into account everythingit hase seen before. This is super important, since it enables in-context learning. It is bad since it has to keep track of everything that is has seen before. With state space models we have an hidden state that it is updated every time we have an new input. Because of this we can view the hidden state as an compressed representation of of everything it has observed before. Again this is a good and bad thing. It is good since this compressed representation is smaller than the whole sequence, making it more efficient. It is bad since the hidden state has to be large enough to store verything that is important and at the same time it has to remain relativelly small to be efficient, AND (it is capilat for a reason!) the mechanism that updates the sate has to do it in an meaningful way (this is something we are going to explore in more detail). 
3. **Alternatives**, I get it this is subjective, but if we are going only do research into Transformers we may never find anything better.

Before we dive into the details of Mamba and Mamba-2 let me give you the brief summary:

1. **Mamba**
The idea behind Mamba is to make the updating mechanics of its hidden state input dependent. This may sound intuitive, but with previous SSM variants like H3, were Linear Time Invariant, which means that their updating mechanism did not change with time.

2. **Mamba-2**
Mamba-2 is in general just a simplification of Mamba, stronger constraint on the structure of the hidden space update matrix and moving some projections at the beginning of the layer, which enables the usage of common scaling strategies used in transformers, like tensor and sequence parallelism and being able to split input sequences into multiple GPUs. Also the authors build up an solid teoretical foundation behind SSMs and Semi-Separable Matrices and they also prove they have a primal-dual relationship with Linear Attention.

# Mamba
## Structured State Space Models (S4)

Structured State Space model is defined as a 1-dimensional function of sequences $x(t) \in R \rightarrow y(t) \in R$ mapped trough a hidden state $h(t) \in R^N$.

The actual model consist of four parameters $\Delta, A, B, C$ and we can express it as:

$$h(t) = Ah(t) + Bx(t) $$
$$y(t) = Ch(t)$$

- $A \in R^{N \times N}$ is contained to be diagonal
- $B \in R^{N \times 1}$
- $C \in R^{1 \times N}$

Because of the constrained we can represent all matrices with N numbers, and to generalize to a multi-dimensional input we apply the SSM independently to each channel. Making the total memory requirements $O(BLDN)$.

Since we work with continuous time we need to discretize the model:

$$ h_t = \bar{A} h_{t-1} + \bar{B}x_t $$
$$ y_t = Ch_t $$

- $\bar{A} = f_A(\Delta, A)$
- $\bar{B} = f_B(\Delta, A, B)$
- with $f_A, f_B$ geing discretization rules for example we can use [Zero-Order hold](https://en.wikipedia.org/wiki/Zero-order_hold)

To actually compute this model we use global convolution:

$$ y = x * \bar{K} $$
- K is our kernel that is implicitly parametrized by an SSM

$$ \bar{K} = (C\bar{B}, C\bar{AB}, \cdots, C\bar{A}^k\bar{C}, \cdots)$$

The benefit of this is that we can use Fast Fourier Transform to compute the convolution in $O(N \log N)$ time.

### Linear Time Invariance (LTI)

Just from the definition above we can see that the $(\Delta, A,B, C)$ do not depend on $x$ nor $t$. This is one of the main drawbacks, and it is the reason why State Space Models were strugling with in context learning.

## Selective State Space Models (S6)

![S6](/images/selective_state_space_models.png)

We take the same model as above, but we make the parameters $\Delta, A, B$ functions of the input:

### Algorithm

- we have input $x: (B,L,D)$ (Batch, Length, Dimension)
- output $y: (B, L, D)$

1. $A: (D,N) \leftarrow \text{Parameters}$
2. $B: (B,L,D) \leftarrow s_B(x)$
3. $C: (B,L,D) \leftarrow s_C(x)$
4. $\Delta: (B,L,N) \leftarrow \tau_{\Delta}(\text{Parameter} + s_{\Delta}(x))$
5. $\bar{A}, \bar{B}: (B, L, D, N) \leftarrow \text{discretize}(A,B)$ 
6: $y \leftarrow \text{SSM}(\bar{A}, \bar{B}, C)(x)$

- $A$ is still diagonal
- $s_B(x) = \text{Linear}_N(x)$
- $s_C(x) = \text{Linear}_N(x)$
- $s_{\Delta} = \text{Broadcast}_D(\text{Linear}_1(x))$ (we choose this due to a connection to Recurrent Neural Networks)
- $\tau_{\Delta} = \text{softplus}$ (we choose this due to a connection to Recurrent Neural Networks)
- $\text{Linear}_d$ is parametrized projection to dimension d

### Selective Scan

Since the dynamics of the model is dynamical we cannot use global convolution anymore, because of this we define selective scan that is an hardware aware algorithm. The actual implementation is rather [involved](https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/csrc/selective_scan/selective_scan_fwd_kernel.cuh). The main idea is that we load the parameters $\Delta, A,B,C$ from HBM to SRAM, we perform the discretization and recurrence in SRAM and we write the final output of size (B,L,D) back to HBM. To reduce memory requirements the intermediate steps are not stored but recomputed during the backward pass.

### Bennefits of selection

This selection mechanism gives the model to choose what to store (or not) in its hidden state, based on what it currently sees. The model also may choose to reset its hidden state and start over. This gives the model the possibility of in context learning

## Mamba Layer

The core of the Mamba architecture is the Mamba layer:

![Mamba](/images/mamba_layer.png)

We are already familiar what is happening inside the SSM (Selective Scan) part of the Mamba. Prior to it we have two projections that expand the dimensionality of the input, than we perform short convolution as in M2 Bert with [torch.nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) on one branch on the other branch we apply just SiLu non-linearity (This is the same as the Gated approach found in other LLMs). After that we perform an additional projection, and we have all the inputs prepared for the SSM block. The output of the SSM block is than multiplied with the residual gate branch and finally we project the dimension back to match the input dimension.


# Mamba-2

Mamba is an cool inovation, and it lead to a multiple cool models, especially attention-ssm hybrid models like [Samba](https://github.com/microsoft/Samba) and [Zamba](https://github.com/Zyphra/transformers_zamba), however the authors recognize some of its shortcomings. Its biggest weakpoint when compared to Transformers, its lack of research in terms of scaling. For Transformers we have multiple system optimization on how to split up a model, or how to split up processing long sequences into more GPUs. These techniques include:

1. Tensor Parallelism, this allows of splitting each layer in an large Transformer model onto multiple GPUs on the same node.
2. Sequence Parallelism, this allows to split a sequence onto smaller parts each GPU holding one part.

Mamba-2 is designed in a way that it allows for Sequence Parallelism by passing the recurrent state between multiple GPUs. Tensor Parallelism is possible because of inpendent parallel projections of A,B,C,X inputs of its SSM part.

## Semi-Separable Matrices

This is a special structured Matrix, we say that an lower triangular Matrix M is N-semi separable if every submatrix contained in the lower triangular part hat rank at most N.

Here we are more interested in a special representation of N-semiseparable called Sequentially Semi Separable (SSS).

### Sequentially Semi Separable (N-SSS)

A lower triangular matrix $M \in R^{(T,T)}$ has an N-sequentially semiseparable representation if we can write it as:

$$ M_{ij} = C_j^TA_j \cdots A_{i+1}B_i$$

- $B_0, \cdots, B_{T - 1}, C_0, \cdots, C_{T-1} \in R^N$ are vectors
- $A_0, \cdots, A_{T-1} \in R^{(N,N)} $

To express it in matrix form we define the SSS operator as:

$$ M = SSS(A_{0:T}, B_{0:T}, C_{0:T})$$

It turns out that every N-semiseparable matrix M is also an N-sequentially semiseparable matrix. The main const of N-SSS representation that we can compress down the parameters to $O(NT)$

## State Space Duality

Lets start by exploring a special case of 1-semiseparable (1-SS or just 1SS), this can be written in the Sequentially Semi Separable form as:

$$SSS(a,b,c) = \text{diag}(c) \cdot M \cdot \text{diag}(b) $$
- $M_{ij} = \prod_{t=j}^i a_t = a_{j:i}^{\times}$

M is an 1-SS

$$M = 1SS(a_{0:T}) = \begin{bmatrix} 1 \\\  a_1 && 1 \\\ a_{2}a_1 && a_2 && 1 \\\ \vdots && \vdots && \ddots && \ddots \\\ a_{T-1}\cdots a_1 && a_{T-1}a_2 && \cdots && a_{T-1} && 1 \end{bmatrix}$$

### State Space Models are Separable Matrices

We make an special assumption that we have an Statpe Space Model without projections (no B,C) and the state dimension N = 1 than we can express the multiplication $ y = Mx$ as recurrence:

$$y_t = a_{t:0}x_0 + \cdots + a_{t:t}x_t $$
$$y_t = a_t(a_{t-1:0}x_0 \cdots a_{t-1:t-1}x_{t-1} + a_{t:t}x_t $$
$$y_t = a_t y_{t-1} + x_t$$

We can generalize this further by expressing any State Space Model as matrix multiplication by an N-Semi Separable matrix in an sequentially semiseparable form:

$$y = SSM(A,B,C)(x) = SSS(A,B,C) \cdot x $$


## Linear(Recurrent) and Dual(Quadratic) form

We already now we can express a State Space model as a matrix multiplication by an N-Separable matrix in an sequentially semiseparable form:

$$ y = SSS(A,B,C) \cdot x $$

However if we naively first compute the $SSS$ part and than multiply by $x$ we end up with an $O(T^2)$ complexity. There is an more efficient recurrent way, however lets break down the Quadratic form first, since it has tight connection to Attention.

### Dual (Quadratic) Form

Here we take a small detour from SSMs and look into Linear Attention. We can express the attention mechanism as:

$$Y = \text{softmax}(QK^T) V $$

This is the most common form of attention, called Softmax Attention, by applying an causal mask we get the following:

$$Y = (L \circ \text{softmax}(QK^T)) \cdot V $$

- $L$ is an lower triangular matrix with ones on the diagonal and zeros above the diagonal

In linear attention we drop the softmax and we get:


$$Y = (L \circ (QK^T)) \cdot V $$

This form is way nicer and we can rewrite it using einsum as:


$$Y  = \text{einsum}(TN,SN,SP, TS \rightarrow TP)(Q,K,V,L)$$

Or we can express it as pairwise matrix multiplication:

1. $G = \text{einsum}(TN,SN \rightarrow TS)(Q,K)$ resulting shape (T,S)
2. $M = \text{einsum}(TS,TS \rightarrow TS)(G,L)$ resulting shape (T,S)
3. $Y = \text{einsum}(TS,SP \rightarrow TP)(M,V)$ resulting shape (T,P)
- T, S are the target source dimensions, for autoregressive self-attention they are the same
- P is the head dimensionality

### Linear (Recurrent) Form 
Until now we just removed the softmax operation, but we can go further and we can change the order of matrix association, and we get the following:

$$(QK^T)V = Q(K^TV) $$

With this we can reexpress the definition of Y as:

$$ Y  = Q \cdot \text{cumsum}(K^TV)$$

- cumsum is just the cumulative sum

Here it seems that we got rid of the Causal Mask, this is thechnically not true, since the cumsum operation is an causal operation. To make this more clear we can express the same equation using einsum:

1. $Z = \text{einsum}(SP,SN \rightarrow SPN)(V,K)$ resulting shape (S,P,N)
2. $H = \text{einsum}(TS,SPN \rightarrow TPN)(V,K)$ resulting shape (T,P,N) this being optimized with subquadratic matrix multiplication
3. $Y = \text{einsum}(TN,TPN \rightarrow TP)(V,K)$ resulting shape (T,P)

Lets break down the equation:

1. Here we expand the dimensionality by a factor N
2. Here we use the mask matrix L explicitly, we flatten the dimensions of (P,N) resulting we are multiplying an lever triangular matrix with an vector. This just just an cumulative sum operation:

$$ y = \begin{bmatrix} 1 \\\ \cdots && \ddots \\\ 1 && \cdots && 1 \end{bmatrix}x \Leftrightarrow  \begin{matrix} y_0 = x_0 \\\ y_t = y_{t-1} + x_t\end{matrix} $$

3. Contracts the dimensionality back to P

## State Space Models and Recurrent Linear Attention

We can clearly see the hints of that there should be an connection between the recurrent form of Linear Attention and State Space Model. 

Lets remind us about the definition of the State Space Model using SSS:


$$ Y = SSS(A,B,C) \cdot x $$

The SSS matrix M is defined as:

- $M_{ji} = C_j^TA_{j:i}B_i$

By constraining the A matrix to be diagonal $A = aI$ we can rearrange the terms a bit to get:

$$ M_{ji} = A_{j:i} \cdot (C_j^TB_i)$$

The equation for M in matrix form becomes:

$$L = 1SS(a)$$
$$M = L \circ (CB^T)$$

- $B,C \in R^{(T,N)}$

Now we can compute $Y = MX$ using einsum as:

1. $G = \text{einsum}(TN,SN \rightarrow TS)(C,B)$ resulting shape (T,S)
2. $M = \text{einsum}(TS,TS \rightarrow TS)(G,L)$ resulting shape (T,S)
3. $Y = \text{einsum}(TS,SP \rightarrow TP)(M,X)$ resulting shape (T,P)

If we assume that S = T, we end up with the same equations as in the Recurrent form of Linear Attention. And that is it! We have our Mamba-2.

## Mamba-2 Layer

At the beginning I mentioned that there are few differences between Mamba and Mamba-2. One of them is a stronger constraint on the matrix A, for Mamba-2 it is $A = aI$ in Mamba it was $A = \text{diag}(a)$.

![S6](/images/mamba_2_architecture.png)

In the image above we can the differences between Mamba and Mamba-2. Where the idea of Mamba was of having an function $X \rightarrow Y$ in Mamba-2 we instead think of an mapping of $A,B,C,X \rightarrow Y$. Because of this we can parallelize the computation of the projections at the beginning of the block, this enables tensor parallelism and reduces the number of parameters. This is also an analogy to Attention where X,B,C correspond to Q,K,V. Also the Mamba-2 introduces an larger head dimension P, where Mamba leverages P = 1, Mamba-2 leverages P = {64, 128}, again this is similar to conventions in Transformer Architecture. What does this head dimension in Mamba mean? If we have head dimension 1, we are computing an SSM for each channel independently, by increasing the head dimension we as sort of weight-tying where we share SSMs across multiple channels. 

Overall it may seem that Mamba-2 is less expressive than Mamba, but because of the optimization we are able to train Mamba-2 models with much larger state dimension (in Mamba-1 we had N=16, in Mamba-2 we can go up to N=256 or more), and at the same time being much faster during training.

The mode also ads an additional normalization layer, which improves the stability of the model in larger models. There is nothing more to say about Mamba-2, it is just an more efficient version of Mamba, taking into account a lot of the lessons learned from Transformers and strong theoretical foundation behind SSMs and Semi-Separable Matrices.

### Algorithm

As with Mamba-1 we need an hardware aware (and efficient) algorithm to perform the computation. Form Mamba-2 we need an efficient way to compute the matrix M, but luckily the computation is way simpler than for Mamba-1 and we do not need to implement an low level GPU kernel, and the algorithm consists mostly of matrix multipications.

![Mamba-2 Blocks](/images/mamba_2_diagonal_off_diagonal_blocks.png)

This is an example for $T=9$ where we decompose it into chunks of length $Q = 3$,  we can generalize it as:


1. $M^{(j,j)} = SSM(A_{jQ:(j+1)Q},B_{jQ(j+1)Q},C_{jQ:(j+1)Q}$ for the diagonal blocks
2. $M^{(i,j)} = \begin{bmatrix}C_{jQ}^TA_{jQ:jQ-1} \\ \vdots \\ C^T_{(j+1)Q-1}A_{(j+1)Q-1:jQ-1}\end{bmatrix}A_{jQ-1:(i+1)Q-1} \begin{bmatrix}B_{iQ}^TA_{(i+1)Q-1:iQ} \\ \vdots \\ B_{(i+1)Q-1}^T A_{(i+1)Q-1:(i+1)Q-1}\end{bmatrix}^T$ for the off-diagonal blocks, these are also low rank blocks

#### Diagonal Blocks

The general idea is that Q is rather small, because of this we can use the dual quadratic form of Structured Masked Attention (More later) and do the computation in parallel. 

#### Low Rank Blocks

Here we have three parts (we are going to use the left most bottom block from the image above):

1. $\begin{bmatrix} C_6^T A_{6:5} \\\ C_7^TA_{7:5} \\\ C_8^TA_{8:5} \end{bmatrix}^T$ this are the left factors or C-block factors
2. $A_{5:2}$ this are the center factors or A-block factors
3. $\begin{bmatrix} B_0^T A_{2:0} \\\ B_1^TA_{2:1} \\\ B_2^TA_{2:2} \end{bmatrix}^T$ this are the right factors or B-block factors

#### Pytorch

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
    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)
    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
```
#### Performance

Mamba-2 as Mamba-1 with hidden state size N have the same training speed $O(TN^2)$ and inference speed $N^2$ whoever the biggest improvement is the use of matrix multiplication in Mamba-2, which is way more efficient than the selective scan in Mamba-1.

# State Space Duality Additional Notes

Overall the State Space Duality paper introduces many more concepts, here are arguably the most important:

## Structured Masked Attention

This builds up on the notion of linear attention, where we expressed the causal mask matrix L as an cumulative sum. However can generalize the mask matrix L to any matrix that has fast matrix multiplication.

![[/images/structured_attention.png]]

In this case we view the attention mechanism as the following equations (this is also the quadratic form I mentioned earlier):

$$Y = MV$$

$$M = QK^T \circ L$$

Where L is our mask matrix, that we can choose as we like, in the context of State Space duality we choose it as 
1-semiseparable matrix. 

## Multi patterns for SSMs

Again this builds up analogies to Attention, with multihead attention is just applying self attention multiple times and concatenating the results. We can achieve something similar by applying the SSD algorithm but broadcasting it across multiple dimensions.

### Multi-Contract SSM

This is analogous to Multi Query Attention, where we share K,V across all the heads of Q. For attention this makes a lot of sense since we cache K,V pairs.

In SSMs this is equivalent of sharing X and B across multiple heads of the SSM, and having C (parameters that controls the contration) being independent per head.

### Multi-Expand SSM

Here we share C and X across multiple heads, and B (controls expansion) is inependent per head.

### Multi-Input SSM

Here we share B and C across multiple heads, and X is independent. For an SSM like Mamba, we consider X as the input, because of this it is an better fit to have unique X per head.

Technically we can view S6 layer introduced in Mamba as heaving Head Dimension P = 1, which means that each channel as an independent SSM dynamics A, and it is an Multi Input SSM where we share B and C matrices across all channels. 

# TLDR;

This was probably a lot to take in, to summ it up we introduced Mamba. Mamba is an State Space model which dynamics is dependent of the input, this improves it ability for In Context learning. Because we want efficient computation, we need to derive an hardware efficient algorithm and to do that we need to enforce structure on the matrices used by Mamba. Mamba-2 tackles the efficiency problem, by enforcing even more constraints making the model more contained but easier to scale and allowing for larger hidden dimensions. 
