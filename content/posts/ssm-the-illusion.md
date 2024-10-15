+++ 
draft = true
date = 2024-10-14T10:52:09+02:00
title = "SSMs and the Illusion of State"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

# Abstract
Last time we look into the weakpoints of State Space Models ([Mamba, Mamba2]({{< relref "posts/from-mamba-to-mamba2.md" >}})
), especially whean compared with Attention Based models(LLama, GPT like), they lack in terms of In-context learning. To alleviate this we focused on [SSM-Transformer Hybrids]({{< relref "posts/ssm-transformer-hybrids-guide.md" >}}) and we introduced multiple models that do this differently. Here we look into expressivity of State Space Models, from two formal perspectives. The fist is from a perspective of Circuit complexity and later from the perspective of Formal Languages.

## TLDR
SSMs and Transformers have similar strengths and weaknesses however there is complementary synergy, giving an more theoretical reason for hybrid architectures.

# Circuit Complexity
Circuit complexity is a branch of computational complexity. We can view a circuting as an computational graph in Neural Networks, where each node is an operation with edges between nodes that are connected. In this article there are two important classes of circuit complexity, one $TC^0$ and $NC^1$

## $TC^0$
Are circuits that have bounded (constant) depth, we have no limit on the number of inputs (unbounded fan-in) and we can only use AND, OR and Threshold gates (the output of this gate is 1 if the sum of inputs exceedes a threshold otherwise it is 0). The last constraint is that the size of the circuit is polynomial in the input length. The important of this class is, that problems that lie in this class can be extremely parallelized. 

### Example
The canonical example is Integer multiplication. Following an divide and conquer style approach we can take an sequence of integer, split them into smaller subsequences, multiply the subsequence to get intermediate results (this can be done in parallel) and finally multiply the intermediate values together. If we assume that we can at most multiply 2 integers at a time we can compute the result in $\log_2 (n)$ steps.

## Nick's Class 1 ($NC^1$)
The depth of the circuit has an logarithmic bound ($O(\log n)$), we have an fixed (bounded fan-in) number of inputs, and we can only use AND and OR gates. The last constraint is that the size of the circuit is polynomial in the input length. This class represents problems that are sequential in nature and they are hard to parallelize. 

## Example
Here I introduce an couple of examples, to make things more explicitly.
1. Evaluating boolean formulas, you have an bunch of boolean values with AND and OR between them, and sometimes grouped into groups. Technically this is to a some degree parallelizable, since we can evaluate subgroups independently, but there may be dependencies that needs to be evaluated first.
2. Regular expressions, for those who do not know, regular expressions are complied down to state machines, and you need to process character by character. Again if you have unions you may process different parts in parallel but otherwise you are forced move one character at a time
3. Simulating finite automata, everything that is processed by finite statemachines, where the state depends on the input is determined by the order in which we receive the input, altering this order can result in wastly different states.

# SSMs Transformers and RNNs
Recurrent Neural Networks (RNNs) are sequential by nature and they can solve problems in $NC^1$. However this sequential nature makes them hard to parallelize, and main stream research started to favour Transformers, which on the other hand allow for extreme parallelization. Hovewever Transformers are stuck in $TC^0$ and they cannot expresesss strictly sequential problems. Mamba technically can emulate an Recurrent Neural Network, but this is only theoretical since it requires an infinite number of layers. Because of this Mamba is stuck in $TC^0$.

### Is $TC^0$ enough?
To answer this I just give an couple of problems that lies beyod $TC^0$

1. Tracking chess moves
2. Evaluate Code
3. Tracking entities in a narative
4. Evaluate Graph Connectivity
5. Solving Linear Equations 

![TC_0 vs NC^1](/images/tc_0_vs_nc_1.png)

"Hey but ChatGPT, Claude, Llama is good at generating code!". That is true, but reasoning about what the result will be given some inputs, is beyond their capabilities. They can emulate the reasoning, but they are bound to make mistakes.

#### Expressive power of SSMs

There are two decision choices that bottlenecks expresivity of an SSMs like model.

1. Linearity of the hidden state
		$$h_i = \bar{A}h_{i-1} + \bar{B} x_i$$
	The new state $h_i$ is just an linear combination of the previous hidden state and the input. We can increase the expressivity by introducing nonlinearity:
	$$h_i = \text{sign}(\bar{A}h_{i-1} + \bar{B} x_i)$$
	However this is makes it less obvious how to parallelize the computation. 

2. Input Independence of Transition Matrix A 
	A does not really depend on $x$, in Mamba A is diagonal and in Mamba2 it is just a scalar. If we introduce dependency between $x$ and $A$ we get an more expressive SSM, this already has been done in LiquidSSM paper.
By introducing any of those 2 problems, we can teach the model to track chess pieces.

## Empirical Results
Even thous SSMs nor Transformers are equipped for state tracking, they are capable to emulate this behaviour. And overalll Mamba has an edge over Transformers in the emulation. The main reason is that Mamba is at least theoretically capable of emulating RNNs where Transformers not.

# Formal Language Perspective
In formal languages our input is always a string, and we have grammars that define how we can process this string. There are 3 basic building blocks that we can use to build these grammars:

1. Finite state automata (we already know this is $NC^1$)
2. Counters
3. Stacks

An canonical example are Regular Expressions, as mentioned before processing Regular Expressions is done by an State Machine. This state machine than has counters, and can use stack to track of previously seen characters.

## Context-free, Context-sensitive
There are two types of grammars:
1. Context-free, this is where most programming languages fall into, and they are good grammars. For example if you have an programming language that has *for* keyword, this keyword has a single meaning independently of the context where it is used. You probably saw different type of for loops, but they do the same thing, that is iterating though something that is iterable.
2. Context-sensitive, this is natural language, and it is way less nice. You may have word that have different meaning depending how they are written. Lets imagine an state machine that is processing and context sensitive grammar, and it encounters a word. Now depending on previous words, this can have an vastly different meaning, and in the extreme case it may require to keep track of the whole previously seen text just to interpret the current word. (Or looking into the future). 

## Problems
To study the strengths and weaknesses of SSMs and Transformers we first introduce 3 sample problems, these sample problems emulate string processing showcasing the different weaknesses and strengths of transformers.

![FlipFlop Parity and Dyck](/static/images/flip_flop_parity_dyck.png)

### Flip-Flop
Flip-Flop problem is defined as a sequence of instructions and data. There are 3 instructions: Read, Write, Ignore. Each time we encounter an Write instruction, we store information into memory (this information is either 0 or 1), this information will be recalled by the next Read instruction.

This seems like an simple problem, but it serves as an abstraction for long-range reasoning. Recurrent Neural Networks like LSTM do a pretty good job, however Transformers are stougling. The main painpoint for Transformers to precisely attend to the last Write is that they youd require strong positional dependence in the attention weights and also Transformers do not generalize well to arbitrary lengths since they require explicit positional encoding. In contrast SSMs are able to model this problem to arbitrary lengths by employing two SSM layers.
### Parity
In Parity we are given an bit string, which is just a string that can consist only of 0 and 1, and we require that there is and even number of 1s. If we look it from the perspective of a State Machine, we can encounter the End Of String (EOS) character only if we have read an even number of 1ns. 

As before, RNNs can easily handle Parity, with Transformers this is way more tricky, theoretically it is possible but empirically hard. For SSMs is similarly to Transformers possible, for example Mamba is input dependent which is one requirement for SSMs to handle Parity, but its transition matrix A, is non-negative, which makes this problem very hard, making Parity a problem that Transformers nor Mamba find easy to model.

### Star-free Regular Languages

For simplicity, we can view Regular Languages the same thing as Regular Expressions, and by Star-Free we require them to not involve the Kleen Star *. There is a special conjecture where we can re-express any Star-Free regular language as a Flip-Flop like state tracking, where we recall information that we previously observed. 


Transformers are theoretically equipped to model Star-Free languages, but they fail to do so since it requires unique hard attention, that is hard to construct.


Here are two examples of Star-Free problems,  Unbounded conting and Hierarchical Structures. 

#### Unbounded Counting

The simples is the Dick-1 language, and it can be viewed as matching open and closed brackets. Lets define the following string "(())", it has two open and two closed brackets. This problem can be modeled using an counter which increments by one if it observers "(" and decrements by one for ")". We can formally define Dick-1 as $a^nb^n$, where in our case $a=(, b = )$, what is important about Dick-1 is that is an basic example of a context-free grammar.

An more complex is a Shuffle-Dyck-k language, which his a shuffle of multiple Dick-1, this is defined as $a^nb^nc^n$. Here we require two counters, one for tracking $a^nb^n$ and $b^nc^n$. What is important here that this is an example of an context-sensitive grammar!

As it turns out, this problem can be well modeled using SSMs, showcasing that SSMs can model context-free and context-sensitive grammar as well.

#### Hierarchical Structures
Again we are using bracking matching, lets start with an couple of examples: "([()]), (()[]), (([])[])", we can already see that there is an hierarchical structure. Formally this are bounded-depht Dyck $\text{Dick}_{K,h}$. According to Chomsky-Schutzenberger if we push the depth $h \rightarrow \infty$ we have the fundamental backbone of context-free languages.

Again this is a problem that is a breeze for RNNs. A Two layer Transformer is doing exceptionally well, but this is mainly due the positional encoding. Similarly to Transformers, two layers of SSM can model the problem well, where the first layer is responsible to encode the depth of the brackets and the second layer tracks the last open bracket for each level of depth.

#### Module Counting
Modulo counting is just counting the number of matching entries and module this count. This falls within Star-Free Languages, but SSMs fail to model this problem.


# Results
Transformers and SSMs are bounded to solve the same class of problems, however they have different streghts. Transformers showcase better selective coping behaviour, and SSMs are better at Flip-Flop like state tracking. The biggest limiting factor for SSMs is they non-negativity of hidden state evolution, which allows for better scaling but it reduces they expresivity.

