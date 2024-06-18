+++ 
draft = true
date = 2024-06-12T14:07:17+02:00
title = "BinT5 and HexT5 or how is T5 used for Binary Reverse Engineering"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

# Abstract

For a while now I have a new passion and that is binary reverse engineering and vulnerability exploitation. This interest has led me to create [CodeBreakers](https://codebreakers.re) 
a platform dedicated to applying machine learning to reverse engineering, vulnerability detection, exploitation, and other cybersecurity-related applications.

There are two approaches where [T5]({{< relref "posts/t5-the-old-new-thing.md" >}}) was applied for reverse engineeringg, one is [BinT5](https://arxiv.org/abs/2301.01701) and the other [HexT5](https://www.semanticscholar.org/paper/HexT5%3A-Unified-Pre-Training-for-Stripped-Binary-Xiong-Chen/04c3fccfe01f42afe18dcdb027385f350ab3c9d1) but before we dive into details about these papers let's first take a look at the basics of reverse engineering.


## Reverse Engineering

To understand reverse engineering we need to start with compilation. Compilation is the process of translating a high-level programming language like C (Yes! C is a high level programming language) to machine code. Machine code is something that your CPU can directly execute. The actual execution of the machine code is bit more involved, since an instruction like "add" can be further broken down into micro-code instructions, instruction can actually be executed out of order and with speculative execution things get messy really fast, and a there is a lot of room for mistakes and security vulnerabilities.

But to keep thing simple lets continue with machine code. Machine code is just a sequence of bytes. These bytes can be instruction or data. (Code is data, data is code is one of the basic idea of Von Neumann's architecture). These bytes can be without loose of generality reconstructed back to assembly code. Assembly code is a human readable representation of machine code. One thing to note is that being human readable does not mean that it is easy to understand.

To give you an example of the following ```simple_main.c``` code:

```c
#include <stdio.h>


int main(int argc, char const *argv[])
{
    if (argc > 1) {
        printf("More than one argument");
    }
    printf("Hello, World!\n");
    return 0;
}
```

Compiled with GCC 11.4.0

```
gcc simple_main.c -o simple_main.so -no-pie -fno-stack-protector
```
**Please never ever enable those 2 flags above in production code, i just want to keep the assembly as simple as possible.**

We get [this](https://raw.githubusercontent.com/n1o/n1o.github.io/master/examples/decompiled_simple_main.txt), it aint pretty, and there is probably a lot more than we asked for. But this is still manageable, and in reasonable time I can show the assembly output for our main function from above:

```assembly
0000000000401156 <main>:
  401156:	f3 0f 1e fa          	endbr64 
  40115a:	55                   	push   rbp
  40115b:	48 89 e5             	mov    rbp,rsp
  40115e:	48 83 ec 10          	sub    rsp,0x10
  401162:	89 7d fc             	mov    DWORD PTR [rbp-0x4],edi
  401165:	48 89 75 f0          	mov    QWORD PTR [rbp-0x10],rsi
  401169:	83 7d fc 01          	cmp    DWORD PTR [rbp-0x4],0x1
  40116d:	7e 14                	jle    401183 <main+0x2d>
  40116f:	48 8d 05 8e 0e 00 00 	lea    rax,[rip+0xe8e]        # 402004 <_IO_stdin_used+0x4>
  401176:	48 89 c7             	mov    rdi,rax
  401179:	b8 00 00 00 00       	mov    eax,0x0
  40117e:	e8 dd fe ff ff       	call   401060 <printf@plt>
  401183:	48 8d 05 91 0e 00 00 	lea    rax,[rip+0xe91]        # 40201b <_IO_stdin_used+0x1b>
  40118a:	48 89 c7             	mov    rdi,rax
  40118d:	e8 be fe ff ff       	call   401050 <puts@plt>
  401192:	b8 00 00 00 00       	mov    eax,0x0
  401197:	c9                   	leave  
  401198:	c3                   	ret    
```
The first column is the address in memory where the instruction is stored, the second column are the opcodes, and the third column is the human readable representation of the opcode, with some comments. In the actually code we see some function calls, some memory manipulation and conditional jumps. Somebody who has some idea about the calling convensions and has basic understanding of assembly can, in an reasonable time, understand what is going on. Lets push this a bit further and lets invoke the ```strip``` command to get [this beauty](https://raw.githubusercontent.com/n1o/n1o.github.io/master/examples/decompiled_striped_simple_main.txt). This makes things exponentially harder, there is no main function at all. Luckily for us, we have access to the unstriped version (and we added ```-no-pie``` flag to GCC) and our main function is still at ```0x401156```. But you can already see the pattern, that once we strip the binary we loose a lot of information, like function simbols and their boundaries. Most comercial software are stripped, and you can bet that most mallware is stripped as well.

I hope I convinced you that reverse engineering challenging as is, by stripping binaries makes it even harder. By introducing various source code obfuscations we are standing before a nearly impossible task. 

# BinT5

We build on top of CodeT5 Base (220m) and finetune it to perform reverse engineering, more specifically we want to generate natural language summaries for binary functions. Why summaries? We have to remind ourselves that compilation is not a 1 to 1 function, making the recovery of actually source code an imossible task. But providing an highlevel overview of the behaviour of the function gives the reverse engineer a good starting point, where to focus on and what to expect. 

## Input Data
The authors state that there is no reference dataset for reverse engineering task, so they create their own dataset. They call it Capybara, and it contains 214k decompiled C binaries for projects built for Ubuntu using Travis CIcopiled with GCC using the following opimization flags: -O0, -O1, -O2, -O3. These binaries are than striped. One of the biggest downsides of C projects is a lack of standard single documentation format, because of this the authors apply various heuristics to provide documnentation for the functions.

The actually input of the model is not the assembly code, but the decompiled pseudo C code. For generating pseudo C code the authors use [Ghidra](https://www.ghidra-sre.org/). Why using pseudo C instead of assembly, because pseudo C is more human readable and it was performing better in the experiments. And since CodeT5 was pretrained on C source code, and there is a lot of similarity between C source code and pseudo C code (it is pseudo C code because it does not have to be compilable). 

### Demistriped Functions
By leveraging Ghidra we can lift assembly code to pseudo C code, however in stripped binaries functions can be inlined or ```CALL``` instructions can be replaced by ```JUMP``` instructions. The actual recovering of function boundaries is separate research topic. Because of this imperfections the authors introduce demistriped functions. These are function that are lifted from binaries that are not striped, but we manually remove indentifiers like function names or variable names.

#### Remarks
In my opinion this is an highly unrealistic scenario, limiting any particular real world application.

## Performance
The authors compare BinT5 to [GraphCodeBERT](https://codebreakers.re/articles/detail/bert-codebert-and-graphcodebert/) and [PolyGlotCodeBERT](https://arxiv.org/abs/2112.02043) finetuned on Capybara dataset. On BLEU-4 BinT5 scores 5 to 3 points higher than the previously mentioned models. 

Overall the model struggles a lot with striped binaries, part of the strugle comes from the inherited flaws from the decompilation process, especially with inlined functions, where the decompiler can not recover the function boundaries.

In case of demi-striped functions the model performs significantly better, however this is not a real world scenario, rendering the model less useful.

### Ablevation

#### Duplicates
Capybara dataset contains some level of duplicates, this is natural thing, since different project thed to rely on shared libraries. By removing duplicates (near duplicates) from the dataset, the model looses a lot of its performance on Exact Match, but based on human evaluation the summaries are still reasonable.

#### Source Code Comments vs Function Names
From an performance perspective having comments in the code seems less important than actual identifiers, with function names contributing the most, but again in stripped binaries we do not have function names.

### Data Leakage

There are couple of concerns about data leakage from CodeT5. CodeT5 was pretrained on a dataset containing C/C# code, unfortunately the dataset is not public, so it is hard to assess if the foundational model was pretrained on source code that is similar (or the same) as used in BinT5. The authors claim that the dataleakage is minimal since the performance of the model is comparable to finetuned GraphCodeBERT and PolyGlotCodeBERT (They where not pretrained on any C code) on the Capybara dataset. 

## Remarks to BinT5

It is a nice first paper that applies T5 (or an LLM) to reverse engineering of stripped binaries. However the results are not that promising. The model works nice, but only on a very specific scenario (demi-striped), unfortunately a scenario that is not likely to happen in the real world. 

In my personal opinion, the authors missed the main point of T5, and they did not introduce any new pretraining objectives that would help the model to better understand the binary code. But luckily for us there is HexT5.

# HexT5

HexT5 continues where BinT5 left off, it introduce a couple of pretraining objectives that should help with the model to comprehend the binary code better. The authors also realize that existing reverse engineering methods are narrow focused on a single task like, function name recovery, variable names recovery, binary code summarization or binary function similarity detection. With HexT5 the authors want to tackle all of these tasks at once, stating that models that try to learn multiple objectives at once tend to perform better than models that are trained on a single task. In addition we work with binaries that where compiled for different architectures ```x86, x86_64, arm32, arm64```, different compiler flags: ```-O0, -O1, -O2, -O3)``` and four compilers clang-7.0, clang-9.0, gcc-7.2.0, gcc-8.3.0.

As with BinT5 it builds on top of T5 Base (220m).

## Input Data

As in BinT5 we do not leverage binary code (assembly) directly, but we first decompile it to pseudo C, however in this case we use [Hex-Rays](https://hex-rays.com/ida-pro/) instead of Ghidra. Since Hex-Rays is a commercial product (and an expensive one), with performance that is (arguably) better than Ghidra. 

As the source for the binaries it uses [GNU Binutils](https://www.gnu.org/software/binutils/), where the data is than partitioned in an cross project or cross binary fashion.

1. Cross Project
A binary for example ```ls``` or ```cat``` will be located in either training or test set, including all the functions from that binary, compiled with different compiler flags, different compiles and different architectures.
2. Cross Binary
This splists each binary into a set of functions, with each function being in either training or test set. If a function is in the training set, than all its versions (compiled with different compiler flags, different compiles and different architectures) are in the training set as well.

### Remarks
Cross Project partititions resembles the real world scenario, since it is unlikely that a reverse engineer will have access to some parts of the binary, but not to the others (Obviously this is a bit of a lie, since binaries tend to share libraries, third party code, etc.). Unfortunately Cross Project partitioning has worse performance than Cross Binary partitioning.


### DWARF

The authors use [DWARF](https://dwarfstd.org/) as a bridge between source code and binary code. DWARF is a debugging format that is used to map the binary code to the source code. The authors extract the symbols from the DWARF information, and map them to the decompiled pseudo C code using their addresses. By leveraging DWARF we are able to recover variable names, function names and other identifiers. However again, in stripped binaries the bigest obstacle is the function boundaries detection. Unfortunately this paper does not address it, making the model less useful in real world scenarios.

### Normalization

Variable and Function names in striped binaries tend to have no meaning, because of this we replace them with specially tokens. For variables we use tokens "\<VAR\>, \<VAR2\>, ..., \<VAR100\>, for the functions that we want to reverse engineer we reserve token "\<FUNC\>" all the functions that are being called from the inside we use "\<FUNC1\>, \<FUNC2\>, ..., \<FUNC50\>". Comments inside pseudo code are generated by the decompiler and they bear no meaning, because of this we delete them. There is one edge case, and that are variables that do not have any meaningful name, in this case we replace these variables with "\<UKN\>" instead of "<VAR...>".

## Pretraining Objectives

Before we dive into the objectives, lets us revisit the goal of the model, we want it to be able to recover variable names, function names, summarize binary code and detect binary code similarity. All of these tasks can be seen as a function which takes a programming language (In our case pseudo C) and it produces natural language, with the binary code similarity detection being a bit different.

Why is this important? Most of current LLMs are trying to give us an conversional agent like experience, HexT5 does not try to do that, and because of this we do not need an Natural Language to Programing Language alignment objective. By not being able to promnt the model we have it more constrained, but on the bright side It can more focus on the task at hand.

The meat of HexT5 are the following four pretraining objectives, Masked Span Prediction, Source Identifier Prediction Bimodal Single Generation and Contrastive Learning.

![T5](/images/hextt5_pretraining.png)

### Masked Span Prediction
This is essentially the same as in CodeT5, we take chunks of the input and we mask them, and we want the model to recover the masked tokens. This objective helps the alighment of pseudo code and comments.

### Source Identifier Prediction (SIP)

The actual objective is inspired by [DOBF](https://arxiv.org/abs/2102.07492), and the idea is to teach the model deobfuscation. That means we try to recover the obfuscated variable names and function names. In our cases we try to predict the values hidden behind "\<FUNC\>, \<FUNC..\>" and "\<VAR\>" tokens. 

### Bimodal Single Generation

Goal of the objective is to align the programming language to natural language generation. Here we feed the whole Pseudo C code to the model and we want it to generate a natural language summary of the code. From the models perspective this objective is the same as SIP, to distinguish between the two objectives the authors add a hard prompt token "summarize" before the Pseudo C code (for SIP the prompt token is "identifier_prediction").


### Contrastive Learning

The idea is to push the contextual representations of similar functions closer together and the representations of dissimilar functions further apart. We already saw this for [CodeT5+]({{< relref "posts/code-t5-plus.md" >}}), in case of HexT5 however we do not employ an Momentum Encoder, but instead we use the remaining samples from a given mini-batch as negative samples, and for the positive samples we use the same function but compiled with different compiler flags, different compiler and/or different architecture.

On thing that is not explicitly stated in the paper is if a given function is selected to be in an mini-batch, are all the versions of that function in the mini-batch as well, I expect that this is the case. 

Contrastive Learning is an sequence level objective, that means we take an Pseudo C code snippet, we pass it trough the encoder part of the model, this yields for each token an contextual representation, to actually get the embedding we average the contextuall representations. We take this representation and we pass it in the contrastive learning objective:

$$ L_{CL} = -\log \frac{e^{sim}(V,V^+)/\tau}{\sum_{j\in B} e^{sim (V,V_j^-)/\tau}} $$
- $sim$ is the similarity function, in our case cosine similarity
- $V^+$ are the positive samples
- $V^{-}$ are the negative samples, there is an additional way to sample the negatives, we can apply an random dropout mas to V_j, however I find this approach more confusing 

## Performance

Long story short, HexT5 did set the SoTA in Binary Code Summarization, Variable Name Recovery, Function Name Recovery and Binary Code Similarity Search. What is interesting that in Binary Code Summarization they also benchmarked it against [Gepetto](https://github.com/JusticeRage/Gepetto). Gepetto is a Ida-Pro plugin that leverages OpenAIs Chat-GPT to generate summaries for binary functions.

# Remarks

There are some obvious downsides of booth models. Function boundaries detection is a very serious problem, and 

## Data Leakage

My personal opinion is that data leakage is a serious concern, lets look at a paper from [Shang et. al 2024](https://arxiv.org/abs/2404.09836v1), this is a newever study applying LLMs like GPT-4 and CodeLlama 7B for reverse engineering. They state that these autoregressive causal models perform way better than BinT5 or HexT5 (for HexT5 the reported scores are vastly different between the papers). These big causal foundational models have the capacity to memorize the training data, and since the training data is not public (For CodeLlama we know it was pretrained on an additional 500B tokens or 864GB source code, GPT-4 is unknown but because of Github Copilots Codex we can assume it saw an incredible amount of (not just open source) source code).


# Disclaimer

