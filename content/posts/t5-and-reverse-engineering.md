+++ 
draft = true
date = 2024-06-12T14:07:17+02:00
title = "T5 and Reverse Engineering"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

# Abstract

For a while now I have a new passion and that is binary reverse engineering and vulnerability exploitation, and I gone even so far that I created [CodeBreakers](https://codebreakers.re) which is a platform soly focusing on applying machine learning to reverse engineering, vulnerability detection (exploitation) and other cyber security related applications.

There are two approaches where [T5]({{< relref "posts/t5-the-old-new-thing.md" >}}) was applied for reverse engineeringg, one is [BinT5](https://arxiv.org/abs/2301.01701) and the other [HexT5](https://www.semanticscholar.org/paper/HexT5%3A-Unified-Pre-Training-for-Stripped-Binary-Xiong-Chen/04c3fccfe01f42afe18dcdb027385f350ab3c9d1) but before we dive into details about these papers let's first take a look at the basics of reverse engineering.


## Reverse Engineering

To understand reverse engineering we need to start with compilation. Compilation is the process of translating a high-level programming language like C (Yes! C is a high level programming language) to machine code. Machine code is something that your CPU can directly execute. The actual execution of the machine code is bit more involved, since an instruction like "add" can be further broken down into micro-code instructions, instruction can actually be executed out of order and with speculative execution things get messy really fast, and a there is a lot of room for mistakes and security vulnerabilities.

But to keep thing simple lets continue with machine code. Machine code is just a sequence of bytes. These bytes can be instruction or data. (Code is data, data is code is one of the basic idea of Von Neumann's architecture). These bytes can be without loose of generality reconstructed back to assembly code. Assembly code is a human readable representation of machine code. One thing to note is that being human readable does not mean that it is easy to understand.

To give you an example of the following C code:

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

# BinT5
# HexT5
