+++ 
draft = true
date = 2020-12-17T10:19:42+01:00
title = "Pooled Probabilistic Matrix Factorization"
description = ""
slug = ""
authors = []
tags = ["probabilistic programming", "pyro"]
externalLink = ""
series = []
+++

Probabilistic matrix factorizations is a simple but useful model for building recommendations engines. The model can be expressed as:

$$
P(R|U,V, a^2) = \prod_{i=1}^N \prod_{j=1}^M[\mathcal{N}(R_{ij}|U_iV_j^T, \alpha^{-1})]^{I_{ij}}
$$


```python
import numpy as np
```