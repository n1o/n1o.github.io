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
matrix_factorization_link = "https://app.diagrams.net/#G1H5MpK3xCDkBsIO5DCBKPNoUkVpES5Swp"
+++

Probabilistic Matrix factorization is a simple but useful model for matrix imputation. The main idea is to decompose a tall and wide matrix into a product of two matrices, one tall and thing and one short and wide.

$$
R_{n\times m} = U_{m \times d} \cdot V_{d \times n}
$$

![Matrix Factorization](/images/matrix_factorization.svg)

If you are a Bayesian, you can express this models as:

$$
R_{ij} \sim \mathcal{N}(u_i \cdot v_j^T, \sigma)
$$
$$
u_i \sim \mathcal{N}(\mu_u, \Sigma_u)
$$
$$
v_j \sim \mathcal{N}(\mu_v, \Sigma_v)
$$

* $u_i$ is row in matrix $U$
* $v_j$ is a column in matrix $V$
* $R_{ij}$ is an entry in matrix R

We can use this model when R is sparse, to try to fill in the missing entries. This is something that often occurs in practice. Lets look into the following example:

> You are working for an Advertisement Video on Demand (AVOD) company. The company is young and there is an huge marketing effort to acquire new users. Because of this there are some challenges:
> * Each day the proportion of new users to recurring users is heavily skewed towards new users.
> * We expect to have a high churn rate
> 
> Our goal is to build an video recommendation engine. 



```python
import numpy as np
```