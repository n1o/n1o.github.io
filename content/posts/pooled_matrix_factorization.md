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

For building an recommendation engine we have to find our matrix $R$. That is a matrix that has a row for each user an column for each video with the entries of the matrix being the number of minutes an user spent watching the video. After some data wrangling we end up with this:

![pivot](/images/matrix_factorization_pivot.png)

That table has a lot of ```NaN``` entries, which is to be expected, since most users see only a couple of videos (the most popular ones, or videos that are somehow promoted). 

The actual fraction of observed entries is:
```python
1 - np.sum(np.sum(pd.isna(pivot))) / (pivot.shape[0] * pivot.shape[1])
0.004108162504450008
```

Our goal is to now somehow replace all those ```NaN``` number predictions. Now we could just continue and apply matrix factorization, or even probabilistic matrix factorization, but since we are Bayesians (or at least I am) we can perform some partial pooling. Now you may ask what the hell is pooling. Well pooling or partial pooling allows us to share data between groups. This allows groups with a small number of observations borrow statistical strength from groups that have a large number of observations.

```python
import pyro
```