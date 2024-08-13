+++ 
draft = false
date = 2020-12-17T10:19:42+01:00
title = "Hierarchical Probabilistic Matrix Factorization"
description = "Hierarchical Probabilistic Matrix Factorization. An example implementation in Pyro."
slug = ""
authors = []
tags = ["probabilistic programming", "pyro"]
externalLink = ""
series = []
matrix_factorization_link = "https://app.diagrams.net/#G1H5MpK3xCDkBsIO5DCBKPNoUkVpES5Swp"
+++

Probabilistic Matrix factorization is a simple but useful model for matrix imputation. The main idea is to decompose a tall and wide matrix into a product of two matrices, one tall and thin and one short and wide.

$$
R_{n\times m} = U_{m \times d} \cdot V_{d \times n}
$$

![Matrix Factorization](/images/matrix_factorization.svg)

If you are a Bayesian, you can express this model as:

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

We can use this model when R is sparse, to try to fill in the missing entries. This is something that often occurs in practice. Let's look into the following example:

> You are working for an Advertisement Video on Demand (AVOD) company. The company is young, and there is a large marketing effort to acquire new users. Because of this, there are some challenges:
> * Each day the proportion of new users to recurring users is heavily skewed towards new users.
> * We expect to have a high churn rate
> 
> Our goal is to build a video recommendation engine. 

Building a recommendation engine is essentially the same as performing matrix imputations. Therefore our first goal is to find the right matrix $R$. In our case, $R$ has a row for every user that has watched something, and a column for every video that has been watched by someone. The entries of the matrix are the number of minutes a user spent watching the video. After some data wrangling we end up with this:

```
video_id	0	1	2	3	4	5	6	7	8	9	...	555	556	557	558	559	560	561	562	563	564
user_id																					
0	1.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1	1.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
2	NaN	3.0	NaN	NaN	NaN	NaN	NaN	8.0	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3	NaN	4.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
4	NaN	1.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
5 rows × 565 columns
```

That table has a lot of ```NaN``` entries. The fraction of observed entries is:
```python
1 - np.sum(np.sum(pd.isna(pivot))) / (pivot.shape[0] * pivot.shape[1])
0.004108162504450008
```
This is to be expected since most users watch only a couple of videos (the most popular ones, or videos that are somehow promoted).

Our goal is to replace all those ```NaN``` numbers with predictions. We could just continue and apply matrix factorization, or even probabilistic matrix factorization, but since we are Bayesian (or at least I am) we can do better and build a hierarchical model. Hierarchical modeling enables data sharing between groups. This allows groups with a small number of observations to borrow statistical strength from groups that have a large number of observations. In our case, it allows users that have seen only a few videos, to borrow information from users that are frequent watchers. To build a hierarchical model we have to define a prior over the distribution of users. Our model turns out to be:

$$
R_{ij} \sim \mathcal{N}(u_i \cdot v_j^T, \sigma)
$$
$$
v_j \sim \mathcal{N}(\mu_v, \Sigma_v)
$$
$$
u_i \sim \mathcal{N}(\mu_u, \Sigma_u)
$$
$$
\mu_u \sim \mathcal{N}(\mu_0, \Sigma_0)
$$

We could also define a prior for videos, which would pool less popular videos towards the more popular. But a a bit of good advice is to start simple and grow more complicated in time.

To get our hands a bit dirty let's look at the model implementation in [Pyro](http://pyro.ai/).

```python
 def model(alpha, dim, n, m, nan_mask, not_na, data):
        """
        Perform matrix factorization
        R = U @ V.T
        """
        alpha_loc = torch.tensor(1 / 25)

        loc_u = pyro.sample(
            "loc_u",
            dist.MultivariateNormal(
                loc=torch.zeros(dim),
                precision_matrix=torch.eye(dim) * alpha_loc,
            ),
        )
        precission_u = pyro.sample(
            "precission_u",
            dist.LKJCorrCholesky(
                d=dim, eta=torch.tensor(alpha)
            ),
        )

        observations_scale = pyro.sample(
            "obs_scale",
            dist.InverseGamma(
                concentration=torch.tensor(1.0),
                rate=torch.tensor(1.0),
            ),
        )

        with pyro.plate("users", n):
            U = pyro.sample(
                "U", 
                dist.MultivariateNormal(
                    loc=loc_u, 
                    precision_matrix=precission_u
                )
            )
        with pyro.plate("content", m):
            V = pyro.sample(
                "V", dist.MultivariateNormal(
                    loc=torch.zeros(dim), 
                    precision_matrix=torch.eye(dim)
                )
            )
        with pyro.plate("observations", not_na):
            R = pyro.sample(
                "R",
                dist.Normal(loc=(U @ V.T)[~nan_mask], scale=observations_scale),
                obs=data,
            )
```

Here we use vectorized operations instead of for loops for performance reasons. To perform inference we use Stochastic Variational Inference (SVI) with an AutoDiagonalNormal guide. For some of you this may sound scary, and I am planning to release a series of posts where we will look into Variational Inference (VI) in depth. To give you the long story short: Bayesian inference is about finding the distribution of parameters of interest. In Variational Inference we approximate this distribution by a set of simpler distribution, and we use optimization to make our approximation tight. Anyway lets get back to code:


```python
pyro.clear_param_store()
guide = AutoDiagonalNormal(model)
svi = SVI(model, guide, Adam({"lr": 0.001}), loss=Trace_ELBO())
n, m = train.shape
dim = 5
alpha = 2.0
iterations = 3000
train_loss = []
for i in range(iterations):
    loss = svi.step(alpha, dim, n, m, nan_mask, not_na, data)
    train_loss.append(loss / len(data))
```
After the optimization is done, we can use the guide to retrieve our parameters of interest. To perform matrix completion we are interested in matrix V and U. 

```python
V = guide.median()['V']
U = guide.median()['U']

R = U @ V.T
```

Matrix R holds the predicted number of minutes for all the users and all the movies. We can take these predictions and store them in a database and serve them to users when needed. 

We managed to get some predictions for users that we have seen before, now we can turn our attention to new users. This may sound complex, but since we have defined a distribution above all the users, we can use this distribution to generate new users. 

```python
loc = guide.median()['loc_u']
precision_matrix=guide.median()['precission_u']
U_pooled = dist.MultivariateNormal(loc=loc, precision_matrix=precision_matrix)

generated_users = torch.from_numpy(
    np.array([U_pooled.sample().detach().numpy() for _ in range(1000)])
)
```

Here we generated 1000 users from the posterior distribution. Each user is a sample from a five-dimensional Gaussian. To get their potential number of watched minutes, we have to multiply the generated samples with our matrix V.

```python
potential_watched_minutes = (generated_users @ V.T)
potential_watched_minutes.shape
>>> torch.Size([1000, 565])
```

The matrix potential_watched_minutes represents the predicted watched minutes for each movie in our repository. Since we do not expect to see any of those generated users, a good strategy is to average them out.

```python
average_potential_watched_minutes = potential_watched_minutes.mean(axis = 0)
average_potential_watched_minutes.shape
>>> torch.Size([565])

```

Here we ended with a vector. The ith entry of the vector is the average number of minutes we expect a random, not seen before user will spend watching the ith movie. We can store this vector (similarly to vector R), save it into a database, and serve it as predictions for a new user.

To wrap things up, probabilistic programming gives us superpowers. We can handle potential cases we have not seen before, by generating data from our model and use those generated samples to make decisions.

The full code example can be found [here](https://github.com/n1o/n1o.github.io/blob/master/notebooks/pooled_matrix_factorization.ipynb).
