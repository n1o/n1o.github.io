+++ 
draft = false
date = 2025-04-01T08:44:11+02:00
title = "RL Bite: Monotonic Policy Improvement and Deriving Proximal Policy Optimization (PPO)"
description = ""
slug = ""
authors = []
tags = ["RL Bite", "Policy Learning"]
categories = ["RL Bite", "Policy Learning"]
externalLink = ""
series = ["RL Bite", "Policy Learning" ]
+++

# Abstract

A while ago we looked into [Policy Gradient and Reinforce]({{< relref "posts/rl-bite-policy-gradient-and-reinforce" >}}). Policy gradient is versatile and under mild conditions it is guaranteed to converge to a local minimum (if we choose the correct policy and step size). This is already a huge step up when compared to Q Learning, which may just diverge. However, we may still want stronger guarantees like monotonic improvement at each step.

# The Policy Improvement Lower Bound

This will be a bit math heavy, but the idea is to figure out the lower bound between the new policy $\pi$ and current policy $\pi_k$. Once we have figured it out we can optimize this lower bound using Total Variational Distance (later we use KL divergence), giving us monotonic policy improvement guarantees at each step.

1. Let's derive the Lower Bound:
$$ J(\pi) - J(\pi_k) \geq \frac{1}{1 - \gamma} E_{p_{\pi_k}^{\gamma}(s)\pi_k(a|s)} \left[ \frac{\pi(a|s)}{\pi_k(a|s)} A^{\pi_k}(s, a) \right] - \frac{2 \gamma C^{\pi, \pi_k}}{(1 - \gamma)^2} E_{p_{\pi_k}^{\gamma}(s)}[TV(\pi(\cdot|s), \pi_k(\cdot|s))]$$

- $L(\pi, \pi_k)=E_{p_{\pi_k}^{\gamma}(s)\pi_k(a|s)} \left[ \frac{\pi(a|s)}{\pi_k(a|s)} A^{\pi_k}(s, a) \right]$ this is the surrogate objective
- $\frac{2 \gamma C^{\pi, \pi_k}}{(1 - \gamma)^2} E_{p_{\pi_k}^{\gamma}(s)}[TV(\pi(\cdot|s), \pi_k(\cdot|s))]$ this is a penalty term
- $A^{\pi_k} = Q^{\pi_k}(s,a) - V^{\pi_k}(s)$  is the advantage function
- $\pi^{\gamma}_{\pi_k}$ is the normalized discounted state visitation distribution for $\pi_k$
- $C^{\pi, \pi_k} = \max_s |E_{\pi(a|s)}[A^{\pi_k}(s, a)]|$ 

2. Total Variation Distance

$$TV(p, q) = \frac{1}{2} ||p - q|| = \frac{1}{2} \sum_{s} |p(s) - q(s)|$$
- $||p-q||$  is the $l_1$ norm

3. TV is hard to optimize, because of that we do a slight modification by introducing a trust-region update

$$ \pi_{k+1} = \underset{\pi}{\text{argmax}} \ L(\pi, \pi_k) \ \text{s.t.} \ E_{p_{\pi_k}^{\gamma}(s)}[TV(\pi, \pi_k)(s)] \leq \epsilon$$

This is just constrained optimization, where we constrain the worst case performance decline at each update step.

## Trust Region?
So what the hell is trust region update? Long story short, take gradient descent or any local descent method (called local since we do an approximation at a local point), in general we need to choose a step size. There are some methods where this step size is not fixed but may increase and decrease. In trust region methods, the step size is adjusted based on the quality of the local approximation. The higher the quality of the local approximation (more similar to the function we approximate) the longer steps we can take and vice versa. Because of this we can define the trust-region as a region where the approximation is very high, and we can make a step right at the border of this region.


# Trust Region Policy Optimization (TRPO)

In practice we use KL Divergence instead of TV! 

1. We redefine the objective

$$ \pi_{k+1} = \underset{\pi}{\text{argmax}} \ L(\pi, \pi_k) \ \text{s.t.} \E_{p_{\pi_k}^{\gamma}(s)} [D_{KL}(\pi_k || \pi)(s)] \leq \delta $$

- there is an equivalence between TV and KL divergence if $\delta = \frac{\epsilon^2}{2}$

2. To get the loss we perform a first-order expansion of the surrogate objective:

$$ L(\pi, \pi_k) = E_{p_{\pi_k}^{\gamma}(s) \pi_k(a|s)} \left[ \frac{\pi(a|s)}{\pi_k(a|s)} A^{\pi_k}(s, a) \right] \approx g_k^T (\theta - \theta_k)$$

- here $g_k = \nabla_{\theta}L(\pi_{\theta}, \pi_{k})|_{\theta_k}$

3. Approximate the KL term:

$$ E_{p_{\pi_k}^{\gamma}(s)}[D_{KL}(\pi_k || \pi)(s)] \approx \frac{1}{2} (\theta - \theta_k)^T F_k (\theta - \theta_k)$$
- $F_k = g_kg_k^T$ is the Fisher Information Matrix

4. Now we can express the update rule as:

$$ \theta_{k+1} = \theta_k + \mu_ v_k$$
- with $v_k = F_k^{-1}g_k$ is the Natural gradient Descent
    - in practice we compute $v_k$ by approximately solving the linear system $F_k v = g_k$ using Conjugate Gradient
- $\mu_v = \sqrt{\frac{2\delta}{v_k^T F_k v_k}}$

## Fisher Information Matrix
In Machine Learning, we do a lot of Maximum Likelihood Estimation (MLE) $p(x|\theta)$, thus we maximize the likelihood of the data under certain parameters. This is extremely different to a Bayesian approach where we would maximize $p(\theta|x)$ or the parameters given the data we have. We can use the Fisher Information Matrix to measure how stable our MLE estimate is. In our case we can say that we use it to measure how stable our policy estimation/learning is.
## Natural Gradient Descent
Natural Gradient Descent is a slight extension of Gradient Descent where we use the Fisher Information Matrix for preconditioning the Gradient.

$$-\mu \F^{-1}g$$

So what the hell does this mean? If you look at the equation, and you're kind of familiar with Newton's Method, we swap the inverse Hessian with the inverse Fisher Information Matrix. For those that are not familiar with Newton's Methods: The approach is a second order optimization method. Second order because we do not only use linear approximation given by the gradient, but include curvature from the second derivative in our approximation. Thus in Natural Gradient Descent we replace the second order derivative information with the Fisher Information Matrix. 

## Conjugate Gradient Descent
The intuition is as follows: First imagine that we optimize an n dimensional quadratic function, this can be done in n steps, if we use second order optimization. Why n steps? Well first by using second order information, we estimate the curvature, but since our function is quadratic, this approximation is perfect. Because of this we can individually optimize each dimension and arrive at a global minimum (a quadratic function is convex we have no local minima).

# Proximal Policy Optimization (PPO)

PPO by itself is very anticlimactic. Why? Because it is again just a simplification, in this case of TRPO (which itself is a simplification)

1. Constraint

$$ E_{p_{\pi_k}^{\gamma}(s)}[TV(\pi, \pi_k)(s)] = \frac{1}{2} E_{(s, a) \sim p^{\gamma}_{\pi_k}} \left[ \left| \frac{\pi(a|s)}{\pi_k(a|s)} - 1 \right| \right]$$

Here we require that the support of $\pi$ is contained in the support of $\pi_k$ at every state.

2. Update Rule

$$\pi_{k+1} = \underset{\pi}{\text{argmax}} \ E_{(s, a) \sim p^{\gamma}_{\pi_k}} [\min(\rho_k(s, a) A^{\pi_k}(s, a), \tilde{\rho}_k(s, a) A^{\pi_k}(s, a))]$$
- $\rho_k (s,a) = \frac{\pi(a|s)}{\pi_k(a|s)}$ is the likelihood ratio
- $\tilde{\rho_k}(s,a) = clip(\rho_k(s,a), 1 - \epsilon, 1 + \epsilon)$, where clip(x,l,u) = min(max(x, l),u)

That's it! No magic, just an simplification, of a simplification of an lower-bound to constrain monotonic policy improvement.
