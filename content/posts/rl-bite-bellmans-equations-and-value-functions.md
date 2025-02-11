+++ 
draft = false
date = 2025-02-12T07:00:52+01:00
title = "RL Bite: Bellmans Equations and Value Functions"
description = ""
slug = ""
authors = []
tags = ["RL Bite", "Value Based RL", "Bellman"]
categories = ["RL Bite", "Value Based RL", "Bellman"]
externalLink = ""
series = ["RL Bite", "Value Based RL", "Bellman"]
+++

# Value Based Reinforced Learning
In value based Reinforced Learning we learn a **Value Function**:

$$ V_{\pi}(s) = E_{\pi}[G_0|s_0 = s] = E_{\pi}[\sum_{t=0}^T \gamma^t r_t|s_0  = s] $$

- $G_t$ is the **Total Return at time t**, this is just the sum of Rewards an Agent gets walking the trajectory T (fancy name but this is just a sequence of actions the agent takes)
    - $\gamma^t$ is the **Discount Factor**, long story short this is between $<0,1>$, the closer this value is to zero, the more the agent will focus on the immediate reward, the closer it is to 1 the more it will take future rewards into account. In general using $\gamma=1$ is bad if we do not have an explicit terminating state and the agent will run forever (because of this we can say that the discount factor forces the agent to finish). If we use $\gamma=0$ then we are essentially blind, only caring for the immediate reward, thus it is best to stay away from the boundaries.

The Value Function measures the expected total return if we start at $s$ and we transition across states based on the policy $\pi$

## Q Function a.k.a: Action-Value Function
We will talk mostly about Value Functions, however they are a bit constrained as they do not take the action an agent can take. The value function can be easily extended:

$$Q_{\pi}(s,a) = E_{\pi}[G_0, s_0 = s, a_0 = a] = E_{\pi}[\sum_{t=0}^T \gamma|s_0  = s, a_0 = a] $$

And a Q function is born, which is also known as **Action-value Function**, since we take the action also into account. Learning Q functions is a bit more involved, but way more useful, and I will make a separate post just about them.

# Bellman's Equations
Bellman's equations give us a framework to learn an optimal policy $\pi^*$ for a Value Function. All the equations are true for the Q function as well!

## Optimal Policy

The idea is that there exists an optimal policy $ \pi^{\star}$ for which $V_{\pi^{\star}} \ge V_{\pi}$. For a given Markov Decision Process (MDP) there can be multiple optimal policies, but for a **finite MDP** there has to be at least one **Deterministic** optimal policy.

## Equations

$$V_{\star}(s) = \max_a R(s,a) + \gamma E_{p_S(s'|s,a)} [\max_{a'}V_{*}(s',a')]$$

$$Q_{\star}(s,a) = R(s,a) + \gamma E_{p_S(s'|s,a)} [\max_{a'}Q_{*}(s',a')]$$

These beauties above are the **Bellman's Optimality Equations**, and they are satisfied when we follow the Optimal Policy. This means if we do not have an optimal policy we have an error, meaning $V_{\pi^{\star}} = V_{\star} + \delta(s)$, this difference $\delta(s)$ is called **Bellman's Error**.

## Bellman's Operator

Bellman's Operator is an update rule that is used to derive a new Value Function by minimizing the Bellman's Error.

$$V'(s) = B_M^\pi V(s) \triangleq E_{\pi(a|s)} [R(s, a) + \gamma E_{T(s'|s, a)} [V(s')]]$$

## Bellman's Backup
**Bellman's Backup** is just repeated (iterative) application of Bellman's Operator to a state, that is guaranteed to converge to the Optimal Value Function:

$$ \pi_{\star}(s) = \arg \max_a Q_*(s, a)$$

$$ = \arg \max_a [R(s, a) + \gamma E_{p_S(s'|s, a)} [V_*(s')]] $$

## Summary
This is a lot to take in, but a summary is: 
To define an Optimal Value function, which is a Value Function that has the highest average Reward for a given Markov Decision Process (Agent), we can use Bellman's Equations. These equations hold only if the Value Function is optimal. The difference between the Optimal Value Function and the (Not-Optimal) Value Function is the Bellman's Error. To minimize this Bellman's Error we define the Bellman's Operator, which is just a rule how to derive a new Value Function while minimizing Bellman's Error, and by repeating Bellman's Operator we converge to the Optimal Value Function.
