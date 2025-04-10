+++ 
draft = false
date = 2025-04-10T10:59:56+02:00
title = "RL Bite: Monte Carlo Search Tree"
description = ""
slug = ""
authors = []
tags = ["RL Bite", "Monte Carlo", "Model Based RL"]
categories = ["RL Bite", "Monte Carlo", "Model Based RL"]
externalLink = ""
series = ["RL Bite", "Monte Carlo", "Model Based RL"]

+++

# Abstract

Lets talk a bit about Model Based Reinforced Learning. The idea is that our RL Agent not just learns an policy to follow or/and a value function (Q function) but also tries to model the environment it is in. This is done by learning the transition dynamics $p(s'|s,a)$ (also known as World Model) and a Reward function $\hat{R}(s,a)$. Once we have our world model we can use it to simulate data and learn the models policy on the simulations. So here you may ask why the hell this is useful? First Sample Efficiency! We are able to generate extra data to use during the training. Second in some cases it is super expensive to explore the environment directly (imagine building an robot performing surgery! right). In practice how we train these models is way more interleaved, we do a bit of model learning, than policy improvement, than back to model learning and so fort. Why so? Agan this goes back to the Deadly Triad, where we use the model itself to improve itself, where we end up in this  self-reinforcing feedback loop making everything just worse.


# Decision-time planning
Before go into Monte Carlo Tree Search, lets generalize it a bit. So what is decistion time plannig? Well before an agent makes an step, it does a bit of planing ahead. This is done by taking an known (or learned ahead) world model and explores actions it can take, leading to new states, with further potential actions to take. Since this is an problem where the complexity is exponential we usually bound the maximum exploration depth (usually denoted $d$), and the approach described is also known as **receding horizon control**. Once we expanded the full tree, at every leaf we than compute the reward-to-go (this is the total reward that we would get if we start a the starting node $s_t$ and we walk all the way down to a leaf), and we choose trajectory with yields the highest total reward (so this proach is also known as **forward search**

## Optimizations
Exponential is bad, so lets look into how to make it more efficient:

1. Branch And Bound, this is an heuristics where we eliminate walking down unfavourable paths. It reequires a lower-bound on V and upper-bound on Q. In each state we visit, we order the actions we can take based on their upper-bound, if we find actions that are less than the current best lower bound, we prune it from the tree and we continue this until we hit a leaf node, where we return the lower bound 
2. Sparse Sampling, to reduce the states we need to explore, at each state we sample a set of actions we can take. Yes we may miss the optimal path.
3. Monte Carlo Tree Search

# Monte Carlo Tree Search

As before we start at a root node $s_t$ and we perform $m$ Monte Carlo Rollouts to estimate $Q(s_t, a)$ and return the best action: $a = \arg \max_a Q(s_t,a)$ or distribution of acttions (softmax).
During the rollout we track how many times we have tried each action $a$ in each state $s$ using a counter $N(s,a)$

The rollout has the following steps:
1. If we have not visited $s$ we initialize $N(s,a)=0$ and $Q(s,a)=0$ and we return the estimated value function ($U(s)$)
2. If we have visisted we pick the next action to explore from state $s$, here need to pick every action at least once
3. Once we took each action $a$ exactly once, and than we use an Upper Confindence Bound ([UBC](https://n1o.github.io/posts/rl-bite-exploration-vs-exploitation/#upper-confidence-bound-and-thompson-sampling)) to select subsequent actions:
$$ a = \arg \max_{a} Q(s, a) + c \sqrt{\frac{\log N(s, a)}{N(s, a)}}$$
    - $N(s) = \sum_a N(s,a)$
    - and we than sample the next state $s' \sim p(s'|s,a)$
4. Update the Q function using Temporal Difference ([Bellmans Update](https://n1o.github.io/posts/rl-bite-bellmans-equations-and-value-functions/#bellmans-operator))
$$ $Q(s, a) \leftarrow Q(s, a) + \frac{1}{N(s, a)}(u - Q(s, a))$$
5. We increment $N(s,a)$
![](/images/mcts_alog.png)

## Application
The algorithm is not too complex, lets look an simplified overview where it was applied and how:
### AlphaGo and Extensions 

AlphaGo is just the application of MCTS to two player, zero-sum symmetric game

$$ p^{\pi}(s'|s, a^{i}) = \sum_{a^{j}} \pi(a^{j}|\psi(s)) p(s'|s, a^{i}, a^{j})$$
- $i$ is the main player and $j$ is the opponent

## AlphaZeroGo, AlphaZero
We introduce an Neural Network in MCTS to compute $(v^s, \pi^s) = f(s;\theta)$

- $v^s$ is the expected outcome of the game from state $s$ (+1, -1, 0) (win, loss, draw)
- $\pi_s$ is the policy gives an distribution over actions for state $s$, and is used internally by MCTS to give additional exploration bonus to the most likely actions

The overall loss:
$$ L(\theta) = E_{(s, \pi_{s}^{MCTS}, V^{MCTS}(s)) \sim D} [ (V^{MCTS}(s) - V_{\theta}(s))^{2} - \sum_{a} \pi_{s}^{MCTS}(a) \log \pi_{\theta}(a|s) ]$$
- $ D = \{ (s, \pi_{s}^{MCTS}, V^{MCTS}(s)) \}$ is the dataset collected from a MCTS rollout started at state s
- $\pi_{s}^{MCTS}(a) = [ N(s, a) / (\sum_{b} N(s, b)) ]^{1/\tau}$ this is a distribution over actions at the rood node s with $\tau$ is the temperature
- $V_{s_{t}}^{MCTS} = \sum_{i=0}^{n-1} \gamma^{i} r_{t+i} + \gamma^{k} v_{t+i}$ this is the n step reward to go starting at $s_t$

### MuZero and Extension
AlphaGO assumes that the world model is known, where in MuZero we learn it. The model learned by training a latent representation (embedding function) of the observation $z_t = e_{\phi}(o_t)$, and learning the corresponding latent dynamics model:
$$ (z_t, r_t) = M_{w}(z_t,a_t) $$

The goal of the World Model is to predict the immediate reward and future reward of MCTS to compute the optimal policy

The total loss is:
$$L(\theta, w, \phi) = E_{(o, a_{t}, r, o', \pi_{z}^{MCTS}, V_{z}^{MCTS}) \sim D} \{ (V^{MCTS}(z) - V_{\theta}(e_{\phi}(o)))^{2} - \sum_{a} \pi_{z}^{MCTS}(a) \log \pi_{\theta}(a|e_{\phi}(o)) + (r - M_{w}^{r}(e_{\phi}(o), a_{t}))^{2} \}$$

It has some extra terms to measure how well it predicts the observed rewards, and we optimize it wtr:
- policy/value parameters $\theta$
- model parameters $\theta$
- embedding parameters $\phi$

#### Extensions 
Some notable extensions are:
- **Stochastic MuZero**, for stochastic environments
- **Sampled MuZero**, for large action spaces

