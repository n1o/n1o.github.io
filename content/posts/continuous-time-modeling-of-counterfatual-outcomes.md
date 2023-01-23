---
title: "Paper overview: Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations"
date: 2023-01-16T17:37:36+01:00
draft: true
---

# The problem it solves
Imagine you have an irregullary sampled time series, where at various time points we perform interventions. These interventions may influence the dynamics of the timeseries. The question we want to answer is: 

*If I perform a hypothetical sequence of interventions how will my time series evolve?*

# An example and some details

## Example

Lets assume that we are a medical doctor in a hospital, and we have a patient that has high fever and is at risk of dying. A common practice is to measure his levels of C-reactive protein (CRP) to determine if he need antibiotics. Lets assume that his CRP is high and we administer a certain type of antibotics in a certain dosage. Antibiotics are not a magical wand and it takes some time till their effect shows, because of this we wait some time and we measure the patients CRP levels again. There are 3 general cases that can happe:

1. The CRP levels drop
2. The CRP levels go up
3. The CRP levels remain the same

Here we have to decide if we continue the treatment or change the type of antibiotics or their dosage.  

## Details
We can visuallize the example above as:

![Medical Example](/images/ct_cde_medical_example.jpg)

At every time step we measure the CRP levels. The CRP levels influence the dosage and type of antibiotics, they also are an indicator of how likely a patient is to die. Antibiotics should lover the probability of death by lowering the future CRP levels. This behaviour is intuitive but there is a hidden catch, we have introduced time-dependent confounding. To explain what it is lets start with time-dependent variables and time-dependent treatment.

### Time-Dependent variables
Are variables that change over time and they are repeatedly measured. In our case CRP levels are a time-dependent variable

### Time-Dependent treatment
Treatment (or just action) that is a response to a change in time-dependent variable. In our case this the type and dose of antibiotics.

### Time-Dependent confounding
It is a confounder that is affected by the previous treatment. In our case the current dose (type) of antibiotics has an influence of tuture doses (types) of antibiotics, or for those who have an machine learning background we can say that each treatment introduces an distribution shift.

# Counterfactuals, confounding  and bias

At the beggining we asked the following question:

*If I perform a hypothetical sequence of interventions how will my time series evolve?*

We already introduced time-dependent confounding, so we know that each time we perform some sort of intervention we potentially can introduce an distribution shift. If we want find out the effect of an hypothetical sequence of interventions, we need to take into account all the past interventions that have happend, and keep in mind that every one of the hypothetical can cause an distribution shift.

If we want to have an unbiased counterfactual estimate we have to remove confounding (close all the back doors).

# Treatment Effect Neural Controlled Differential Equation (TC-CDE) Model

The grand idea is that we take the time-depenent variables, time-dependent treatments and we use them to fit a neural controlled differential equation. A neural differential controlled differential equation is just a differential equation whose vector field is a neural network, where the solution to the differential equation is not determined only by its initial condition, but its trajectory is adjusted by subsequent observations.

Lets try to put in our inflamation context. Each time we sample the CRP levels, we measure how bad the inflamation is, and since our imune system is working 24/7 we may get better over time without need for any medication. We can view inflamation as a latent proces that is evolving in time, due our imune system or medication, and the CRP levels are realizations of this latent process. Modeling this latent process is hard, thus what TC-CDE model does is to express this latent process as a solution to a Neural Controlled differential equation.

For counterfactual estimation, modelling the latent process as an NCDE is not enough. To get an unbiased counterfactual estimate we need to make sure that the latent process is not predtive of the treatment that will be administered. And that if we administer a treatment at time t, it wont change the latent value at time t but only influence its future latent dynamics.

## A bit of Math

I allways try to avoid using equations, and I do fail all the time. Anyway for those who are more at home in machine learning and havent seen differential equations in a while (or ever), the werid integral can be viewed as a temporal evolution of some state, where the dynamics of the evolution is governed by a differential equation.

### Latent Path
To get the latent path we first have to define the initial state:

$$Z_{t0} = g_{\eta}(X_{t0},A_{t0}, Y_{t0})$$

- $g_{\theta}$ is a neural network that takes the initial covariates $X_{t0}$, treatment assignment $A_{t0}$ and observed outcome $T_{t0}$ and produces the initial latent space embedding $Z_{t0}$

Next we define our evolution of the initial state:

$$Z_t = S_{t0} + \int_{t_0}^t f_{\theta}(Z_s) \frac{d[S_s, A_s, Y_s]}{ds}ds $$

for $t \in (t_0, t]$

- $Z$ is a response that is a solution to NCDE
- $f_{\theta}$ is the latent vector field parametrized by a neural network


### Objective function

To mitigate confounding bias we have to make sure that the latent path is not predictive of future treatments to achieve this we introduce 2 neural networks

1. $h_v: R^l \rightarrow R^d$ used to predict the outcome $\hat{y}_s = h_v(Z_s)$

2. $h_a: R^l \rightarrow [0,1]$ used to predict the treatment $\hat{p}_s = \hat{p}(a_s = 1) = h_{a}(Z_s)$

If there are k>1 observations in the timewindow $[t,t']$ with observation times $(t_1, \cdots, t_k)$ our loss function is defined as

$$L = \frac{1}{n}\sum_{i=1}^n L_i^{(y)} - \mu L_i^{(a)} $$

- $L^{(y)} = \frac{1}{k} (y_{t_j} - \hat{y}_{t_j})^2$ this is the sequare mean of outcome prediction
- $L^{(a)} = - \frac{1}{k} \sum_{j=1}^k a_{t_j} \log(\hat{p}_{t_j}) + (1-a_j)\log (1 - \hat{p}_{t_j})$ this is the cross entropy of treatment predictions

Since there is a substraction before $\mu$ we essentially maximize the cross entropy ensuring that $z_t$ is not predictive of treatment assignment $A_t$. This leads to balancing representations removing bias form time-dependent confounders allowing reliable counterfactual estimation where $\mu$ controlls the tradeoff between treatment and outcome predictions.

### Hypothetical path

Our initial goal was to evaluate the effect of some hypothetical treatment schedule. Because of this we need to go from $t$ to some future $t'$:

$$Z_{t'} = Z_t = \int_{t}^{t'}f_{\theta}(Z_s)\frac{dA_s}{ds}ds $$
- $Z_t$ is the encoded latent state of a patient up to time t
- $A_s$ is the hypothetical treatment schedule $t < s < t'$

To get the predicted outcome we use our neural network trained before

$$\hat{y}_{t'} = h_v(Z_{t'})$$

# Summary

Being able to generate unbiased estimates of some outcome value given hypothetical treatment schedules (or just some sort of actions) is a powerful tool. And not requiring an fixed schedule for these actions makes it also very versatile. And I look forward for extensions on these method.

# Disclaimer

I did read the original paper, and look at the code that was supplied with it. I did write this article on my own, however since I am not a native speaker, and I do not have an easy access to anybody who could reliably prof read it, I did use ChatGPT for proof reading and improving my stilization. If you are really interested how the original was worded. I will provide a link below.

# Sources
1. https://github.com/seedatnabeel/TE-CDE
2. https://arxiv.org/abs/2206.08311
3. 
