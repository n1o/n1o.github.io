---
title: "Paper review: Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations"
date: 2023-01-16T17:37:36+01:00
draft: true
---

# General idea
We have an irregullary sampled time series, where at various time points we perform interventions. These interventions may influence the dynamics of the timeseries. The question we want to answer is: 

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
It is a confounder that is affected by the previous treatment. What this is saying that if we want an unbiased measurement of treatment effect, we have to take into account all the previously applied treatments. 