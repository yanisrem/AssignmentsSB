# Final Project: Mid-Term Prediction of Industrial Production (2000-2006): Bayesian vs. Frequentist Approach

## Introduction

This repository contains the project based on "Macro1.csv" dataset.

## Features
Python files where algorithms are implemented : 

* ```gibbs_sampler.py```: Gibbs sampler implementation from homework n°2.
* ```init_parameter.py```: Initialize parameters for linear model with Gibbs Sampler.
* ```mh_rw.py```: Metropolis-Hastings Random Walk to sample $\sigma^2$ according to its prior distribution.
* ```preprocessing.py```: Data pre-processing (management of missing values, normalization, stationarization, train-test division).
* ```vectorize.py```: Vectorize data and parameters for computational challenges.

Notebook files :

* ``` frequentist_linear_model.ipynb ```: Frequentist linear model to predict industrial production.
* ``` gibbs_linear_model.ipynb ```: Bayesian linear model with Gibbs sampler to predict industrial production.
* ``` hamiltonien_monte_carlo.ipynb ```: Bayesian linear model with Hamiltonien Monte Carlo sampler to predict industrial production.
* ``` preprocessing.ipynb ```:  Study and data preprocessing.
* ``` random_walk_metropolis.ipynb ```: Metropolis-Hastings Random Walk to sample $\sigma^2$ according to its prior distribution.
