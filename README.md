# Stochastic Stein VI in JAX: Experiments with Bayesian Neural Networks

[**Overview**](#overview)
| [**Example usage**](#example-usage)
| [**Note**](#note)
| [**Documentation**](https://luisfiegl.github.io/SteinVI/)

## Overview

This repository introduces a **stochastic version of Stein Variational Inference** (also referred to as SVGD: Stein Variational Gradient Descent), based on the JAX and BlackJAX libraries. SVGD is a powerful method for probabilistic modeling and JAX provides high-performance computing and automatic differentiation. We further accellerated the inference procedure by mini-batching the data and SVGD particles. Hereby, we have focussed on the application in Bayesian Neural Networks. Our proposed functions work with arbitrary (unnormalized) posteriors and offer plenty of scope for customisation. This repository also contains example Jupyter notebooks for demonstration.
Make sure to also check out our [documentation](https://luisfiegl.github.io/SteinVI/) for further explanantion of our modules.

## Example usage:

**Fitting and evaluating a Flax BNN with stochastic SVGD:**
<br>A simple imple self-contained binary classification problem
Simulating the datasets:
```py
import jax
import jax.numpy as jnp
from flax import linen as nn
import sys, os
sys.path.insert(0, os.path.abspath(".."))
from modules.evaluation_functions.bnn_functions import *

key = jax.random.PRNGKey(12)
n_samples = 100

X = jax.random.uniform(key, shape=(n_samples, 2))
Y = jnp.sum(X, axis=1) >= 1

Xs_train = X[: n_samples // 2 ,:]
Xs_test = X[n_samples // 2 :,:]
Ys_train = Y[: n_samples // 2]
Ys_test = Y[n_samples // 2 :]
```
Creating a BNN with Flax:
```py
hidden_layer_width = 5
n_hidden_layers = 2

class NN(nn.Module):
    n_hidden_layers: int
    layer_width: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.n_hidden_layers):
            x = nn.Dense(features=self.layer_width)(x)
            x = nn.tanh(x)
        return nn.Dense(features=1)(x)

bnn = NN(n_hidden_layers, hidden_layer_width)
```
Fitting and evaluating the BNN with stochastic Stein VI:
```py
Ys_pred_train, Ys_pred_test, _, Y_probabilities_train, Y_probabilities_test = fit_and_eval(
    key, bnn, logdensity_fn_of_bnn, Xs_train, Ys_train, Xs_test, None, num_steps=400,batch_size_particles = 20, batch_size_data = 32, num_particles=200
    )
```

## Note

This repository was created by Luis Fiegl and Jean-Pierre Weideman as part of the *Applied Deep Learning* course at LMU Munich, which took place in the winter term 23/24.