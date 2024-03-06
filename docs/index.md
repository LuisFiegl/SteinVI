- [Introduction](#introduction)
- [Examples](#examples)
  - [Finance](Examples/finance.md)


# Welcome to Blackjax!


```{warning}
The documentation corresponds to the current state of the `main` branch. There may be differences with the latest released version.
```


Blackjax is a library of samplers for [JAX](https://github.com/google/jax) that works on CPU as well as GPU. It is designed with two categories of users in mind:


- People who just need state-of-the-art samplers that are fast, robust and well tested;
- Researchers who can use the library's building blocks to design new algorithms.


It integrates really well with PPLs as long as they can provide a (potentially unnormalized) log-probability density function compatible with JAX.




# Hello World


```{code-block} Python
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np


import blackjax


observed = np.random.normal(10, 20, size=1_000)
def logdensity_fn(x):
    logpdf = stats.norm.logpdf(observed, x["loc"], x["scale"])
    return jnp.sum(logpdf)


# Build the kernel
step_size = 1e-3
inverse_mass_matrix = jnp.array([1., 1.])
nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)


# Initialize the state
initial_position = {"loc": 1., "scale": 2.}
state = nuts.init(initial_position)


# Iterate
rng_key = jax.random.key(0)
step = jax.jit(nuts.step)
for _ in range(1_000):
    rng_key, nuts_key = jax.random.split(rng_key)
    state, _ = nuts.step(nuts_key, state)
```


:::{note}
If you want to use Blackjax with a model implemented with a PPL, go to the related tutorials in the left menu.
:::




# Installation


::::{tab-set}


:::{tab-item} Latest
```{code-block} bash
pip install blackjax
```
:::


:::{tab-item} Nightly
```{code-block} bash
pip install blackjax-nightly
```
:::


:::{tab-item} Conda
```{code-block} bash
conda install blackjax -c conda-forge
```
:::


::::


:::{admonition} GPU instructions
:class: tip


BlackJAX is written in pure Python but depends on XLA via JAX. By default, the
version of JAX that will be installed along with BlackJAX will make your code
run on CPU only. **If you want to use BlackJAX on GPU/TPU** we recommend you follow
[these instructions](https://github.com/google/jax#installation) to install JAX
with the relevant hardware acceleration support.
:::


```{toctree}
---
maxdepth: 1
caption: Examples
hidden:
---
Finance <Examples/finance.md>
```


* **Neural network API** (`flax.linen`): Dense, Conv, {Batch|Layer|Group} Norm, Attention, Pooling, {LSTM|GRU} Cell, Dropout
