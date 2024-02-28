# Stochastic Stein VI in JAX: Experiments with Bayesian Neural Networks

[**Overview**](#overview)
| [**Example usage**](#example-usage)
| [**Note**](#note)
| [**Documentation**](https://luisfiegl.github.io/SteinVI/)

## Overview

This repository introduces a **stochastic version of Stein Variational Inference** (also referred to as SVGD: Stein Variational Gradient Descent), based on the JAX and BlackJAX libraries. SVGD is a powerful method for probabilistic modeling and JAX provides high-performance computing and automatic differentiation. We further accellerated the inference procedure by mini-batching the data and SVGD particles. Hereby, we have focussed on the application in Bayesian Neural Networks. Our proposed functions work with arbitrary (unnormalized) posteriors and offer plenty of scope for customisation. This repository also contains example Jupyter notebooks for demonstration.

## Example usage:

**Creating an initial SVGD state, using a BNN:**

```py
initial_position = model.init(init_key, jnp.ones(X_train.shape[-1]))
_, unravel_fct = ravel_pytree(initial_position)

logprob = partial(logdensity_fn, model=model, unravel_function = unravel_fct)

num_bnn_parameters = sum(p.size for p in jax.tree_util.tree_flatten(initial_position)[0])
initial_particles = jax.random.normal(jax.random.PRNGKey(3),shape=(num_particles,num_bnn_parameters))

svgd = svgd_function.svgd(jax.grad(logprob), optax.sgd(0.3), svgd_function.rbf_kernel, svgd_function.update_median_heuristic)
initial_state = svgd.init(initial_particles, svgd_function.median_heuristic({"length_scale": 1}, initial_particles))
```

**Fitting and evaluating a Flax BNN with stochastic SVGD:**

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
```py
Ys_pred_train, Ys_pred_test, ppc_grid_single = fit_and_eval(
    eval_key, bnn, logdensity_fn_of_bnn, Xs_train, Ys_train, X_test, grid, num_steps=400,batch_size_particles = 20, batch_size_data = 32, num_particles=200
    )
```

## Note

This repository was created by Luis Fiegl and Jean-Pierre Weideman as part of the Applied Deep Learning course at LMU Munich, which took place in the winter term 23/24.