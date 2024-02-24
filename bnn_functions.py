import jax
from jax import jit

from datetime import date
from functools import partial
from warnings import filterwarnings

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from sklearn.datasets import make_moons
from sklearn.preprocessing import scale
from jax.flatten_util import ravel_pytree

from tqdm import tqdm
import optax
import svgd_function


filterwarnings("ignore")
import matplotlib as mpl

cmap = mpl.colormaps["coolwarm"]

@jit
def rbf_kernel(x, y, length_scale=1):
    arg = ravel_pytree(jax.tree_util.tree_map(lambda x, y: (x - y) ** 2, x, y))[0]
    return jnp.exp(-(1 / length_scale) * arg.sum())

@jit
def median_heuristic(kernel_parameters, particles):
    particle_array = jax.vmap(lambda p: ravel_pytree(p)[0])(particles)

    def distance(x, y):
        return jnp.linalg.norm(jnp.atleast_1d(x - y))

    vmapped_distance = jax.vmap(jax.vmap(distance, (None, 0)), (0, None))
    A = vmapped_distance(particle_array, particle_array)  # Calculate distance matrix
    pairwise_distances = A[
        jnp.tril_indices(A.shape[0], k=-1)
    ]  # Take values below the main diagonal into a vector
    median = jnp.median(pairwise_distances)
    kernel_parameters["length_scale"] = (median**2) / jnp.log(particle_array.shape[0])
    return kernel_parameters

def minibatch_selection(num_particles, num_steps, batch_size, rng_key):
    all_params = jnp.arange(0, num_particles)
    num_batches_per_epoch = num_particles//batch_size
    num_epochs = round((num_steps / num_batches_per_epoch)+0.49)
    selected_params = []

    keys_to_iterate = jax.random.split(rng_key, num=num_epochs)

    for key in keys_to_iterate:
        shuffled = jax.random.shuffle(key, all_params)
        for i in range(num_batches_per_epoch):
            batch = shuffled[i*batch_size:(i+1)*batch_size]

            if len(selected_params)<num_steps:
                selected_params.append(batch)
    
    return selected_params

class DataLoader:
    def __init__(self, X, Y, batch_size=32, shuffle=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_datapoints = len(self.X)
        self.indices = np.arange(len(self.X))
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_X = self.X[batch_indices]
        batch_Y = self.Y[batch_indices]

        self.current_index += self.batch_size

        if self.current_index >= self.num_datapoints:
            self.current_index = 0
            if self.shuffle:
                np.random.shuffle(self.indices)

        return batch_X, batch_Y

def inference_loop(rng_key,Xs_train, Ys_train, step_fn, initial_state, num_samples, num_particles, batch_size_particles, batch_size_data):

    all_indices = jnp.array(minibatch_selection(num_particles, num_samples, batch_size_particles, rng_key))
    
    dataloader = DataLoader(Xs_train, Ys_train, batch_size=batch_size_data, shuffle=True)

    def one_step(state, selected_indices, **grad_params):
        selected_indices = jnp.array(selected_indices)
        state = step_fn(state, selected_indices, **grad_params)
        return state
    
    state = initial_state
    for i in tqdm(range(all_indices.shape[0])):
        x_batch, y_batch = next(dataloader)
        state = one_step(state, all_indices[i], X=x_batch, Y=y_batch)

    return state

def get_predictions(model, samples, X, rng_key):
    vectorized_apply = jax.vmap(model.apply, in_axes=(0, None), out_axes=0)
    z = vectorized_apply(samples, X)
    predictions = tfd.Bernoulli(logits=z).sample(seed=rng_key)

    return predictions.squeeze(-1)

def get_mean_predictions(predictions, threshold=0.5):
    # compute mean prediction and confidence interval around median
    mean_prediction = jnp.mean(predictions, axis=0)
    print(mean_prediction[3])
    return mean_prediction > threshold

def logprior_fn(params):
    leaves, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([jnp.ravel(a) for a in leaves])
    return jnp.sum(tfd.Normal(0, 1).log_prob(flat_params))


def loglikelihood_fn(params, X, Y, model, unravel_function):
    params_dict = unravel_function(params)
    logits = jnp.ravel(model.apply(params_dict, X))
    return jnp.sum(tfd.Bernoulli(logits).log_prob(Y))


def logdensity_fn_of_bnn(params, X, Y, model, unravel_function):
    return logprior_fn(params) + loglikelihood_fn(params, X, Y, model, unravel_function)

def fit_and_eval(
    rng_key,
    model,
    logdensity_fn,
    X_train,
    Y_train,
    X_test,
    grid,
    num_samples=400,
    batch_size_particles = 20,
    batch_size_data = 32,
    num_particles = 200
):
    (
        init_key,
        inference_key,
        train_key,
        test_key,
        grid_key,
    ) = jax.random.split(rng_key, 5)

    initial_position = model.init(init_key, jnp.ones(X_train.shape[-1]))
    _, unravel_fct = ravel_pytree(initial_position)

    # initialization
    logprob = partial(logdensity_fn, model=model, unravel_function = unravel_fct)

    num_bnn_parameters = sum(p.size for p in jax.tree_util.tree_flatten(initial_position)[0])
    initial_particles = jax.random.normal(jax.random.PRNGKey(3),shape=(num_particles,num_bnn_parameters))

    #optax.adam(0.3)
    svgd = svgd_function.svgd(jax.grad(logprob), optax.sgd(0.3), rbf_kernel, svgd_function.update_median_heuristic)
    initial_state = svgd.init(initial_particles, median_heuristic({"length_scale": 1}, initial_particles))
    
    step_fn = jax.jit(svgd.step)
    final_state = inference_loop(inference_key,X_train, Y_train, step_fn, initial_state, num_samples, num_particles, batch_size_particles, batch_size_data)

    particle_dicts = final_state.particles
    samples = jnp.apply_along_axis(unravel_fct, arr=particle_dicts, axis=1)

    predictions = get_predictions(model, samples, X_train, train_key)
    Y_pred_train = get_mean_predictions(predictions)
    predictions = get_predictions(model, samples, X_test, test_key)
    Y_pred_test = get_mean_predictions(predictions)
    if grid != None:
        pred_grid = get_predictions(model, samples, grid, grid_key)
    else:
        pred_grid = grid

    return Y_pred_train, Y_pred_test, pred_grid