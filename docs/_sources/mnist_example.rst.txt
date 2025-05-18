MNIST SVGD Classification Example
===============================

This example demonstrates how to use Stein Variational Gradient Descent (SVGD) for Bayesian neural network (BNN) inference on the MNIST dataset for multi-class classification.

Overview
--------
- Uses a multi-layer perceptron (MLP) with Flax/JAX.
- Inference is performed using SVGD, implemented in `modules/evaluation_functions/bnn_functions_MNIST.py`.
- The model outputs 10 logits (one per MNIST class).
- The code is compatible with JAX transformations and multi-class classification.

Usage
-----
1. Prepare the MNIST dataset using scikit-learn's `fetch_openml` and preprocess as needed.
2. Define the neural network model:

   .. code-block:: python

      class NN(nn.Module):
          n_hidden_layers: int
          layer_width: int

          @nn.compact
          def __call__(self, x):
              for i in range(self.n_hidden_layers):
                  x = nn.Dense(features=self.layer_width)(x)
                  x = nn.tanh(x)
              return nn.Dense(features=10)(x)  # 10 classes for MNIST

3. Use the functions from `modules/evaluation_functions/bnn_functions_MNIST.py` to perform SVGD inference:

   .. code-block:: python

      from modules.evaluation_functions.bnn_functions_MNIST import fit_and_eval, logdensity_fn_of_bnn
      # ... set up model, data, and rng_key ...
      results = fit_and_eval(
          rng_key, model, logdensity_fn_of_bnn, X_train, y_train, X_test,
          grid=None, num_steps=40, batch_size_particles=20, batch_size_data=32, num_particles=100
      )
      Ys_pred_train, Ys_pred_test, _, _, _ = results

4. Compute accuracy:

   .. code-block:: python

      train_acc = (Ys_pred_train == y_train).mean()
      test_acc = (Ys_pred_test == y_test).mean()
      print(f"Train accuracy: {train_acc:.2%}")
      print(f"Test accuracy: {test_acc:.2%}")

Related Code
------------
- :mod:`modules.evaluation_functions.bnn_functions_MNIST`
- :mod:`modules.evaluation_functions.svgd_function` 