MNIST Multi-Class Classification Functions
=========================================

.. automodule:: modules.application_functions.bnn_functions_MNIST
   :members:
   :undoc-members:
   :show-inheritance:

**Main Functions and Classes:**

- **minibatch_selection**
  
  Performs mini-batch selection on SVGD particles for the inference loop.

- **DataLoader**
  
  Class for mini-batch selection on data for the inference loop. Provides an iterator over mini-batches.

- **inference_loop**
  
  Runs the SVGD inference loop with mini-batching over data and particles.

- **get_predictions**
  
  Returns class predictions for each sample using the ensemble of BNN particles (multi-class, using Categorical).

- **get_mean_predictions**
  
  Computes the most likely class for each sample by taking the mode across particles (multi-class).

- **get_probabilities**
  
  Returns class probabilities for each sample by averaging over particles (multi-class).

- **logprior_fn**
  
  Computes the log prior probability of the BNN parameters.

- **loglikelihood_fn**
  
  Computes the log likelihood of the data under the BNN model (multi-class, using Categorical).

- **logdensity_fn_of_bnn**
  
  Returns the unnormalized log posterior for the BNN.

- **fit_and_eval**
  
  Fits a BNN on train data using SVGD and evaluates on test data. Returns predictions and probabilities for train and test sets.

**Usage:**
See the :doc:`mnist_example` for a full example of how to use these functions for multi-class classification on MNIST. 