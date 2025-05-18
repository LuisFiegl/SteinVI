.. SteinVI documentation master file, created by
   sphinx-quickstart on Wed Mar  6 14:51:17 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

   <div style="font-size: 50px; color: black;">Welcome to<br>Randomized Stein VI</div>
===================================

This repository introduces a randomized version of Stein Variational Inference (also referred to as SVGD: Stein Variational Gradient Descent), based on the JAX and BlackJAX libraries. **We accellerated the inference procedure by mini-batching the data and SVGD particles**. Hereby, we have focussed on the application in Bayesian Neural Networks. Our proposed functions work with arbitrary (unnormalized) posteriors and offer plenty of scope for customisation. This repository also contains two example Jupyter notebooks for demonstration. 

Modules
==================

.. toctree::
   :maxdepth: 20
   :caption: Evaluation Modules:
   :includehidden:

   bnn_functions_linker
   svgd_function_linker
   mnist_functions

.. toctree::
   :maxdepth: 20
   :caption: Functions for the Finance Example:
   :includehidden:

   finance_functions_linker

Examples
==================

.. raw:: html

   <ul>
       <li><a href="single_bnn_withSVGD.html">Simple Moon-Dataset</a></li>
       <li><a href="finance_application.html">Finance</a></li>
       <li><a href="mnist_example.html">MNIST SVGD Classification</a></li>
       <li><a href="SVGD_idea.html">SVGD Idea Presentation</a></li>
   </ul>

.. toctree::
   :caption: Examples
   :hidden:

   single_bnn
   finance
   SVGD_idea
   mnist_example