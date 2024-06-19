# gpSLDS
NOTE: in progress...

This repository contains an implementation of Gaussian Process Switching Linear Dynamical Systems (gpSLDS), described in the paper here (insert link).

## Repo structure
```
gpslds/                         Source code for gpSLDS model implementation.
    em.py                           Implements variational EM and contains the main gpSLDS fitting function.
    initialization.py               Functions for initializing model parameters.
    kernels.py                      GP kernel functions, including our smoothly switching linear kernel.
    likelihoods.py                  Gaussian and Poisson observation models.
    quadrature.py                   Quadrature object for approximating kernel expectations.
    simulate_data.py                Helper functions for sampling from the model.
    transition.py                   Defines GP object for model fitting.
    utils.py                        Variety of helper functions.
data/                           Code and data files for main synthetic data example.
    fit_plds.py                     Script for fitting Poisson LDS to initialize Poisson Process observation model parameters.
    generate_synthetic_data.py      Script for generating synthetic data.
    synthetic_data.pkl              Pickle file containing synthetic data.
    synthetic_plds_emissions.pkl    Pickle file containing initial observation model parameters for synthetic data.
synthetic_data_demo.ipynb       Demo notebook fitting gpSLDS to synthetic data.
```

