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

## Data format
To use the gpSLDS on your own data, you will need to ensure that you have:
- A JAX array `ys_binned` of shape `(n_trials, n_timesteps, n_output_dims)`. To process data in effectively continuous-time, `n_timesteps` should represent the number of time bins at a small discretization step relative to the data sampling rate. We assume that data has been zero-padded in the case of varying length trials.
- A JAX array `t_mask` of shape `(n_trials, n_timesteps)`. This is 1 for observed timesteps and 0 for unobserved timesteps.
- A JAX array `trial_mask` of shape `(n_trials, n_timesteps)`. This is 1 for timesteps in an observed trial and 0 for a zero-padded timestep.
- (Optional) A JAX array `inputs` of shape `(n_trials, n_timesteps, n_input_dims)` consisting of external stimuli.

For an example, please see `synthetic_data_demo.ipynb` which demonstrates data formatting and model fitting on a synthetic example.
