import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from jax import lax, jit, grad, vmap
from functools import partial

import numpy as np
from numpy.polynomial.legendre import leggauss

def compute_sigmas(quadrature, m, S):
    return m + (jnp.linalg.cholesky(S) @ quadrature.unit_sigmas[...,None]).squeeze(-1) # (n_quad**K, K)

def gaussian_int(fn, m, S, weights, unit_sigmas):
    """
    Approximates E[f(x)] wrt x ~ N(m, S) with Gauss-Hermite quadrature.
    fn: function with range in R

    Note, we do not use this to approximate kernel expectations because 
    there, fn(.) does not have range in R for memory efficiency reasons.
    """
    # sigmas = m + (jnp.linalg.cholesky(S) @ unit_sigmas[...,None]).squeeze(-1)
    # return jnp.dot(weights, vmap(fn)(sigmas))
    
    sigmas = m + (jnp.linalg.cholesky(S) @ unit_sigmas[...,None]).squeeze(-1) # (n_quad**K, K)
    
    def _step(carry, arg):
        weight, sigma = arg
        return carry + weight * fn(sigma), None

    arg = (weights, sigmas)
    approx_exp, _ = lax.scan(_step, 0., arg)
    return approx_exp

def gauss_legendre(n, a=-1., b=1.):
    """Computes weights and abscissas for Gauss-Legendre quadrature """
    h = b - a
    x, w = leggauss(n)

    # adjust integration limits
    if a != -1. or b != 1.:
        x = (x + 1.) * (h / 2.) + a
        w = (h / 2.) * w

    x = jnp.array(x)
    w = jnp.array(w)
    return x, w

def bin_regularly_sampled_data(dt, ys, bin_size):
    """
    Bin regularly-sampled observations into smaller time bins.

    dt: timestep used in EM algorithm
    ys: (n_trials, n_bins, D) observed data
    bin_size: amount of time per time bin in ys (must be same unit as dt)
    """
    n_trials, n_bins, D = ys.shape
    trial_duration = n_bins * bin_size 
    t_obs = np.arange(n_bins) * bin_size # time-stamps of observations
    t_mask, _ = np.histogram(t_obs, int(trial_duration / dt), (0, trial_duration)) # t_mask is the same across trials
    t_mask = t_mask.astype(bool) 
    ys_binned = np.zeros((n_trials, len(t_mask), D))
    ys_binned[:,t_mask.astype(bool),:] = ys
    t_mask = t_mask[None].repeat(n_trials, axis=0) # repeat across trial dimension
    
    return jnp.array(ys_binned), jnp.array(t_mask)
    
def bin_sparse_data(ys, t_obs, t_max, dt):
    """
    Bin sparsely sampled data into discrete time bins.
    TODO: potentially make more efficient?
    right now it is O(n_trials * n_samps_per_trial)

    Parameters
    -------------
    ys: (n_trials, n_samps, D) sparse observations
    t_obs: (n_trials, n_samps) timestamps of observations
    t_max: scalar, duration of trial
    dt: scalar, bin size

    Returns
    -------------
    ys_binned: (n_trials, T, D) binned observations
    t_mask: (n_trials, T) mask for observed timestamps
    """

    T = int(t_max / dt)
    n_trials, n_samps, D = ys.shape
    all_t_mask = []
    all_ys_binned = []

    for i in range(n_trials):
        hist, bins = np.histogram(t_obs[i], T, (0, t_max)) # (T, ) containing counts of obs in each time bin
        t_idx = np.nonzero(hist)[0] # containing time bin indices with >= 1 obs
        ys_binned = np.zeros((T, D))
        for j, idx in enumerate(t_idx):
            if j < len(t_idx) - 1:
                y_inds_in_bin = np.nonzero((bins[idx] <= t_obs[i]) & (t_obs[i] < bins[idx+1]))[0] # get indices of n_samps where obs are in this bin
                ys_binned[idx] = ys[i, y_inds_in_bin].mean(0) # taken mean of those obs
            else:
                y_inds_in_bin = np.nonzero((bins[idx] <= t_obs[i]) & (t_obs[i] <= bins[idx+1]))[0] # get indices of n_samps where obs are in this bin
                ys_binned[idx] = ys[i, y_inds_in_bin].mean(0) # taken mean of those obs

        all_t_mask.append(hist != 0)
        all_ys_binned.append(ys_binned)

    all_t_mask = np.stack(all_t_mask)
    all_ys_binned = np.stack(all_ys_binned)
    return jnp.array(all_ys_binned), jnp.array(all_t_mask)

def get_transformation_for_latents(C, C_hat):
    """
    Compute linear transformation matrix P which maps learned latents to true latents (or another latent space).

    Parameters:
    ------------
    C: (D, K) true affine mapping parameter
    C_hat: (D, K) learned affine mapping parameter

    Returns:
    ------------
    P: (K, K) linear map from learned latent space to true latent space
    """
    U, S, Vt = jnp.linalg.svd(C, full_matrices=False)
    P = Vt.T @ jnp.diag(1./S) @ U.T @ C_hat
    return P

def make_gram(kernel_fn, kernel_params, Xs, Xps, jitter=1e-8):
    """
    Compute gram matrix between inputs Xs and Xps.

    Parameters:
    ---------------
    kernel_fn: function from Kernel class
    kernel_params: dict
    Xs: (n_points, K) first batch of input points
    Xps: (n_points_2, K) second batch of input points
    jitter: optional jitter to add to gram matrix, should be None if Xs != Xps

    Returns:
    ---------------
    K: (n_points, n_points_2) gram matrix
    """
    K = vmap(vmap(partial(kernel_fn, kernel_params=kernel_params), (None, 0)), (0, None))(Xs, Xps)
    if jitter is not None:
        K += jitter * jnp.eye(len(Xs))
    return K

def get_induced_f(kernel_fn, kernel_params, Xs, zs, f_zs, jitter=1e-8):
    """Compute posterior mean and covariance of dynamics on Xs given observed values f_zs at zs."""
    Kzz = make_gram(kernel_fn, kernel_params, zs, zs, jitter=jitter)
    Kxz = make_gram(kernel_fn, kernel_params, Xs, zs, jitter=None)
    Kxx = make_gram(kernel_fn, kernel_params, Xs, Xs, jitter=jitter)
    f_mean = Kxz @ jnp.linalg.solve(Kzz, f_zs)
    f_cov = Kxx - Kxz @ jnp.linalg.solve(Kzz, Kxz.T)
    return f_mean, f_cov

def get_posterior_f_mean(kernel_fn, kernel_params, Xs, zs, q_u_mu, jitter=1e-8):
    """
    Compute posterior mean of dynamics on a new set of points given variational distribution of inducing points.
    
    Parameters:
    ---------------
    kernel_fn: function from Kernel class
    kernel_params: dict
    Xs: (n_points, K) new set of points in latent space
    zs: (M, K) locations of inducing points
    q_u_mu: (K, M) posterior mean at inducing points
    q_u_sigma: (M, M) posterior variance at inducing points (same across the D latent dimensions)

    Returns:
    ---------------
    f_mean: (n_points, K) posterior mean at new points
    """
    
    Kxz = make_gram(kernel_fn, kernel_params, Xs, zs, jitter=None)
    Kzz = make_gram(kernel_fn, kernel_params, zs, zs, jitter=jitter)
    
    f_mean = (Kxz @ jnp.linalg.solve(Kzz, q_u_mu.T)) # marginalized over q_u
    return f_mean

def get_posterior_f_var(kernel_fn, kernel_params, Xs, zs, q_u_sigma, jitter=1e-8):
    """Compute posterior variance of dynamics on a new set of points given variational distribution of inducing points."""
    Kxx = make_gram(kernel_fn, kernel_params, Xs, Xs, jitter=jitter)
    Kxz = make_gram(kernel_fn, kernel_params, Xs, zs, jitter=None)
    Kzz = make_gram(kernel_fn, kernel_params, zs, zs, jitter=jitter)

    f_var = jnp.diag(Kxx - Kxz @ jnp.linalg.solve(Kzz, Kxz.T) + Kxz @ jnp.linalg.solve(Kzz, q_u_sigma) @ jnp.linalg.solve(Kzz, Kxz.T)) # marginalized over q_u
    return f_var

def get_learned_partition(partition_fn, kernel_params, Xs):
    """
    Compute value of pi(x) at each point in Xs.

    Parameters:
    -------------
    partition_fn: construct_partition function from SSL kernel class
    kernel_params: SSL kernel params dict
    Xs: (n_points, K) batch of points to evaluate pi

    Returns:
    -------------
    learned_pis: (n_points, num_states) pi evaluated at Xs
    """
    learned_pis = vmap(partition_fn, (0, None, None))(Xs, kernel_params['W'], kernel_params['log_tau'])
    return learned_pis
    
def get_most_likely_state(partition_fn, kernel_params, Xs):
    """Compute most likely state at each point in Xs."""
    learned_pis = get_learned_partition(partition_fn, kernel_params, Xs)
    most_likely_states = jnp.argmax(learned_pis, 1)
    return most_likely_states
    
