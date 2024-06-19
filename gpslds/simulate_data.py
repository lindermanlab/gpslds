import jax
jax.config.update("jax_enable_x64", True)
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, lax
from jax.nn import softplus
from functools import partial
import pickle

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

def simulate_sde(key, x0, f, dt, n_timesteps, inputs=None, B=None, sigma=1.):
    """
    General-purpose function for simulating a latent SDE according to drift function f.

    Parameters:
    -------------
    key: jr.PRNGKey
    x0: (K, ) initial condition
    f: drift function
    dt: integration timestep
    t_max: total duration of trial
    inputs: (n_timesteps, I) or None, inputs in discrete time
    B: (K, I) or None, direction of input-driven dynamics
    sigma: noise variance of simulated latents

    Returns:
    -------------
    xs: (n_timesteps, K) simulated latent SDE path
    """
    def _step(x, arg):
        key, input = arg
        next_x = tfd.Normal(x + f(x) * dt + B @ input * dt, scale=jnp.sqrt(sigma * dt)).sample(seed=key).astype(jnp.float64)
        return next_x, x

    if inputs is None:
        inputs = jnp.zeros((n_timesteps, 1))
        B = jnp.zeros((len(x0), 1))
        
    keys = jr.split(key, n_timesteps)
    _, xs = lax.scan(_step, x0, (keys, inputs))
    return xs

def simulate_gaussian_obs(key, xs, C, d, R):
    """
    Simulate Gaussian observations at every timestep of a latent SDE.

    Parameters:
    -------------
    key: jr.PRNGKey
    xs: (n_timesteps, K) latent SDE path
    C: (D, K) output mapping from latents to observations
    d: (D, ) bias in output mapping
    R: (D, ) observation noise variance

    Returns:
    -------------
    ys_dense: (n_timesteps, D) noisy observations
    """
    simulate_single_obs = lambda key, x: tfd.Normal(C @ x + d, scale=jnp.sqrt(R)).sample(seed=key).astype(jnp.float64)
    keys = jr.split(key, len(xs))
    ys_dense = vmap(simulate_single_obs)(jnp.array(keys), xs)
    return ys_dense

def simulate_poisson_obs(dt, key, xs, C, d, link):
    """
    Simulate Poisson process observations from a latent SDE.
    y|x ~ Pois(link_fn(Cx + d)*dt)

    Parameters:
    ---------------
    dt: time discretization at which to simulate observations
    key: jr.PRNGKey
    xs: (n_timesteps, K) latent SDE path
    C: (D, K) output mapping from latents to observations
    d: (D, ) bias in output mapping
    link: str indicating link function in {'exp', 'softplus'}
    
    Returns:
    ---------------
    poisson_obs: (n_timesteps, D) Poisson counts
    log_rate: (n_timesteps, D) log intensity function
    """
    if link == 'exp':
        link_fn = jnp.exp
    elif link == 'softplus':
        link_fn = softplus
    activations = (C[None] @ xs[...,None]).squeeze(-1) + d # (n_timesteps, K)
    rate = link_fn(activations) * dt
    poisson_obs = tfd.Poisson(rate=rate).sample(seed=key) # (n_timesteps, K)
    return poisson_obs, rate
    
    