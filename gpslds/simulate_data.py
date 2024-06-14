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

def generate_synthetic_data(key, D=50, n_trials=100, t_max=2.5):
    """
    Simulate 2D dynamics consisting of two rotational linear systems separated by a vertical decision boundary, and sample latents and Poisson process observations.
    This function can be used to reproduce the main synthetic data results of the paper.

    Parameters:
    ---------------
    key: jr.PRNGKey
    D: output dimension
    n_trials: number of trials
    t_max: duration (in seconds) of each trial

    Returns:
    --------------
    dt: integration timestep for gpSLDS
    true_f: dynamics function
    xs: (n_trials, n_timesteps, K) latent states
    spikes: (n_trials, n_timesteps, D) poisson process observation timestamps
    C: (D, K) output mapping from latents to observations
    d: (D, ) bias in output mapping
    """
    def f_weights(x, W, tau, basis_set):
        activations = W.T @ basis_set(x)
        weights = tfb.SoftmaxCentered().forward(activations / tau)
        return weights # (num_states, )
    
    K = 2 # latent dimension
    bin_size = 0.01 # equivalent time-step in discrete-time

    # ----------- GENERATE TRUE DYNAMICS ------------
    # rotation dynamics in discrete time, i.e. x_{k+1} = x_k + Ax_k + b + eps, eps~N(0, bin_size)
    rot_scale = 1.003
    theta_left = -jnp.pi / 100. # state active when x < 0
    rot_left = rot_scale * jnp.array([[jnp.cos(theta_left), -jnp.sin(theta_left)],
                   [jnp.sin(theta_left), jnp.cos(theta_left)]])
    theta_right = jnp.pi / 100. # state active when x > 0
    rot_right = rot_scale * jnp.array([[jnp.cos(theta_right), -jnp.sin(theta_right)],
                   [jnp.sin(theta_right), jnp.cos(theta_right)]])
    
    fp_left = jnp.array([-3., 0.])
    fp_right = jnp.array([+3., 0.])
    
    A_left_dsc = rot_left - jnp.eye(K)
    A_right_dsc = rot_right - jnp.eye(K)
    b_left_dsc = -A_left_dsc @ fp_left
    b_right_dsc = -A_right_dsc @ fp_right

    # convert to continuous time
    A_left = A_left_dsc / bin_size
    A_right = A_right_dsc / bin_size
    b_left = b_left_dsc / bin_size
    b_right = b_right_dsc / bin_size

    # define true dynamics function
    W = jnp.array([0., 1., 0.])[:,None]
    tau = 0.4
    basis_set = lambda x: jnp.array([1., x[0], x[1]])
    def true_f(x):
        # weights = f_weights(x, W, tau, basis_set)
        activations = W.T @ basis_set(x)
        weights = tfb.SoftmaxCentered().forward(activations / tau)
        return weights[1] * (A_left @ x + b_left) + weights[0] * (A_right @ x + b_right)

    # ------------- SAMPLE TRUE LATENT STATES -------------
    dt = 0.001
    n_timesteps = int(t_max / dt)
    
    # generate initial conditions
    x0s_left = jnp.tile(jnp.array([[-7., +0.]]), (int(n_trials/2), 1))
    x0s_right = jnp.tile(jnp.array([[+7., +0.]]), (n_trials - int(n_trials/2), 1))
    x0s = jnp.vstack((x0s_left, x0s_right))
    
    # simulate latent SDE
    key, *keys_xs = jr.split(key, n_trials + 1)
    xs = vmap(partial(simulate_sde, f=true_f, dt=dt, n_timesteps=n_timesteps, sigma=1.))(jnp.array(keys_xs), x0s)

    # ------------- SAMPLE OBSERVATIONS ---------------
    # generate affine mapping params
    key, key_signs, key_C, key_d = jr.split(key, 4)
    random_signs = tfd.Bernoulli(probs=0.5).sample((D, K), seed=key_signs) * 2 - 1 # random choice {-1, 1}
    C = tfd.Uniform(0, 5).sample((D, K), seed=key_C).astype(jnp.float64) * random_signs.astype(jnp.float64)
    d = tfd.Normal(6, 1).sample((D, ), seed=key_d).astype(jnp.float64) 
    
    keys = jr.split(key, n_trials)
    poisson_obs, rates = vmap(partial(simulate_poisson_obs, dt, C=C, d=d, link='softplus'))(jnp.array(keys), xs)
    spikes = (poisson_obs > 0).astype(jnp.float64)
    
    return dt, true_f, xs, spikes, C, d
    
    