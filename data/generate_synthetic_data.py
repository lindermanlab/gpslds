import jax
jax.config.update("jax_enable_x64", True)
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
from functools import partial
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import pickle

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gpslds.simulate_data import simulate_sde, simulate_poisson_obs

def generate_synthetic_dynamics(x,
                                bin_size=0.01,
                                rot_scale=1.003,
                                theta_left=-jnp.pi / 100.,
                                theta_right=jnp.pi / 100.,
                                fp_left=jnp.array([-3., 0.]),
                                fp_right=jnp.array([+3., 0.]),
                                tau=0.4):
    """
    Generate 2D dynamics consisting of two rotational linear systems separated by a vertical decision boundary.
    This function can be used to reproduce the main synthetic data results of the paper.

    Parameters:
    -------------
    x: (K, ) location in latent space
    bin_size: discrete timestep to generate rotation dynamics
    rot_scale: multiplicative factor for rotation dynamics
    theta_left: rotation angle for system active to the left of x1 = 0
    theta_right: rotation angle for system active to the right of x1 = 0
    fp_left: fixed point for system active to the left of x1 = 0
    fp_right: fixed point for system active to the right of x1 = 0
    tau: dynamics smoothness parameter 

    Returns:
    -------------
    f: (K, ) dynamics at x
    """
    K = 2 # latent dimension

    # rotation dynamics in discrete time, i.e. x_{k+1} = x_k + Ax_k + b + eps, eps~N(0, bin_size)
    rot_left = rot_scale * jnp.array([[jnp.cos(theta_left), -jnp.sin(theta_left)],
                   [jnp.sin(theta_left), jnp.cos(theta_left)]])
    rot_right = rot_scale * jnp.array([[jnp.cos(theta_right), -jnp.sin(theta_right)],
                   [jnp.sin(theta_right), jnp.cos(theta_right)]])
    
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
    basis_set = lambda x: jnp.array([1., x[0], x[1]])
    W = jnp.array([0., 1., 0.])[:,None]
    activations = W.T @ basis_set(x)
    weights = tfb.SoftmaxCentered().forward(activations / tau)
    f = weights[1] * (A_left @ x + b_left) + weights[0] * (A_right @ x + b_right)
    return f

def sample_latents_and_observations(key,
                           dt=0.001,
                           t_max=2.5,
                           D=50,
                           n_trials=100,
                           x0_left=jnp.array([-7., +0.]),
                           x0_right=jnp.array([+7., +0.])):
    """
    Sample latent states and Poisson process observations from synthetic dynamics.
    This function can be used to reproduce the main synthetic data results of the paper.

    Parameters:
    ------------------
    key: jr.PRNGKey
    dt: integration timestep for gpSLDS
    t_max: duration of each trial (seconds)
    D: output dimension
    n_trials: number of trials
    x0_left: (K, ) initial condition for first half of trials (left of decision boundary)
    x0_right: (K, ) initial condition for second half of trials (right of decision boundary)

    Returns:
    ------------------
    xs: (n_trials, n_timesteps, K) latent states
    spikes: (n_trials, n_timesteps, D) poisson process observation timestamps
    C: (D, K) output mapping from latents to observations
    d: (D, ) bias in output mapping
    """
    K = 2
    n_timesteps = int(t_max / dt)
    
    # generate initial conditions
    x0s_left = jnp.tile(x0_left[None], (int(n_trials/2), 1))
    x0s_right = jnp.tile(x0_right[None], (n_trials - int(n_trials/2), 1))
    x0s = jnp.vstack((x0s_left, x0s_right))
    
    # simulate latent SDE
    key, *keys_xs = jr.split(key, n_trials + 1)
    xs = vmap(partial(simulate_sde, f=generate_synthetic_dynamics, dt=dt, n_timesteps=n_timesteps, sigma=1.))(jnp.array(keys_xs), x0s)

    # generate affine mapping params
    key, key_signs, key_C, key_d = jr.split(key, 4)
    random_signs = tfd.Bernoulli(probs=0.5).sample((D, K), seed=key_signs) * 2 - 1 # random choice {-1, 1}
    C = tfd.Uniform(0, 5).sample((D, K), seed=key_C).astype(jnp.float64) * random_signs.astype(jnp.float64)
    d = tfd.Normal(6, 1).sample((D, ), seed=key_d).astype(jnp.float64) 
    
    keys = jr.split(key, n_trials)
    poisson_obs, rates = vmap(partial(simulate_poisson_obs, dt, C=C, d=d, link='softplus'))(jnp.array(keys), xs)
    spikes = (poisson_obs > 0).astype(jnp.float64)

    return dt, xs, spikes, C, d

def main():
    """Generate synthetic data and save to pickle file"""
    
    # generate synthetic data
    key = jr.PRNGKey(420)
    dt, xs, spikes, C, d = sample_latents_and_observations(key)

    # save dataset
    file_name = "synthetic_data.pkl"
    dataset = [dt, xs, spikes, C, d]
    with open(file_name, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    main()