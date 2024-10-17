# General imports
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

# Imports from gpSLDS
from gpslds.quadrature import GaussHermiteQuadrature
from gpslds.likelihoods import Gaussian
from gpslds.sde import OrnsteinUhlenbeckSDE

# Other imports
from sdeint import itoEuler
def generate_fixed_dynamics(f, L, tgrid, n_sample, init_cond):
    """
    Generates samples from an SDE with drift coefficient f and diffusion L
    Note that the function f is assumed to be fixed here
    """
    return jnp.stack([itoEuler(f, L, init_cond, tgrid) for i in range(n_sample)], axis=0)

if __name__ == "__main__":

    # Generate synthetic dataset according to a one-dimensional Ornstein-Uhlenbeck process
    N = 1001
    ntrials = 1
    theta = 0.5
    x0 = 1
    tgrid = jnp.linspace(0.0, 10.0, N)
    f = lambda x, t: (-1)*theta*x
    L = lambda x, t: 1

    x = generate_fixed_dynamics(f, L, tgrid, x0, ntrials)

    # Randomly select 40 observations to observe
    seed = jr.PRNGKey(1)
    seed1, seed2 = jr.split(seed, 2)
    D = 40
    C, d = 2, 3
    sigma = 0.1
    y = tfd.Normal(loc=C * x + d, scale=sigma).sample(seed=seed2)
    idx = jr.choice(seed1, jnp.arange(N), (D,), replace=False)
    yobs = y[:, idx, :]
    t_mask = jnp.zeros((y.shape[0], y.shape[1]))
    t_mask = t_mask.at[:, idx].set(1)

    # Instantiate likelihood object
    likelihood = Gaussian(y, t_mask)   

    # Instantiate the prior SDE
    prior_sde = OrnsteinUhlenbeckSDE(decay = 1.2)

    # Instantiate the CVI SDE model