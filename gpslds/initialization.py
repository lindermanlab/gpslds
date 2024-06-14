import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def initialize_zs(K, zs_lim, num_per_dim):
    """
    Initializes inducing points on a num_per_dim^K grid between -zs_lim and zs_lim.

    Parameters
    -------------------
    K: latent dimension
    zs_lim: positive scalar, bound on each axis of grid
    num_per_dim: number of inducing points per latent dimension

    Returns
    -------------------
    zs: (num inducing points, K) inducing points
    """
    assert zs_lim > 0, "zs_lim must be positive"
    
    zs_per_dim = jnp.linspace(-zs_lim, zs_lim, num_per_dim)
    all_zs_per_dim = [zs_per_dim for _ in range(K)]
    zs = jnp.stack(jnp.meshgrid(*all_zs_per_dim), axis=-1).reshape(-1, K)
    return zs

def initialize_affine_params(K, ys):
    """Initialize C, d with PCA (for Gaussian observation model)."""
    ys_stacked = jnp.vstack(ys) # (total_n_samps, K)
    d_init = ys_stacked.mean(0)
    ys_centered = ys_stacked - d_init
    U, S, Vt = jnp.linalg.svd(ys_centered, full_matrices=False)
    C_init = Vt[:K].T # (D, K)
    return C_init, d_init

def initialize_vem(n_trials, n_timesteps, K, M, I, mean_init, var_init):
    """Initialize parameters for variational EM algorithm"""
    S0 = var_init * jnp.repeat(jnp.eye(K)[None], n_trials, 0)
    V0 = var_init * jnp.repeat(jnp.eye(K)[None], n_trials, 0)
    As = (1. / var_init) * jnp.repeat(jnp.repeat(jnp.eye(K)[None], n_timesteps, 0)[None], n_trials, 0)
    bs = jnp.zeros((n_trials, n_timesteps, K))
    ms = jnp.zeros((n_trials, n_timesteps, K))
    Ss = (1. / var_init) * jnp.repeat(jnp.repeat(jnp.eye(K)[None], n_timesteps, 0)[None], n_trials, 0)
    q_u_mu = mean_init * jnp.ones((K, M))
    q_u_sigma = var_init * jnp.eye(M)[None].repeat(K, axis=0)
    B = jnp.zeros((K, I))

    return S0, V0, As, bs, ms, Ss, q_u_mu, q_u_sigma, B