import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax, jit, grad, vmap
from functools import partial
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

class SparseGP:
    def __init__(self, zs, kernel, jitter=1e-4):
        self.zs = zs
        self.kernel = kernel
        self.jitter = jitter

    def prior_term(self, q_u_mu, q_u_sigma, kernel_params):
        """Computes sum of KL[q(u_d)||p(u_d)]]."""
        Kzz = vmap(vmap(partial(self.kernel.K, kernel_params=kernel_params), (None, 0)), (0, None))(self.zs, self.zs) + self.jitter * jnp.eye(len(self.zs)) # (M, M)
        q_dist = tfd.MultivariateNormalFullCovariance(q_u_mu, q_u_sigma) # one sample is shape (K, M)
        p_dist = tfd.MultivariateNormalFullCovariance(0, Kzz) # one sample is shape (M, ) (prior dist is same across dimensions)
        kl = tfd.kl_divergence(q_dist, p_dist).sum()
        return -kl

    # --------- Closed-form expectations wrt q(f) and q(x) ----------

    def f(self, m, S, q_u_mu, q_u_sigma, kernel_params, Kzz_inv):
        """
        Computes E[f(x)] wrt q(f) and q(x) = N(x|m, S).

        Parameters
        ------------
        kernel: kernel instance
        zs: (M, K) inducing point input locations
        m: (K, ) variational mean of x
        S: (K, K) variational covariance of x
        q_u_mu: (K, M) variational mean of u
        q_u_sigma: (K, M, M) variational (diagonal) variance of u, factorized over M

        Returns
        -----------
        E_f: (K, ) expectation of f(x)
        """
        M, K = self.zs.shape
        E_Kxz = self.kernel.E_Kxz(self.zs, m, S, kernel_params)[None] # (1, M)
        # E_Kxz = vmap(partial(self.kernel.E_Kxz, m=m, S=S, kernel_params=kernel_params))(self.zs)[None] # (1, M)
        E_f = E_Kxz @ Kzz_inv @ q_u_mu.T
        return E_f[0]

    def ff(self, m, S, q_u_mu, q_u_sigma, kernel_params, Kzz_inv):
        """
        Computes E[f(x)'f(x)] wrt q(f) and q(x) = N(x|m, S).
        """
        M, K = self.zs.shape
        E_KzxKxz = self.kernel.E_KzxKxz(self.zs, m, S, kernel_params)
        # E_KzxKxz = vmap(vmap(partial(self.kernel.E_KzxKxz, m=m, S=S, kernel_params=kernel_params), (None, 0)), (0, None))(self.zs, self.zs) # (M, M)

        term1 = K * (self.kernel.E_Kxx(m, S, kernel_params) - jnp.trace(Kzz_inv @ E_KzxKxz))
        term2 = jnp.trace(Kzz_inv @ q_u_sigma.sum(0) @ Kzz_inv @ E_KzxKxz)
        term3 = jnp.trace(E_KzxKxz @ Kzz_inv @ q_u_mu.T @ q_u_mu @ Kzz_inv)
        return term1 + term2 + term3 # scalar

    def dfdx(self, m, S, q_u_mu, q_u_sigma, kernel_params, Kzz_inv):
        """
        Computes E[df/dx] wrt q(f) and q(x) = N(x|m, S).
        """
        M, K = self.zs.shape
        E_dKzxdx = self.kernel.E_dKzxdx(self.zs, m, S, kernel_params)
        # E_dKzxdx = vmap(partial(self.kernel.E_dKzxdx, m=m, S=S, kernel_params=kernel_params))(self.zs)
        return q_u_mu @ Kzz_inv @ E_dKzxdx # (D, D)
