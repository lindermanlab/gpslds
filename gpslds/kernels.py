import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax, jit, grad, vmap
import tensorflow_probability.substrates.jax as tfp
tfb = tfp.bijectors
from functools import partial
from utils import gaussian_int

class Kernel:
    def __init__(self, quadrature): 
        self.quadrature = quadrature
        
    # --------- Expectations computed with quadrature ----------
    def E_Kxx(self, m, S, kernel_params):
        """Computes E[k(x,x)] wrt q(x) = N(x|m,S)."""
        fn = lambda x: self.K(x, x, kernel_params)
        return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas) # scalar

    def E_Kxz(self, z, m, S, kernel_params):
        """Computes E[k(x,z)] wrt q(x) = N(x|m,S)."""
        fn = lambda x: self.K(x, z, kernel_params)
        return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas) # scalar

    def E_KzxKxz(self, z1, z2, m, S, kernel_params):
        """Computes E[k(z1,x)k(x,z2)] wrt q(x) = N(x|m,S)."""
        fn = lambda x: self.K(z1, x, kernel_params) * self.K(x, z2, kernel_params)
        return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas) # scalar

    def E_dKzxdx(self, z, m, S, kernel_params):
        """Computes E[dk(z,x)/dx] wrt q(x) = N(x|m,S)."""
        fn = grad(partial(self.K, z, kernel_params=kernel_params))
        return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas) # (D, )

class RBF(Kernel):
    def __init__(self, quadrature):
        super().__init__(quadrature)

    def K(self, x1, x2, kernel_params):
        """
        Computes RBF kernel between two inputs.

        Parameters
        -------------
        x1, x2: (K, ) input points
        kernel_params: dict containing
        - length_scales: (K, ) length scale for each state dimension
        - output_scale: scalar
        """
        sq_diffs = (((x1 - x2) / kernel_params["length_scales"])**2).sum()
        return kernel_params["output_scale"]**2 * jnp.exp(-0.5 * sq_diffs) # scalar

    # --------- Closed-form expectations ----------
    def E_Kxx_closed(self, m, S, kernel_params):
        return kernel_params["output_scale"]**2 # scalar

    def E_Kxz_closed(self, z, m, S, kernel_params):
        K = len(m)
        integral = tfd.MultivariateNormalFullCovariance(m, S + jnp.diag(kernel_params["length_scales"]**2)).prob(z)
        const = kernel_params["output_scale"]**2 * jnp.sqrt((2 * jnp.pi)**K) * kernel_params["length_scales"].prod()
        return const * integral

    def E_KzxKxz_closed(self, z1, z2, m, S, kernel_params):
        K = len(m)
        S_inv = jnp.linalg.solve(S, jnp.eye(K))
        L_inv = jnp.diag(1. / kernel_params["length_scales"]**2)
        linear_term = jnp.linalg.solve(S, m) + (z2 / kernel_params["length_scales"]**2)
        new_mean = jnp.linalg.solve(S_inv + L_inv, linear_term)
        new_cov = jnp.linalg.solve(S_inv + L_inv, jnp.eye(K)) + jnp.diag(kernel_params["length_scales"]**2)

        const = kernel_params["output_scale"]**4 * (2 * jnp.pi)**K * (kernel_params["length_scales"]**2).prod()
        prob1 = tfd.MultivariateNormalFullCovariance(m, S + jnp.diag(kernel_params["length_scales"]**2)).prob(z2)
        prob2 = tfd.MultivariateNormalFullCovariance(new_mean, new_cov).prob(z1)

        return const * prob1 * prob2 

    def E_dKzxdx_closed(self, z, m, S, kernel_params):
        Psi1 = self.E_Kxz(z, m, S, kernel_params)
        L = jnp.diag(kernel_params["length_scales"]**2)
        return Psi1 * jnp.linalg.solve(L + S, z - m) # (D, )

class SimpleLinear(Kernel):
    def __init__(self, quadrature, noise_var=1.):
        super().__init__(quadrature)
        self.noise_var = noise_var

    def K(self, x1, x2, kernel_params):
        """Linear kernel with M = I, c = 0."""
        return (x1 * x2).sum() + self.noise_var

class Linear(Kernel):
    def __init__(self, quadrature, noise_var=1.):
        super().__init__(quadrature)
        self.noise_var = noise_var

    def K(self, x1, x2, kernel_params):
        """Linear kernel with fixed M and noise_var, learnable c parameter."""
        c = kernel_params["fixed_point"]
        M = self.noise_var * jnp.ones(len(x1)) 
        return (M * (x1 - c) * (x2 - c)).sum() + self.noise_var

class FullLinear(Kernel):
    def __init__(self, quadrature):
        super().__init__(quadrature)

    def K(self, x1, x2, kernel_params):
        """
        Computes linear kernel with learnable M, c, noise_var.
        
        Parameters:
        ----------------
        x1, x2: (K, ) input locations
        kernel_params: dict containing
        - 'fixed_point': (K, )
        - 'log_M': (K, ) log diagonal of M
        - 'log_noise_var: scalar, log of noise_var
        """
        c = kernel_params["fixed_point"]
        M = jnp.exp(kernel_params["log_M"]) # (D, )
        noise_var = jnp.exp(kernel_params["log_noise_var"])
        return (M * (x1 - c) * (x2 - c)).sum() + noise_var

class SSL(Kernel):
    def __init__(self, quadrature, linear_kernel, basis_set=None):
        super().__init__(quadrature)
        self.linear_kernel = linear_kernel
        self.basis_set = basis_set

    def construct_partition(self, x, W, log_tau):
        """Construct partition function pi at a given latent space location x."""
        activations = W.T @ self.basis_set(x)
        pi = tfb.SoftmaxCentered().forward(activations / jnp.exp(log_tau))
        return pi

    def K(self, x1, x2, kernel_params):
        """
        Compute smoothly switching linear (SSL) kernel. 

        Parameters:
        --------------
        x1, x2: (K, ) input locations
        kernel_params: nested dict containing
        - linear_params: list of length num_states, where each entry is a dict containing linear kernel params
        - W: (num_bases, num_states-1) partition function basis weights
        - log_tau: scalar, partition function smoothing parameter
        """
        linear_params = kernel_params["linear_params"] # list of linear params, one dict per regime
        W = kernel_params["W"]
        log_tau = kernel_params["log_tau"]p
        pi_x1 = self.construct_partition(x1, W, log_tau)
        pi_x2 = self.construct_partition(x2, W, log_tau)
        linear_kernels = jnp.array([self.linear_kernel.K(x1, x2, param) for param in linear_params]) # (num_states,)
        return (pi_x1 * pi_x2 * linear_kernels).sum()

    