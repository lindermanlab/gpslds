import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax, jit, grad, vmap
import tensorflow_probability.substrates.jax as tfp
tfb = tfp.bijectors
from functools import partial
from .utils import gaussian_int

from abc import ABC, abstractmethod

class Kernel(ABC):
    def __init__(self, quadrature): 
        self.quadrature = quadrature
        
    # --------- Expectations computed with quadrature ----------
    def E_Kxx(self, m, S, kernel_params):
        """Computes E[k(x,x)] wrt q(x) = N(x|m,S)."""
        fn = lambda x: self.K(x, x, kernel_params)
        return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas) # scalar

    def E_Kxz(self, t, z, m, S, kernel_params):
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
    
    @abstractmethod
    def K(self, x1, x2, kernel_params):
        raise NotImplementedError
    
    def make_gram(self, kernel_params, zs, zps, jitter=1e-8):
        K = vmap(vmap(partial(self.K, kernel_params=kernel_params), (None, 0)), (0, None))(zs, zps)
        if jitter is not None:
            K += jitter * jnp.eye(len(zs))
        return K
    
    def __call__(self, x1, x2, kernel_params):
        return self.K(x1, x2, kernel_params)

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
        log_tau = kernel_params["log_tau"]
        pi_x1 = self.construct_partition(x1, W, log_tau)
        pi_x2 = self.construct_partition(x2, W, log_tau)
        linear_kernels = jnp.array([self.linear_kernel.K(x1, x2, param) for param in linear_params]) # (num_states,)
        return (pi_x1 * pi_x2 * linear_kernels).sum()


class TimeDepKernel(ABC):
    """
    Base class for a kernel which is a function of both the latent state x as well as time
    """
    def __init__(self, quadrature): 
        """
        quadrature: a quadrature object for the latent dimensions 
        """
        self.quadrature = quadrature
        
    # --------- Expectations computed with quadrature ----------
    # NOTE: Now, kernel expectation is also a function of time, although time 
    # does not affect quadrature (since the integral is only taken wrt x)
    def E_Kxx(self, t, m, S, kernel_params):
        """Computes E[k(x,x)] wrt q(x) = N(x|m,S)."""
        t = jnp.expand_dims(t, -1)
        fn = lambda x: self.K(jnp.concatenate([x, t], axis=-1), jnp.concatenate([x, t], axis=-1), 
                              kernel_params)
        return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas) # scalar

    def E_Kxz(self, t, z, m, S, kernel_params):
        """Computes E[k(x,z)] wrt q(x) = N(x|m,S)."""
        t = jnp.expand_dims(t, -1)
        fn = lambda x: self.K(jnp.concatenate([x, t], axis=-1), z, kernel_params)
        return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas) # scalar

    def E_KzxKxz(self, t, z1, z2, m, S, kernel_params):
        """Computes E[k(z1,x)k(x,z2)] wrt q(x) = N(x|m,S)."""
        t = jnp.expand_dims(t, -1)
        fn = lambda x: self.K(z1, jnp.concatenate([x, t], axis=-1), kernel_params) * self.K(jnp.concatenate([x, t], axis=-1), z2, kernel_params)
        return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas) # scalar

    def E_dKzxdx(self, t, z, m, S, kernel_params):
        """Computes E[dk(z,x)/dx] wrt q(x) = N(x|m,S)."""
        # Derivative is not taken wrt time
        t = jnp.expand_dims(t, -1)
        fn = grad(lambda x1: self.K(jnp.concatenate([x1, t]), z, kernel_params=kernel_params))
        return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas) # (D, )
    
    @abstractmethod  
    def K(self, w1, w2, kernel_params):
        """
        Each of w1, w2 is a (D + 1)-dimensional vector representing a space-time coordinate
        """
        return NotImplementedError
    
    @abstractmethod
    def make_gram(self, kernel_params, zs, zps, jitter=1e-8):
        """
        Make the gram matrix corresponding to zs and zps
        """
        return NotImplementedError
    
    def __call__(self, x1, x2, kernel_params):
        return self.K(x1, x2, kernel_params)
    

class TimeDepSSL(TimeDepKernel):
    """
    A time-dependent smoothly switching linear kernel
    The boundaries between linear regimes may also depend on time (see Hu et. al 2024) 
    """

    def __init__(self, quadrature, linear_kernel, basis_set=None):
        super().__init__(quadrature)
        self.linear_kernel = linear_kernel
        self.basis_set = basis_set

    def construct_partition(self, x, W, log_tau):
        return SSL.construct_partition(self, x, W, log_tau)

    def K(self, w1, w2, kernel_params): 

        # Extract state from combined state
        x1, x2 = w1[..., 0:-1], w2[..., 0:-1]
        linear_params = kernel_params["linear_params"] # list of linear params, one dict per regime
        W = kernel_params["W"]
        log_tau = kernel_params["log_tau"]
        # Partition is a function of both latent state and time
        pi_w1 = self.construct_partition(w1, W, log_tau)
        pi_w2 = self.construct_partition(w2, W, log_tau)
        # Linear kernel(s) are only evaluated at spatial points
        linear_kernels = jnp.array([self.linear_kernel.K(x1, x2, param) for param in linear_params]) # (num_states,)
        return (pi_w1 * pi_w2 * linear_kernels).sum()
    
    def make_gram(self, kernel_params, zs, zps, jitter=1e-8):
        K = vmap(vmap(partial(self.K, kernel_params=kernel_params), (None, 0)), (0, None))(zs, zps)
        if jitter is not None:
            K += jitter * jnp.eye(len(zs))
        return K

class ProductKernel(TimeDepKernel):
    """
    Instantiates a time-dependent kernel which factorizes as the Kronecker product of a spatial kernel and time-dependent kernel
    """

    def __init__(self, 
                quadrature, 
                spatial_kernel: Kernel,
                temporal_kernel: TimeDepKernel,
                ):
        
        super().__init__(quadrature)
        self.spatial_kernel = spatial_kernel
        self.temporal_kernel = temporal_kernel

    def K(self, w1, w2, kernel_params):
        """
        Evaluate the kernel as the product of the spatial and temporal kernels
        """
        x1, x2 = w1[..., 0:-1], w2[..., 0:-1]
        t1, t2 = w1[..., -1], w2[..., -1]
        return self.spatial_kernel.K(x1, x2, kernel_params['spatial_ker_params']) * self.temporal_kernel.K(t1, t2, kernel_params['temporal_ker_params'])

    # TODO: integrate instantiation of gram matrix with existing code
    def make_gram(self, kernel_params, zs, zps, jitter=1e-8):
        """
        Computes the gram matrix of the product kernel as the Kronecker product of the gram matrices corresponding 
        to the spatial and temporal kernels
        """
        raise NotImplementedError