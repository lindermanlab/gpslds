import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

class GaussHermiteQuadrature:
    def __init__(self, K, n_quad=10):
        self.weights, self.unit_sigmas = self.compute_weights_and_sigmas(K, n_quad)

    def compute_weights_and_sigmas(self, K, n_quad):
        """
        Computes weights and sigma-points for Gauss-Hermite quadrature.
    
        Parameters
        ------------
        K: scalar, state dimension
        n_quad: scalar, number of quadrature points per dimension
    
        Returns
        ------------
        weights: (n_quad**K, ) weights
        unit sigmas: (n_quad**K, K) unit sigma-points (i.e. relative to N(0,1))
        """
        samples_1d, weights_1d = jnp.array(hermegauss(n_quad))
        weights_1d /= weights_1d.sum()
        weights_rep = [weights_1d for _ in range(K)]
        samples_rep = [samples_1d for _ in range(K)]
        weights = jnp.stack(jnp.meshgrid(*weights_rep), axis=-1).reshape(-1, K).prod(axis=1)
        unit_sigmas = jnp.stack(jnp.meshgrid(*samples_rep), axis=-1).reshape(-1, K)
        return weights, unit_sigmas
         
            