import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import vmap, lax
from jax.nn import softplus
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from functools import partial
from utils import compute_sigmas, gaussian_int
from em import sgd

class Likelihood:
    def __init__(self, ys_binned, t_mask):
        """Initialize Likelihood class"""
        self.ys_binned = ys_binned
        self.t_mask = t_mask

    def ell_over_time(self, ys, t_mask, ms, Ss, output_params):
        """Compute expected log-likelihood over one trial"""
        raise NotImplementedError()

    def ell_over_trials(self, batch_inds, ms, Ss, output_params):
        """Compute expected log-likelihood over all trials in a batch"""
        return vmap(partial(self.ell_over_time, output_params=output_params))(self.ys_binned[batch_inds], self.t_mask[batch_inds], ms, Ss).sum()

    def ell_over_all_trials(self, ms, Ss, output_params):
        """Compute expected log-likelihood over all trials"""
        return vmap(partial(self.ell_over_time, output_params=output_params))(self.ys_binned, self.t_mask, ms, Ss).sum()

    def update_output_params(self, ms, Ss, output_params, loss_fn, n_iters_m=200, learning_rate=.08):
        """By default, learn output parameters with SGD (except for Gaussian case, where there are closed-form updates, see below)"""
        output_params, _ = sgd(loss_fn, output_params, n_iters_m=n_iters_m, learning_rate=learning_rate)
        return output_params

class Gaussian(Likelihood):
    def __init__(self, ys_binned, t_mask):
        """
        Initialize Gaussian likelihood class, y ~ N(Cx + d, R).

        Parameters:
        --------------
        ys_binned: (n_trials, n_timesteps, D) Gaussian data
        t_mask: (n_trials, n_timesteps) binary mask indicating data which is observed
        """
        super().__init__(ys_binned, t_mask)
        self.n_trials, self.T, self.D = ys_binned.shape
        self.n_total_obs = t_mask.sum()
    
    def ell(self, y, m, S, output_params):
        """Compute expectation wrt q(x) = N(x|m,S) of Gaussian log-likelihood at an observation y."""
        C, d, R = output_params['C'], output_params['d'], output_params['R']
        ll = tfd.Normal(C @ m + d, jnp.sqrt(R)).log_prob(y).sum()
        correction = -0.5 * jnp.trace(S @ C.T @ (C / R[:, None]))
        return ll + correction

    def ell_over_time(self, ys, t_mask, ms, Ss, output_params):
        """Compute expected Gaussian log-likelihood over one trial."""
        ell_on_grid = vmap(partial(self.ell, output_params=output_params))(ys, ms, Ss) # vmap across time
        return jnp.where(t_mask != 0, ell_on_grid, 0).sum()

    def update_output_params(self, ms, Ss, output_params, loss_fn, n_iters_m=None, learning_rate=None):
        """
        Perform closed-form updates for output mapping parameters for Gaussian likelihood model.

        Parameters:
        ------------
        ms: (n_trials, T, K) mean across trials and time points
        Ss: (n_trials, T, K, K) covariance across trials and time points
        output_params: dict containing C, d, and R
        loss_fn: empty argument

        Returns:
        -----------
        output_params: updated dict containing new values of C, d, and R
        """
        K = ms.shape[-1]
        C, d, R = output_params['C'], output_params['d'], output_params['R']

        # stack things across trials
        ys_binned_stacked = self.ys_binned.reshape(-1, self.D)
        t_mask_stacked = self.t_mask.reshape(-1)
        ms_stacked, Ss_stacked = ms.reshape(-1, K), Ss.reshape(-1, K, K)
    
        # update for C
        C_term1_on_grid = vmap(jnp.outer)(ys_binned_stacked - d, ms_stacked) # (-1, D, K)
        C_term1 = (t_mask_stacked[:,None,None] * C_term1_on_grid).sum(0)
        C_term2_on_grid = Ss_stacked + vmap(jnp.outer)(ms_stacked, ms_stacked) # (-1, K, K)
        C_term2 = (t_mask_stacked[:,None,None] * C_term2_on_grid).sum(0)
        C = jnp.linalg.solve(C_term2, C_term1.T).T # (D, K)
    
        # update for d
        d_term1_on_grid = ys_binned_stacked - (C @ ms_stacked[...,None]).squeeze(-1) # (-1, D)
        d_term1 = (t_mask_stacked[:,None] * d_term1_on_grid).sum(0) # (D, )
        d = 1. / self.n_total_obs * d_term1 # (D, )
    
        # update for R
        all_mus = (C @ ms_stacked[...,None]).squeeze(-1) + d # (-1, D)
        all_vars = vmap(jnp.diag)(C @ Ss_stacked @ C.T) # (-1, D)
        R_term1 = (t_mask_stacked[:,None] * ys_binned_stacked**2).sum(0) # (D, )
        R_term2 = -2 * (t_mask_stacked[:,None] * ys_binned_stacked * all_mus).sum(0) # (D, )
        R_term3 = (t_mask_stacked[:,None] * (all_vars + all_mus**2)).sum(0) # (D, )
        R = 1. / self.n_total_obs * (R_term1 + R_term2 + R_term3)

        output_params = {'C': C, 'd': d, 'R': R}
        return output_params

class Poisson(Likelihood):
    def __init__(self, ys_binned, t_mask, dt, quadrature, link='softplus'):
        """
        Initialize Poisson likelihood class, y ~ Pois(g(Cx + d)*dt)

        Parameters:
        --------------
        ys_binned: (n_trials, n_timesteps, D) Poisson data
        t_mask: (n_trials, n_timesteps) binary mask indicating data which is observed
        dt: discretization time-step of data
        quadrature: Gauss-Hermite quadrature object
        link: {'exp', 'softplus'} link function g()
        """
        super().__init__(ys_binned, t_mask)
        self.dt = dt
        self.quadrature = quadrature
        self.link = link

    def ell(self, y, m, S, output_params):
        """Compute expectation wrt q(x) = N(x|m,S) of Poisson log-likelihood at an observation y."""
        C, d = output_params['C'], output_params['d']
        if self.link == 'exp':
            cov_term = 0.5 * jnp.diag(C @ S @ C.T)
            log_rate = C @ m + d + cov_term + jnp.log(self.dt)
            ll = tfd.Poisson(log_rate=log_rate).log_prob(y).sum() # sum over neurons -> scalar
            correction = (-y * cov_term).sum()
            return ll + correction
        elif self.link == 'softplus':
            fn = lambda x: tfd.Poisson(rate=self.dt * softplus(C @ x + d)).log_prob(y).sum()
            # return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas)
            # use Gauss-Hermite quadrature
            sigmas = compute_sigmas(self.quadrature, m, S)
            def _step(carry, arg):
                weight, sigma = arg
                return carry + weight * fn(sigma), None
            result, _ = lax.scan(_step, 0., (self.quadrature.weights, sigmas))
            return result # scalar
        else:
            raise NotImplementedError()

    def ell_over_time(self, ys, t_mask, ms, Ss, output_params):
        """Compute expected Poisson log-likelihood over one trial."""
        ell_on_grid = vmap(partial(self.ell, output_params=output_params))(ys, ms, Ss) # vmap across time
        return jnp.where(t_mask != 0, ell_on_grid, 0).sum()

class PoissonProcess(Likelihood):
    def __init__(self, ys_binned, t_mask, dt, quadrature, link='softplus'):
        """
        Initialize Poisson process likelihood class, t_i|x ~ PP(g(Cx+d)).

        Parameters:
        --------------
        ys_binned: (n_trials, n_timesteps, D) binary array representing event times
        t_mask: (n_trials, n_timesteps) binary mask indicating when trial is observed (this is the same as 'trial_mask' in em.py)
        dt: integration timestep of model
        quadrature: Gauss-Hermite quadrature object
        link: {'exp', 'softplus'} link function g()
        """
        super().__init__(ys_binned, t_mask)
        self.dt = dt
        self.quadrature = quadrature
        self.link = link

    def ell_cont(self, m, S, output_params):
        """Compute expectation wrt q(x) = N(x|m,S) of continuous-time part of Poisson process log-likelhood."""
        C, d = output_params['C'], output_params['d']
        if self.link == 'exp':
            return -jnp.exp(C @ m + d + 0.5 * jnp.diag(C @ S @ C.T)) 
        elif self.link == 'softplus':
            fn = lambda x: -softplus(C @ x + d)
            # return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas) 
            # use Gauss-Hermite quadrature
            sigmas = compute_sigmas(self.quadrature, m, S)
            def _step(carry, arg):
                weight, sigma = arg
                return carry + weight * fn(sigma), None
            result, _ = lax.scan(_step, jnp.zeros(C.shape[0]), (self.quadrature.weights, sigmas))
            return result # (D,)
        else:
            raise NotImplementedError()

    def ell_jump(self, m, S, output_params):
        """Compute expectation wrt q(x) = N(x|m,S) of observation-driven part of Poisson process log-likelhood."""
        C, d = output_params['C'], output_params['d']
        if self.link == 'exp':
            return C @ m + d 
        elif self.link == 'softplus':
            fn = lambda x: jnp.log(softplus(C @ x + d))
            # return gaussian_int(fn, m, S, self.quadrature.weights, self.quadrature.unit_sigmas) 
            # use Gauss-Hermite quadrature
            sigmas = compute_sigmas(self.quadrature, m, S)
            def _step(carry, arg):
                weight, sigma = arg
                return carry + weight * fn(sigma), None
            result, _ = lax.scan(_step, jnp.zeros(C.shape[0]), (self.quadrature.weights, sigmas))
            return result # (D,)
        else:
            raise NotImplementedError()

    def ell_over_time(self, ys, t_mask, ms, Ss, output_params):
        """Computed expected Poisson process log-likelihood over one trial."""
        ell_cont_on_grid = vmap(partial(self.ell_cont, output_params=output_params))(ms, Ss) # (T, K)
        ell_cont = self.dt * (t_mask[:,None] * ell_cont_on_grid).sum()
        ell_jump_on_grid = vmap(partial(self.ell_jump, output_params=output_params))(ms, Ss) # (T, K)
        t_obs_mask = (ys > 0) * t_mask[:,None] # (T, K)
        ell_jump = (t_obs_mask * ell_jump_on_grid).sum()
        ell_term = ell_cont + ell_jump
        return ell_term