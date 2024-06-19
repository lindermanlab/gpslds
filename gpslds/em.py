import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from jax import lax, jit, grad, vmap
import numpy as np
from functools import partial
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from utils import make_gram
from initialization import initialize_vem
import optax
import wandb

# --------- ELBO FUNCTIONS ----------

def kl(fn, m, S, A, b, input, B, q_u_mu, q_u_sigma, kernel, kernel_params):
    """
    Compute the integrand of expected KL[q(x)||p(x)] at a given timepoint.

    Parameters
    ------------
    m: (K,) mean vector
    S: (K, K) covariance matrix
    A: (K, K) transition matrix
    b: (K,) bias vector
    input: (I,) input vector
    B: (K, I) input dynamics matrix

    Returns
    ------------
    kl: scalar, integrand of expected KL[q(x)||p(x)]
    """
    # compute autonomous KL part
    kl = fn.ff(m, S, q_u_mu, q_u_sigma, kernel_params)
    kl += 2 * jnp.trace(A.T @ fn.dfdx(m, S, q_u_mu, q_u_sigma, kernel_params) @ S)
    kl += jnp.trace(A.T @ A @ (S + jnp.outer(m, m)))
    kl += 2 * jnp.dot(m, A.T @ fn.f(m, S, q_u_mu, q_u_sigma, kernel_params))
    kl += jnp.dot(b, b - 2 * fn.f(m, S, q_u_mu, q_u_sigma, kernel_params) - 2 * A @ m)
    
    # compute input-dependent KL part
    kl += 2 * jnp.dot(B @ input, fn.f(m, S, q_u_mu, q_u_sigma, kernel_params) + A @ m - b)
    kl += jnp.dot(B @ input, B @ input)
    
    kl = 0.5 * kl
    return kl

def kl_over_time(dt, fn, trial_mask, ms, Ss, As, bs, inputs, B, q_u_mu, q_u_sigma, kernel, kernel_params):
    """Compute expected KL[q(x)||p(x)] (an integral over time) for a single trial."""
    kl_on_grid = vmap(partial(kl, fn, B=B, q_u_mu=q_u_mu, q_u_sigma=q_u_sigma, kernel=kernel, kernel_params=kernel_params))(ms, Ss, As, bs, inputs)
    kl_term = dt * (kl_on_grid * trial_mask).sum()
    return kl_term

def compute_elbo_per_trial(dt, fn, likelihood, ys, t_mask, trial_mask, ms, Ss, As, bs, inputs, B, q_u_mu, q_u_sigma, output_params, kernel, kernel_params):
    """Compute ELBO for a single trial."""
    ell_term = likelihood.ell_over_time(ys, t_mask, ms, Ss, output_params)
    kl_term = kl_over_time(dt, fn, trial_mask, ms, Ss, As, bs, inputs, B, q_u_mu, q_u_sigma, kernel, kernel_params)
    elbo = ell_term - kl_term + fn.prior_term(q_u_mu, q_u_sigma, kernel_params)
    return elbo

def compute_elbo(dt, fn, likelihood, batch_inds, trial_mask, ms, Ss, As, bs, inputs, B, output_params, kernel, kernel_params):
    """Compute ELBO over a batch of trials. Used to perform inference and learning over batches of data."""
    q_u_mu, q_u_sigma = update_q_u(dt, fn, trial_mask, ms, Ss, As, bs, inputs, B, kernel, kernel_params) 
    kl_term = vmap(partial(kl_over_time, dt, fn, B=B, q_u_mu=q_u_mu, q_u_sigma=q_u_sigma, kernel=kernel, kernel_params=kernel_params))(trial_mask, ms, Ss, As, bs, inputs[batch_inds]).sum()
    ell_term = likelihood.ell_over_trials(batch_inds, ms, Ss, output_params)
    prior_term = fn.prior_term(q_u_mu, q_u_sigma, kernel_params)
    elbo = ell_term - kl_term + prior_term 
    return elbo

def compute_elbo_all_trials(dt, fn, likelihood, trial_mask, ms, Ss, As, bs, inputs, B, output_params, kernel, kernel_params, q_u_mu, q_u_sigma):
    """Compute ELBO over entire dataset. Used for method evaluation at each vEM iter."""
    kl_term = vmap(partial(kl_over_time, dt, fn, B=B, q_u_mu=q_u_mu, q_u_sigma=q_u_sigma, kernel=kernel, kernel_params=kernel_params))(trial_mask, ms, Ss, As, bs, inputs).sum()
    ell_term = likelihood.ell_over_all_trials(ms, Ss, output_params)
    prior_term = fn.prior_term(q_u_mu, q_u_sigma, kernel_params)
    elbo = ell_term - kl_term + prior_term 
    return elbo

# ----------- E-STEP -------------
def forward_pass(dt, As, bs, m0, S0):
    """
    Perform one forward pass to update the variational mean and cov of q(x) at each time step.

    Parameters
    ----------
    dt: integration timestep
    As: (T, K, K) transition matrix for each time step
    bs: (T, K) bias vector for each time step
    m0: (K,) initial mean vector
    S0: (K, K) initial covariance matrix

    Returns
    -------
    ms: (T, K) mean vector at each time step
    Ss: (T, K, K) covariance matrix at each time step
    """
    K = bs.shape[-1]

    def _step(carry, arg):
        m_prev, S_prev = carry
        A, b = arg
        m = m_prev - dt * (A @ m_prev - b)
        S = S_prev - dt * (A @ S_prev + S_prev @ A.T - dt * A @ S_prev @ A.T - jnp.eye(K)) # adding O(dt^2) correction term for accuracy
        S = 0.5 * (S + S.T)
        return (m, S), (m_prev, S_prev)

    initial_carry = (m0, S0)
    _, (ms, Ss) = lax.scan(_step, initial_carry, (As, bs))
    
    return ms, Ss

def backward_pass(dt, As, bs, ms, Ss, elbo_fn):
    """
    Perform one backward pass to update the Lagrange multipliers at each time step.

    Parameters
    ----------
    dt: timestep
    As: (T, K, K) transition matrix for each time step
    bs: (T, K) bias vector for each time step
    ms: (T, K) variational mean from forward pass
    Ss: (T, K, K) variational cov from forward pass
    elbo_fn: function only of ms, Ss, As, and bs

    Returns
    -------
    lmbdas: (T, K) Lagrange multiplier for mean constraint at each time step
    Psis: (T, K, K) Lagrange multiplier for cov constraint at each time step
    """
    K = bs.shape[-1]
    lmbdaT = jnp.zeros(K)
    PsiT = jnp.zeros((K, K))
    P = 0.5 * jnp.ones((K, K)) + 0.5 * jnp.eye(K)

    dLdms, dLdSs = grad(elbo_fn, argnums=(0, 1))(ms, Ss, As, bs) 
    dLdSs = 0.5 * (dLdSs + dLdSs.transpose(0,2,1))
    
    def _step(carry, arg):
        lmbda_next, Psi_next = carry
        A, b, dLdm, dLdS = arg
        lmbda = lmbda_next - dt * A.T @ lmbda_next - dLdm
        Psi = Psi_next - dt * (A.T @ Psi_next + Psi_next @ A) - dLdS * P 
        Psi = 0.5 * (Psi + Psi.T)
        return (lmbda, Psi), (lmbda_next, Psi_next)

    initial_carry = (lmbdaT, PsiT)
    args = (As, bs, dLdms, dLdSs)
    _, (lmbdas, Psis) = lax.scan(_step, initial_carry, args, reverse=True)

    return lmbdas, Psis

def variational_step(dt, fn, likelihood, ys, t_mask, trial_mask, As, bs, m0, S0, inputs, B, q_u_mu, q_u_sigma, output_params, kernel, kernel_params):
    """Perform a single forward and backward pass in the E-step to update variational parameters."""
    # redefine elbo
    elbo_fn = lambda ms, Ss, As, bs: compute_elbo_per_trial(dt, fn, likelihood, ys, t_mask, trial_mask, ms, Ss, As, bs, inputs, B, q_u_mu, q_u_sigma, output_params, kernel, kernel_params)

    # forward/backward solve ODEs
    ms, Ss = forward_pass(dt, As, bs, m0, S0)
    lmbdas, Psis = backward_pass(dt, As, bs, ms, Ss, elbo_fn)

    # update variational parameters
    As = -vmap(partial(fn.dfdx, q_u_mu=q_u_mu, q_u_sigma=q_u_sigma, kernel_params=kernel_params))(ms, Ss) + 2 * Psis 
    bs = vmap(partial(fn.f, q_u_mu=q_u_mu, q_u_sigma=q_u_sigma, kernel_params=kernel_params))(ms, Ss) + (As @ ms[...,None]).squeeze(-1) + (B[None] @ inputs[...,None]).squeeze(-1) - lmbdas 

    # mask out As and bs after trial end (helps with stability)
    As = As * trial_mask[...,None,None]
    bs = bs * trial_mask[...,None]

    return ms, Ss, lmbdas, Psis, As, bs

def update_init_variational_params(mu0, V0, lmbda0, Psi0):
    """Perform closed-form updates for variational posterior q(x0) = N(x0|m0, S0)."""
    D = mu0.shape[0]
    m0 = mu0 - V0 @ lmbda0
    S0 = jnp.linalg.solve((2 * Psi0 + jnp.linalg.solve(V0, jnp.eye(D))), jnp.eye(D))

    return m0, S0

def e_step(dt, fn, likelihood, batch_inds, trial_mask, As, bs, m0, S0, inputs, B, q_u_mu, q_u_sigma, output_params, kernel, kernel_params, n_iters_e):
    """Perform a single E-step over a batch of trials"""    
    batch_variational_step = vmap(partial(variational_step, dt, fn, likelihood, B=B, q_u_mu=q_u_mu, q_u_sigma=q_u_sigma, output_params=output_params, kernel=kernel, kernel_params=kernel_params))

    def _step(carry, arg):
        As_prev, bs_prev = carry
        ms, Ss, lmbdas, Psis, As, bs = batch_variational_step(likelihood.ys_binned[batch_inds], likelihood.t_mask[batch_inds], trial_mask, As_prev, bs_prev, m0, S0, inputs) 
        return (As, bs), (ms, Ss, lmbdas, Psis, As, bs)

    initial_carry = (As, bs)
    # _, (ms, Ss, lmbdas, Psis, As, bs, elbo_vals) = lax.scan(_step, initial_carry, jnp.arange(n_iters_e))
    _, (ms, Ss, lmbdas, Psis, As, bs) = lax.scan(_step, initial_carry, jnp.arange(n_iters_e))
    # print("E-step elbos: ", elbo_vals)

    return ms[-1], Ss[-1], lmbdas[-1], Psis[-1], As[-1], bs[-1]

# ------------------ M-STEP --------------------
    
def update_q_u(dt, fn, trial_mask, ms, Ss, As, bs, inputs, B, kernel, kernel_params):
    """Perform closed-form updates for variational parameters of inducing points."""
    
    # Define helper functions on single-trial level
    def _q_u_sigma_int(dt, fn, trial_mask, ms, Ss, kernel, kernel_params):
        E_KzxKxz_over_zs = vmap(vmap(partial(kernel.E_KzxKxz, kernel_params=kernel_params), (None, 0, None, None)), (0, None, None, None))
        E_KzxKxz_on_grid = vmap(E_KzxKxz_over_zs, (None, None, 0, 0))(fn.zs, fn.zs, ms, Ss) # (T, M, M)
        int_E_KzxKxz = dt * (E_KzxKxz_on_grid * trial_mask[...,None,None]).sum(0)
        return int_E_KzxKxz

    def _q_u_mu_int1(dt, fn, trial_mask, ms, Ss, As, bs, inputs, B, kernel, kernel_params):
        E_Kxz_over_zs = vmap(partial(kernel.E_Kxz, kernel_params=kernel_params), (0, None, None))
        Psi1 = vmap(E_Kxz_over_zs, (None, 0, 0))(fn.zs, ms, Ss) # (T, M)
        f_q = (-As @ ms[...,None]).squeeze(-1) + bs # (T, D)
        input_correction = (B[None] @ inputs[...,None]).squeeze(-1) # (T, D)
        integrand_on_grid = vmap(jnp.outer)(Psi1, f_q - input_correction) # (T, M, D)
        int1 = dt * (integrand_on_grid * trial_mask[...,None,None]).sum(0) # (M, D)
        return int1

    def _q_u_mu_int2(dt, fn, trial_mask, ms, Ss, As, kernel, kernel_params):
        E_dKzxdx_over_zs = vmap(partial(kernel.E_dKzxdx, kernel_params=kernel_params), (0, None, None))
        Psid1 = vmap(E_dKzxdx_over_zs, (None, 0, 0))(fn.zs, ms, Ss) # (T, M, D)
        integrand_on_grid = Psid1 @ Ss @ As.transpose((0, 2, 1)) # (T, M, D)
        int2 = dt * (integrand_on_grid * trial_mask[...,None,None]).sum(0) # (M, D)
        return int2

    # Perform updates
    Kzz = make_gram(kernel.K, kernel_params, fn.zs, fn.zs, jitter=fn.jitter)
    int_E_KzxKxz = vmap(partial(_q_u_sigma_int, dt, fn, kernel=kernel, kernel_params=kernel_params))(trial_mask, ms, Ss).sum(0) 
    q_u_sigma = (Kzz @ jnp.linalg.solve(Kzz + int_E_KzxKxz, Kzz))[None].repeat(ms.shape[-1], 0) # (D, M, M)

    int1 = vmap(partial(_q_u_mu_int1, dt, fn, B=B, kernel=kernel, kernel_params=kernel_params))(trial_mask, ms, Ss, As, bs, inputs).sum(0) 
    int2 = vmap(partial(_q_u_mu_int2, dt, fn, kernel=kernel, kernel_params=kernel_params))(trial_mask, ms, Ss, As).sum(0) 
    q_u_mu = (Kzz @ jnp.linalg.solve(Kzz + int_E_KzxKxz, int1 - int2)).T 

    return q_u_mu, q_u_sigma

def update_init_params(m0, S0):
    """Performs trial-specific updates for prior p(x0) = N(x0|mu0, V0)."""
    mu0, V0 = m0, S0
    return mu0, V0

def update_B(dt, fn, ms, Ss, As, bs, inputs, q_u_mu, q_u_sigma, kernel, kernel_params, jitter=1e-4):
    """Computes closed-form update for input effect matrix B."""
    
    def _compute_hs(fn, ms, Ss, As, bs, q_u_mu, q_u_sigma, kernel_params):
        """Computes h(t) := E[f(t)] + A(t)m(t) - b(t)."""
        E_f = vmap(partial(fn.f, q_u_mu=q_u_mu, q_u_sigma=q_u_sigma, kernel_params=kernel_params))(ms, Ss) 
        hs = E_f + (As @ ms[...,None]).squeeze(-1) - bs
        return hs 

    def _int_outer_prod_inputs(inputs, jitter):
        """Computes \int_0^T u(t) u(t)^T dt."""
        n_inputs = inputs.shape[-1]
        outer_prod = vmap(jnp.outer)(inputs, inputs) + jitter * jnp.eye(n_inputs) 
        return dt * outer_prod.sum(0) 
    
    def _int_outer_prod(hs, inputs):
        """Computes \int_0^T h(t) u(t)^T dt."""
        outer_prod = vmap(jnp.outer)(hs, inputs) 
        return dt * outer_prod.sum(0) 

    inputs_term = vmap(partial(_int_outer_prod_inputs, jitter=jitter))(inputs).sum(0) 
    hs = vmap(partial(_compute_hs, fn, q_u_mu=q_u_mu, q_u_sigma=q_u_sigma, kernel_params=kernel_params))(ms, Ss, As, bs) 
    outer_prod_term = vmap(_int_outer_prod)(hs, inputs).sum(0) 
    B = -jnp.linalg.solve(inputs_term, outer_prod_term.T).T
    return B

# ------------ MODEL FITTING FUNCTIONS ------------
    
def sgd(loss_fn, params, n_iters_m, learning_rate):
    """Performs SGD with Adam on a loss function with respect to params"""
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    def _step(carry, arg):
        params_prev, opt_state_prev = carry
        loss, grads = jax.value_and_grad(loss_fn)(params_prev)
        updates, opt_state = optimizer.update(grads, opt_state_prev, params_prev)
        params = optax.apply_updates(params_prev, updates)
        return (params, opt_state), -loss # returning elbo

    initial_carry = (params, opt_state)
    (final_params, _), all_elbos = lax.scan(_step, initial_carry, jnp.arange(n_iters_m))
    # print("M-step elbos: ", all_elbos)

    return final_params, all_elbos[-1]

def fit_variational_em(key, 
                       K, 
                       dt, 
                       fn, 
                       likelihood, 
                       trial_mask, 
                       output_params, 
                       kernel, 
                       kernel_params, 
                       inputs=None,
                       m0=None, 
                       mu0=None, 
                       n_iters=100, 
                       n_iters_e=15, 
                       n_iters_m=1, 
                       learning_rates=None, 
                       batch_size=30, 
                       wandb=None):
    """
    Run the full (stochastic) variational EM algorithm.

    Parameters:
    -----------------
    key: jr.PRNGKey (for selecting random mini-batches)
    K: latent dimension
    dt: integration timestep
    fn: class object from transition.py
    likelihood: class object from likelihoods.py
    trial_mask: (n_trials, n_timesteps) binary mask indicating when trials are active (handles varying-length trials)
    output_params: dict containing output mapping parameters
    kernel: class object from kernels.py
    kernel_params: dict containing kernel parameters
    inputs: optional (n_trials, n_timesteps, I) external inputs, default 0
    m0, mu0: optional {posterior, prior} mean for x0, default 0
    n_iters: number of vEM iters to run
    n_iters_e: number of E-step iters to run per vEM iter
    n_iters_m: number of M-step iters to run per vEM iter for learning kernel hyperparameters
    learning_rates: (n_iters, ) learning rate in each vEM iter for learning kernel hyperparameters with Adam
    batch_size: size of minibatch used in stochastic vEM
    wandb: optional wandb object for logging elbos

    Returns:
    -----------------
    ms, Ss: (n_trials, n_timesteps, K), (n_trials, n_timesteps, K, K) posterior marginals of latent states
    As, bs: (n_trials, n_timesteps, K, K), (n_trials, n_timesteps, K) variational parameters for dynamics
    B: (K, I) learned input effect matrix
    q_u_mu, q_u_sigma: (K, M), (K, M, M) variational parameters for inducing points
    output_params, kernel_params: dicts containing learned parameters
    elbos_lst: (n_iters,) list of elbos at each vEM iter
    """
    @jit
    def _step(batch_inds, m0, S0, ms, Ss, As, bs, B, q_u_mu, q_u_sigma, mu0, V0, output_params, kernel_params):
        """Runs a single E-step and M-step"""
        m0_batch, S0_batch, mu0_batch, V0_batch, As_batch, bs_batch, trial_mask_batch, inputs_batch = m0[batch_inds], S0[batch_inds], mu0[batch_inds], V0[batch_inds], As[batch_inds], bs[batch_inds], trial_mask[batch_inds], inputs[batch_inds]
        
        # run e-step
        ms_batch, Ss_batch, lmbdas, Psis, As_batch, bs_batch = e_step(dt, fn, likelihood, batch_inds, trial_mask_batch, As_batch, bs_batch, m0_batch, S0_batch, inputs_batch, B, q_u_mu, q_u_sigma, output_params, kernel, kernel_params, n_iters_e=n_iters_e)
        ms = ms.at[batch_inds].set(ms_batch)
        Ss = Ss.at[batch_inds].set(Ss_batch)
        As = As.at[batch_inds].set(As_batch)
        bs = bs.at[batch_inds].set(bs_batch)
        
        # update init condition variational params (for the next e-step)
        m0_batch, S0_batch = vmap(update_init_variational_params)(mu0_batch, V0_batch, lmbdas[:,0], Psis[:,0])
        m0 = m0.at[batch_inds].set(m0_batch)
        S0 = S0.at[batch_inds].set(S0_batch)

        # update output mapping parameters
        loss_fn_output_params = lambda output_params: -compute_elbo(dt, fn, likelihood, batch_inds, trial_mask_batch, ms_batch, Ss_batch, As_batch, bs_batch, inputs_batch, B, output_params, kernel, kernel_params)
        output_params = likelihood.update_output_params(ms_batch, Ss_batch, output_params, loss_fn_output_params)

        # learn kernel parameters
        loss_fn_kernel_params = lambda kernel_params: -compute_elbo(dt, fn, likelihood, batch_inds, trial_mask_batch, ms_batch, Ss_batch, As_batch, bs_batch, inputs_batch, B, output_params, kernel, kernel_params)
        kernel_params, _ = sgd(loss_fn_kernel_params, kernel_params, n_iters_m, learning_rates[i])

        # update input matrix B
        B = update_B(dt, fn, ms_batch, Ss_batch, As_batch, bs_batch, inputs_batch, q_u_mu, q_u_sigma, kernel, kernel_params)

        # update dynamics
        q_u_mu, q_u_sigma = update_q_u(dt, fn, trial_mask_batch, ms_batch, Ss_batch, As_batch, bs_batch, inputs_batch, B, kernel, kernel_params)

        # update init condition prior
        mu0_batch, V0_batch = update_init_params(m0_batch, S0_batch)
        mu0 = mu0.at[batch_inds].set(mu0_batch)
        V0 = V0.at[batch_inds].set(V0_batch)

        # compute ELBO on whole dataset for evaluation
        elbo_val = compute_elbo_all_trials(dt, fn, likelihood, trial_mask, ms, Ss, As, bs, inputs, B, output_params, kernel, kernel_params, q_u_mu, q_u_sigma)   

        return m0, S0, ms, Ss, lmbdas, Psis, As, bs, B, q_u_mu, q_u_sigma, mu0, V0, output_params, kernel_params, elbo_val
    
    n_trials, n_timesteps, _ = likelihood.ys_binned.shape

    # initialize parameters for variational EM
    if inputs is None:
        inputs = jnp.zeros((n_trials, n_timesteps, 1))
    if m0 is None:
        m0 = jnp.zeros((n_trials, K))
    if mu0 is None:
        mu0 = jnp.zeros((n_trials, K))
    mean_init, var_init = 0., dt * 10
    M = len(fn.zs)
    I = inputs.shape[-1]
    S0, V0, As, bs, ms, Ss, q_u_mu, q_u_sigma, B = initialize_vem(n_trials, n_timesteps, K, M, I, mean_init, var_init)

    # initialize a default learning rate schedule
    if learning_rates is None:
        learning_rates = jnp.arange(1, n_iters+1) ** (-0.5)

    # fit model
    elbos_lst = []
    for i in range(n_iters):
        key_i = jr.fold_in(key, i)
        batch_inds = jr.choice(key_i, n_trials, (batch_size,), replace=False)
        m0, S0, ms, Ss, lmbdas, Psis, As, bs, B, q_u_mu, q_u_sigma, mu0, V0, output_params, kernel_params, elbo_val = _step(batch_inds, m0, S0, ms, Ss, As, bs, B, q_u_mu, q_u_sigma, mu0, V0, output_params, kernel_params)
        
        elbos_lst.append(elbo_val)
        if wandb is not None:
            wandb.log({"elbo": elbo_val})
        print(f"iter: {i}, elbo = {elbo_val}")
        if not jnp.isfinite(elbo_val):
            break

    return ms, Ss, As, bs, B, q_u_mu, q_u_sigma, output_params, kernel_params, elbos_lst

