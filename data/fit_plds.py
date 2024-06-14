import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from ssm import LDS
import pickle

data_filename = "synthetic_data.pkl"
with open(data_filename, 'rb') as f:
    dataset = pickle.load(f)
dt, xs, spikes, C, d = dataset

K = xs.shape[-1]
D = spikes.shape[-1]

def bin_spikes_per_trial(sp_trial, bin_size, dt):
    """Bin spikes into spike counts"""
    num_bin_size = int(bin_size / dt)
    sp_trial_reshaped = sp_trial.T.reshape((D, -1, num_bin_size))
    sp_trial_reshaped_binned = sp_trial_reshaped.sum(-1) # (D, num_rslds_bins)
    return sp_trial_reshaped_binned.T
    
bin_size = 0.01
spikes_binned = [bin_spikes_per_trial(spikes_trial, bin_size=bin_size, dt=dt) for spikes_trial in spikes]
spikes_binned = [np.array(spikes_binned_trial).astype(int) for spikes_binned_trial in spikes_binned]

np.random.seed(8)
plds = LDS(D, K, emissions="poisson", emission_kwargs=dict(link="softplus", bin_size=bin_size))
plds.dynamics.A = np.eye(K)
plds.dynamics.b = np.zeros(K)
plds.dynamics.Sigmas = 0.1 * np.eye(K)[None]

elbos, q = plds.fit(spikes_binned, num_iters=30, initialize=False)
C_plds = plds.emissions.Cs[0]
d_plds = plds.emissions.ds[0]

save_filename = "synthetic_plds_emissions.pkl"
results = [elbos, q, C_plds, d_plds]
with open(save_filename, 'wb') as f:
    pickle.dump(results, f)

