import numpy as np
import matplotlib.pyplot as plt
from hp_selection.utils import compute_alpha_max, solve_irmxne_problem
from mne.inverse_sparse.mxne_inverse import _prepare_gain, _reapply_source_weighting, _make_sparse_stc
import mne
from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates
from hp_selection.ll_warm_start import LLForReweightedMTL


def load_data(condition):
    data_path = sample.data_path()
    fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
    ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
    cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'

    noise_cov = mne.read_cov(cov_fname)
    evoked = mne.read_evokeds(ave_fname, condition=condition,
                              baseline=(None, 0))
    evoked.crop(tmin=0.05, tmax=0.15)

    evoked = evoked.pick_types(eeg=False, meg=True)
    forward = mne.read_forward_solution(fwd_fname)
    return evoked, forward, noise_cov

# random_state = 0
# corr = 0.99
# n_samples = 30
# n_features = 90
# n_tasks = 15
# nnz = 2
# snr = 2

grid_length = 15

tol = 1e-8
loose = 0.9
n_orient = 1 if loose == 0 else 3

# rng = np.random.RandomState(random_state)
# sigma = np.sqrt(1 - corr ** 2)
# U = rng.randn(n_samples)

# X = np.empty([n_samples, n_features], order="F")
# X[:, 0] = U
# for j in range(1, n_features):
#     U *= corr
#     U += sigma * rng.randn(n_samples)
#     X[:, j] = U

# support = rng.choice(n_features, nnz, replace=False)
# W = np.zeros((n_features, n_tasks))

# for k in support:
#     W[k, :] = rng.normal(size=(n_tasks))

# Y = np.dot(X, W)

# noise = rng.randn(n_samples, n_tasks)
# sigma = 1 / norm(noise) * norm(Y) / snr

# Y += sigma * noise

evoked, forward, noise_cov = load_data("Left Auditory")

all_ch_names = evoked.ch_names

# Handle depth weighting and whitening (here is no weights)
forward, gain, gain_info, whitener, source_weighting, _ = _prepare_gain(
    forward, evoked.info, noise_cov, pca=False, depth=0.9,
    loose=loose, weights=None, weights_min=None, rank=None)

# Select channels of interest
sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
M = evoked.data[sel]

# Spatial whitening
Y = np.dot(whitener, M)
X = gain

alpha_max = compute_alpha_max(X, Y, n_orient) / X.shape[0]
grid = np.geomspace(alpha_max, alpha_max * 0.1, grid_length)

criterion = LLForReweightedMTL(1, grid, 5, n_orient=n_orient, random_state=0)
best_alpha = criterion.get_val(X, Y)[1]

plt.figure()
plt.semilogx(grid / np.max(grid), criterion.ll_path_, label="LL",
             marker="x", markeredgecolor="red")
plt.semilogx(grid / np.max(grid), criterion.trace_path_, label="Trace",
             marker="o", markeredgecolor="green")
plt.semilogx(grid / np.max(grid), criterion.log_det_path_, label="Log det",
             marker="x", markeredgecolor="blue")
plt.axvline(best_alpha / np.max(grid), linestyle="--", linewidth=2,
            label="best $\lambda$", c="r")
plt.legend()
plt.xlabel("$\lambda / \lambda_{max}$")
plt.ylabel("LL")
plt.show()

# Refitting
best_X, best_as = solve_irmxne_problem(X, Y, best_alpha * X.shape[0], n_orient,
                                       n_mxne_iter=5)

best_X = _reapply_source_weighting(best_X, source_weighting, best_as)

stc = _make_sparse_stc(best_X, best_as, forward, tmin=evoked.times[0],
                       tstep=1. / evoked.info['sfreq'])
plot_sparse_source_estimates(forward["src"], stc, bgcolor=(1, 1, 1), opacity=0.1)
