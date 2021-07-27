import functools, os
import joblib
from pathlib import Path

import numpy as np

import mne
from mne.datasets import sample, somato
from mne.inverse_sparse.mxne_inverse import (_prepare_gain, is_fixed_orient,
                                             _reapply_source_weighting,
                                             _make_sparse_stc)
from mne.inverse_sparse.mxne_optim import iterative_mixed_norm_solver


def groups_norm2(A, n_orient=1):
    """Compute squared L2 norms of groups inplace."""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def sum_squared(X):
    X_flat = X.ravel(order="F" if np.isfortran(X) else "C")
    return np.dot(X_flat, X_flat)


def norm_l2_05(X, n_orient, copy=True):
    """Compute l_2,p norm"""
    if X.size == 0:
        return 0.0
    if copy:
        X = X.copy()
    return np.sum(np.sqrt(np.sqrt(groups_norm2(X, n_orient))))


def norm_l2_inf(X, n_orient, copy=True):
    """Compute l_2,inf norm"""
    if X.size == 0:
        return 0.0
    if copy:
        X = X.copy()
    return np.sqrt(np.max(groups_norm2(X, n_orient)))


def compute_alpha_max(G, M, n_orient):
    """Compute alpha max"""
    return norm_l2_inf(np.dot(G.T, M), n_orient, copy=True)


def apply_solver(solver, evoked, forward, noise_cov, depth=0.9, loose=0.9,
                 n_mxne_iter=5, random_state=0):
    """
    Preprocess M/EEG data and apply solver to the preprocess data.
    """
    all_ch_names = evoked.ch_names

    # Handle depth weighting and whitening (here is no weights)
    forward, gain, gain_info, whitener, source_weighting, _ = _prepare_gain(
        forward, evoked.info, noise_cov, pca=False, depth=depth,
        loose=loose, weights=None, weights_min=None, rank=None)

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # Spatial whitening
    M = np.dot(whitener, M)
    n_orient = 1 if is_fixed_orient(forward) else 3

    X, active_set = solver(gain, M, n_orient, n_mxne_iter=n_mxne_iter,
                           random_state=random_state)
    X = _reapply_source_weighting(X, source_weighting, active_set)

    stc = _make_sparse_stc(X, active_set, forward, tmin=evoked.times[0],
                           tstep=1. / evoked.info['sfreq'])
    return stc


def build_full_coefficient_matrix(active_set, n_times, coef):
    """Building full coefficient matrix and filling active set with
    non-zero coefficients"""
    final_coef_ = np.zeros((len(active_set), n_times))
    if coef is not None:
        final_coef_[active_set] = coef
    return final_coef_


def solve_irmxne_problem(G, M, alpha, n_orient, n_mxne_iter=5, tol=1e-8):
    X, active_set, _ = iterative_mixed_norm_solver(M, G, alpha, n_mxne_iter,
                                                   n_orient=n_orient,
                                                   debias=False, tol=tol)
    return X, active_set


@functools.lru_cache(None)
def get_dgemm():
    from scipy import linalg

    return linalg.get_blas_funcs("gemm", (np.empty(0, np.float64),))


def norm_l2_1(X, n_orient, copy=True):
    if X.size == 0:
        return 0.0
    if copy:
        X = X.copy()
    return np.sum(np.sqrt(groups_norm2(X, n_orient)))


def primal_mtl(X, Y, coef, active_set, alpha, n_orient=1):
    """Primal objective function for multi-task
    LASSO
    """
    Y_hat = np.dot(X[:, active_set], coef)
    R = Y - Y_hat
    penalty = norm_l2_1(coef, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty
    return p_obj


def get_duality_gap_mtl(X, Y, coef, active_set, alpha, n_orient=1):
    Y_hat = np.dot(X[:, active_set], coef)
    R = Y - Y_hat
    penalty = norm_l2_1(coef, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty

    dual_norm = norm_l2_inf(np.dot(X.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(
        R * Y_hat
    )
    gap = p_obj - d_obj
    return gap, p_obj, d_obj


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


def load_somato_data():
    data_path = somato.data_path()
    subject = "01"
    task = "somato"

    raw_fname = os.path.join(data_path, "sub-{}".format(subject), "meg",
                             "sub-{}_task-{}_meg.fif".format(subject, task))
    fwd_fname = os.path.join(data_path, "derivatives",
                             "sub-{}".format(subject),
                             "sub-{}_task-{}-fwd.fif".format(subject, task))

    # Read evoked
    raw = mne.io.read_raw_fif(raw_fname)
    events = mne.find_events(raw, stim_channel="STI 014")
    reject = dict(grad=4000e-13, eog=350e-6)
    picks = mne.pick_types(raw.info, meg=True, eog=True)

    event_id, tmin, tmax = 1, -1.0, 3.0
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        reject=reject, preload=True)
    evoked = epochs.filter(1, None).average()
    evoked = evoked.pick_types(meg=True)
    evoked.crop(tmin=0.03, tmax=0.05)  # Choose a timeframe not too large

    # Handling forward solution
    forward = mne.read_forward_solution(fwd_fname)
    noise_cov = mne.compute_covariance(epochs, rank="info", tmax=0.0)

    return evoked, forward, noise_cov


def load_data_from_camcan(folder_name, data_path, orient):
    data_path = Path(data_path)

    subject_dir = data_path / "subjects"

    fwd_fname = data_path / "meg" / f"{folder_name}_task-passive-fwd.fif"
    ave_fname = data_path / "meg" / f"{folder_name}_task-passive-ave.fif"
    cleaned_epo_fname = (
        data_path / "meg" / f"{folder_name}_task-passive_cleaned-epo.fif"
    )

    # Building noise covariance
    cleaned_epochs = mne.read_epochs(cleaned_epo_fname)
    noise_cov = mne.compute_covariance(cleaned_epochs, tmax=0, rank="info")

    evokeds = mne.read_evokeds(ave_fname, condition=None, baseline=(None, 0))
    evoked = evokeds[-2]

    if not os.path.exists(f"../data/camcan/evokeds/{folder_name}"):
        os.mkdir(f"../data/camcan/evokeds/{folder_name}")
    joblib.dump(evoked, f"../data/camcan/evokeds/{folder_name}/evoked_{orient}_full.pkl")

    forward = mne.read_forward_solution(fwd_fname)

    evoked.crop(tmin=0.08, tmax=0.15)  # 0.08 - 0.15
    evoked = evoked.pick_types(eeg=False, meg=True)

    return evoked, forward, noise_cov
