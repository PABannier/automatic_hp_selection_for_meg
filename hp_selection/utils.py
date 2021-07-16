import numpy as np
from numpy.linalg import norm
from mne.inverse_sparse.mxne_inverse import (_prepare_gain, is_fixed_orient,
                                             _reapply_source_weighting,
                                             _make_sparse_stc)
from mne.inverse_sparse.mxne_optim import (mixed_norm_solver, 
                                           iterative_mixed_norm_solver)

def groups_norm2(A, n_orient=1):
    """Compute squared L2 norms of groups inplace."""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def norm_l2_05(X, n_orient, copy=True):
    """Compute l_2,p norm"""
    if X.size == 0:
        return 0.0
    if copy:
        X = X.copy()
    return np.sqrt(np.sqrt(groups_norm2(X, n_orient)))


def norm_l2_inf(X, n_orient, copy=True):
    """Compute l_2,inf norm"""
    if X.size == 0:
        return 0.0
    if copy:
        X = X.copy()
    return np.sqrt(np.max(groups_norm2(X, n_orient)))


def compute_alpha_max(G, M, n_orient):
    """Compute alpha max"""
    return norm_l2_inf(np.dot(G.T, M), n_orient, copy=False)


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


def solve_irmxne_problem(G, M, alpha, n_orient, n_mxne_iter=5):
        if n_mxne_iter == 1:
            X, active_set, _ = mixed_norm_solver(M, G, alpha, 
                                                 n_orient=n_orient)
        else:
            X, active_set, _ = iterative_mixed_norm_solver(M, G, alpha, 
                                                           n_mxne_iter, 
                                                           n_orient=n_orient)
        return X, active_set
