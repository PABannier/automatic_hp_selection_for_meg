import functools
import numpy as np
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
