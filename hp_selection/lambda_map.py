import numpy as np
from mne.inverse_sparse.mxne_optim import (mixed_norm_solver, 
                                           iterative_mixed_norm_solver)
from mne.inverse_sparse.mxne_inverse import (_prepare_gain, _make_sparse_stc,
                                             is_fixed_orient, _log_exp_var,
                                             _reapply_source_weighting)
from hp_selection.utils import norm_l2_inf, groups_norm2


def solve_using_lambda_map(evoked, forward, noise_cov, depth=0.9, loose=0.9,
                           n_mxne_iter=5):
    alpha_init = 1 # ???????
    return mixed_norm_hyperparam(evoked, forward, noise_cov, alpha_init, 
                                 depth=depth, loose=loose, 
                                 n_mxne_iter=n_mxne_iter)


def mixed_norm_hyperparam(evoked, forward, noise_cov, alpha, b=1/3, loose=0.9,
                          depth=0.9, maxit=3000, tol=1e-4, active_set_size=10,
                          debias=True, time_pca=True, weights=None,
                          weights_min=0., solver='auto', n_mxne_iter=5,
                          dgap_freq=10, rank=None, pick_ori=None, hp_iter=9,
                          random_state=None, verbose=None):
    """
    This function is almost identical to mixed_norm in MNE. The hyperparameter
    selection process is different however. MNE implements a SURE-based approach
    by choosing over a grid of possible candidates the alpha that minimizes the
    SURE.

    The goal of this function is to implement lambda map hyperparameter
    selection presented by Bekhti et al.

    Parameters
    ----------

    alpha: float
        Initial value of alpha.

    b: float, default = 1/3
        Hyperparameter in the Gamma hyperprior. Bekhti et al. found that 1/3
        yields decent results for MEG problems.

    hp_iter: int, default = 9
        Number of iterations for alpha computation.
    """
    def g(w, n_dip_per_pos):
        if n_mxne_iter == 1:
            return np.sqrt(groups_norm2(w.copy(), n_dip_per_pos))
        else:
            return np.sqrt(np.sqrt(groups_norm2(w.copy(), n_dip_per_pos)))

    pca = True
    if not isinstance(evoked, list):
        evoked = [evoked]

    all_ch_names = evoked[0].ch_names
    if not all(all_ch_names == evoked[i].ch_names
               for i in range(1, len(evoked))):
        raise Exception('All the datasets must have the same good channels.')

    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked[0].info, noise_cov, pca, depth, loose, rank,
        weights, weights_min)

    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    M = np.concatenate([e.data[sel] for e in evoked], axis=1)

    # Whiten data
    print('Whitening data matrix.')
    M = np.dot(whitener, M)

    if time_pca:
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        if not isinstance(time_pca, bool) and isinstance(time_pca, int):
            U = U[:, :time_pca]
            s = s[:time_pca]
            Vh = Vh[:time_pca]
        M = U * s

    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3
    alpha_max = norm_l2_inf(np.dot(gain.T, M), n_dip_per_pos, copy=False)
    mode = alpha_max / 2
    a = mode * b + 1

    k = 1 if n_mxne_iter == 1 else 0.5

    alphas = []
    alphas.append(alpha)

    for i_hp in range(hp_iter):
        if n_mxne_iter == 1:
            X, active_set, _ = mixed_norm_solver(
                M, gain, alpha, maxit=maxit, tol=tol,
                active_set_size=active_set_size, n_orient=n_dip_per_pos,
                debias=debias, solver=solver, dgap_freq=dgap_freq,
                verbose=verbose)
        else:
            X, active_set, _ = iterative_mixed_norm_solver(
                M, gain, alpha, n_mxne_iter, maxit=maxit, tol=tol,
                n_orient=n_dip_per_pos, active_set_size=active_set_size,
                debias=debias, solver=solver, dgap_freq=dgap_freq,
                verbose=verbose)

        scale = np.shape(X)[0] * np.shape(X)[1]
        alpha = (scale / k + a) / (np.sum(g(X, n_dip_per_pos)) + np.shape(X)[1])
        alphas.append(alpha)

        if abs(alphas[-2] - alphas[-1]).max() < 1e-2:
            print('Hyperparameter estimated: Convergence reached after'
                  '  %d iterations!' % i_hp)
            break

    if time_pca:
        X = np.dot(X, Vh)
        M = np.dot(M, Vh)

    gain_active = gain[:, active_set]
    if mask is not None:
        active_set_tmp = np.zeros(len(mask), dtype=bool)
        active_set_tmp[mask] = active_set
        active_set = active_set_tmp
        del active_set_tmp

    if active_set.sum() == 0:
        raise Exception("No active dipoles found. alpha is too big.")

    # Reapply weights to have correct unit
    X = _reapply_source_weighting(X, source_weighting, active_set)
    source_weighting[source_weighting == 0] = 1  # zeros
    gain_active /= source_weighting[active_set]
    del source_weighting
    M_estimate = np.dot(gain_active, X)

    outs = list()
    cnt = 0
    for e in evoked:
        tmin = e.times[0]
        tstep = 1.0 / e.info['sfreq']
        Xe = X[:, cnt:(cnt + len(e.times))]
        out = _make_sparse_stc(Xe, active_set, forward, tmin, tstep,
                               pick_ori=pick_ori)
        outs.append(out)
        cnt += len(e.times)

    _log_exp_var(M, M_estimate, prefix='')

    if len(outs) == 1:
        out = outs[0]
    else:
        out = outs

    return out

