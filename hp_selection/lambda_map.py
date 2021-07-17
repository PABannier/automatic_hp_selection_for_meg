import numpy as np
from numpy.linalg import norm
from mne.inverse_sparse.mxne_optim import mixed_norm_solver
from mne.inverse_sparse.mxne_debiasing import compute_bias
from hp_selection.utils import norm_l2_inf, groups_norm2


def _weights(X, G, alpha, active_set, n_orient):
    if np.shape(alpha):
        # alpha = np.tile(alpha, [n_orient, 1]).ravel(order='F')
        G_tilde = np.dot(G, np.diag(1. / alpha))
        alpha_tilde = 1.
        X_tilde = np.dot(np.diag(alpha[active_set]), X)
    else:
        G_tilde = G / alpha
        alpha_tilde = 1.
        X_tilde = X * alpha
    return X_tilde, G_tilde, alpha_tilde


def solve_using_gamma_map(G, M, n_orient, n_mxne_iter=5, hp_iter=9, a=1, b=1):
    if n_mxne_iter > 1:
        return iterative_mixed_norm_solver_hyperparam(M, G, alpha, n_mxne_iter,
                                                      hp_iter=hp_iter, a=a,
                                                      b=b, n_orient=n_orient)
    else:
        return mixed_norm_solver_hyperparam(M, G, alpha, hp_iter, a=a, b=b,
                                            n_orient=n_orient)


def mixed_norm_solver_hyperparam(M, G, alpha, hp_iter, maxit=3000,
                                 tol=1e-8, verbose=None, active_set_size=50,
                                 debias=True, n_orient=1, solver='auto',
                                 a=1., b=1., update_alpha=True):
    def g(w):
        return np.sqrt(groups_norm2(w.copy(), n_orient))

    # Compute the parameter a of the Gamma distribution
    alpha_max = norm_l2_inf(np.dot(G.T, M), n_orient, copy=False)
    mode = alpha_max / 2.
    a = mode * b + 1.

    E = list()

    active_set = np.ones(G.shape[1], dtype=np.bool)
    X = np.zeros((G.shape[1], M.shape[1]))
    alphas = []
    alphas.append(alpha)

    for k in range(hp_iter):
        active_set = np.ones(G.shape[1], dtype=np.bool)
        X = np.zeros((G.shape[1], M.shape[1]))

        X0 = X.copy()
        active_set_0 = active_set.copy()
        G_tmp = G[:, active_set]

        if np.shape(alpha):
            alpha_tmp = alpha[active_set]
        else:
            alpha_tmp = alpha

        if active_set_size is not None:
            if np.sum(active_set) > (active_set_size * n_orient):
                X, _active_set, _ = mixed_norm_solver(
                    M, G_tmp, alpha_tmp, debias=False, n_orient=n_orient,
                    maxit=maxit, tol=tol, active_set_size=active_set_size,
                    solver=solver, update_alpha=update_alpha, verbose=verbose)
            else:
                X, _active_set, _ = mixed_norm_solver(
                    M, G_tmp, alpha_tmp, debias=False, n_orient=n_orient,
                    maxit=maxit, tol=tol, active_set_size=None, solver=solver,
                    update_alpha=update_alpha, verbose=verbose)
        else:
            X, _active_set, _ = mixed_norm_solver(
                M, G_tmp, alpha_tmp, debias=False, n_orient=n_orient,
                maxit=maxit, tol=tol, active_set_size=None, solver=solver,
                update_alpha=update_alpha, verbose=verbose)

        print('active set size %d' % (_active_set.sum() / n_orient))

        if _active_set.sum() > 0:
            active_set[active_set] = _active_set

            X_tilde, G_tilde, alpha_tilde = _weights(X, G, alpha,
                                                     active_set,
                                                     n_orient=n_orient)
            p_obj = 0.5 * norm(M - np.dot(G_tilde[:, active_set], X_tilde),
                               'fro') ** 2. + alpha_tilde * np.sum(g(X))
            E.append(p_obj)

            # Check convergence
            if ((k >= 1) and np.all(active_set == active_set_0) and
                    np.all(np.abs(X - X0) < tol)):
                print('Convergence reached after %d reweightings!' % k)
                break
        else:
            active_set = np.zeros_like(active_set)
            p_obj = 0.5 * norm(M) ** 2.
            E.append(p_obj)
            break

        if np.shape(alpha):
            gX = (g(X) if (n_orient == 1) else
                  np.tile(g(X), [n_orient, 1]).ravel(order='F'))
            alpha[active_set] = (62. + a) / (gX + b)
        else:
            alpha = (62. + a) / (np.sum(g(X)) + b)
        alphas.append(alpha)

        if abs(alphas[-2] - alphas[-1]) < 1e-2:
            print('Hyperparameter estimated: Convergence reached after %d iterations!' % k)
            break

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    return X, active_set, E, alphas


def iterative_mixed_norm_solver_hyperparam(M, G, alpha, n_mxne_iter, hp_iter=9,
                                           maxit=3000, tol=1e-8, verbose=None,
                                           active_set_size=50, debias=True,
                                           n_orient=1, solver='auto',
                                           a=1., b=1., update_alpha=True):
    def g(w):
        if n_mxne_iter == 1:
            return np.sqrt(groups_norm2(w.copy(), n_orient))
        else:
            return np.sqrt(np.sqrt(groups_norm2(w.copy(), n_orient)))

    def gprime(w):
        return 2. * np.repeat(g(w), n_orient).ravel()
    # 1/0
    k = 1 if n_mxne_iter == 1 else 0.5

    # Compute the parameter a of the Gamma distribution
    alpha_max = norm_l2_inf(np.dot(G.T, M), n_orient, copy=False)
    mode = alpha_max / 2.
    a = mode * b + 1.

    E = list()

    alphas = []
    alphas.append(alpha.copy()) if np.shape(alpha) else alphas.append(alpha)

    for i_hp in range(hp_iter):
        active_set = np.ones(G.shape[1], dtype=np.bool)
        weights = np.ones(G.shape[1])
        X = np.zeros((G.shape[1], M.shape[1]))
        for i_iter in range(n_mxne_iter):
            X0 = X.copy()
            active_set_0 = active_set.copy()
            G_tmp = G[:, active_set] * weights[np.newaxis, :]
            alpha_tmp = (alpha[active_set][::n_orient] if np.shape(alpha)
                         else alpha)

            if active_set_size is not None:
                if np.sum(active_set) > (active_set_size * n_orient):
                    X, _active_set, _ = mixed_norm_solver(
                        M, G_tmp, alpha_tmp, debias=False, n_orient=n_orient,
                        maxit=maxit, tol=tol, active_set_size=active_set_size,
                        solver=solver, verbose=verbose)
                else:
                    X, _active_set, _ = mixed_norm_solver(
                        M, G_tmp, alpha_tmp, debias=False, n_orient=n_orient,
                        maxit=maxit, tol=tol, active_set_size=None,
                        solver=solver, verbose=verbose)
            else:
                X, _active_set, _ = mixed_norm_solver(
                    M, G_tmp, alpha_tmp, debias=False, n_orient=n_orient,
                    maxit=maxit, tol=tol, active_set_size=None, solver=solver,
                    verbose=verbose)

            print('active set size %d' % (_active_set.sum() / n_orient))

            if _active_set.sum() > 0:
                active_set[active_set] = _active_set

                # Reapply weights to have correct unit
                X *= weights[_active_set][:, np.newaxis]
                weights = gprime(X)
                if np.shape(alpha):
                    p_obj = \
                        0.5 * norm(M - np.dot(G[:, active_set],  X), 'fro') \
                        ** 2. + np.sum(alpha[active_set][::n_orient] * g(X))
                else:
                    p_obj = 0.5 * norm(M - np.dot(G[:, active_set],  X),
                            'fro') ** 2. + alpha * np.sum(g(X))
                E.append(p_obj)

                # Check convergence
                if ((i_iter >= 1) and np.all(active_set == active_set_0) and
                        np.all(np.abs(X - X0) < tol)):
                    print('Convergence reached after %d reweightings!' % i_iter)
                    break
            else:
                active_set = np.zeros_like(active_set)
                p_obj = 0.5 * norm(M) ** 2.
                E.append(p_obj)
                break

        # Compute the parameter a of the Gamma distribution
        # alpha_max = norm_l2_inf(np.dot(G_tmp.T, M), n_orient, copy=False)
        # alpha_max *= 0.01
        # alpha_max = 1.

        # 1/0

        if np.shape(alpha):
            scale = np.shape(X)[1]
            # gX = np.ones((active_set.shape)) * np.sum(g(X))
            gX = (g(X) if (n_orient == 1) else
                  np.tile(g(X), [n_orient, 1]).ravel(order='F'))
            alpha[active_set] = (scale / k + a) / (gX + b)
            # alpha = (64. / k + a) / (gX + b)
            # 1/0
        else:
            scale = np.shape(X)[0] * np.shape(X)[1]
            alpha = (scale / k + a) / (np.sum(g(X)) + np.shape(X)[1])
            # 1/0
            print('alpha: %s' % alpha)
        alphas.append(alpha)

        if abs(alphas[-2] - alphas[-1]).max() < 1e-2:
            print('Hyperparameter estimated: Convergence reached after'
                  '  %d iterations!' % i_hp)
            break

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    alphas = np.array(alphas)[:, active_set] if np.shape(alpha) else alphas
    return X, active_set, E, alphas
