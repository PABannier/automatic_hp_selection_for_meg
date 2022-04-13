import numpy as np

from calibromatic.mixed_norm import MixedNorm
from calibromatic.utils import norm_l2_inf, groups_norm2


def fit_lambda_map(G, M, n_orient, b=5, n_iter=10):
    """Calibrate Lasso model with a Bayesian maximum-a-posteriori criterion.

    Parameters
    ----------
    G : array, shape (n_sensors, n_sources)
        The gain matrix.

    M : array, shape (n_sensors, n_times)
        The measurement matrix.

    n_orient : int
        Number of orientations. 1 if fixed orientation, otherwise free.

    b : float
        Parameter for hyperprior.

    n_iter : int
        Number of iterations to find optimal regularization parameter.

    Returns
    -------
    X : array, shape (n_sensors, ws_size)
        The coefficient matrix restricted to the active set.

    active_set : array, shape (n_sources)
        Boolean array containing the activated sources.

    References
    ----------
    .. [1] Y. Bekhti, R. Badeau, A. Gramfort
      "Hyperparameter Estimation in Maximum a Posteriori Regression using sparsity with
      an application to brain imaging", EUSIPCO 2017,
      https://hal.archives-ouvertes.fr/hal-01531238
    """
    def g(w, n_orient):
        if n_orient == 1:
            return np.sqrt(groups_norm2(w.copy(), n_orient))
        else:
            return np.sqrt(np.sqrt(groups_norm2(w.copy(), n_orient)))

    alpha_max = norm_l2_inf(np.dot(G.T, M), n_orient, copy=True)
    mode = alpha_max / 2
    a = mode * b + 1

    print("alpha max: ", alpha_max)
    k = 1 if n_orient == 1 else 0.5

    alpha = alpha_max / 2
    alphas = []
    alphas.append(alpha)
    estimator = None

    for i_hp in range(n_iter):
        print("[FITTING] Iteration %d :: alpha %.3f" % (i_hp, alpha))
        estimator = MixedNorm(alpha, n_orient=n_orient, p0=10)
        estimator.fit(G, M)

        # Computation of new alpha
        alpha = (M.size / k + a - 1) / np.sum(g(estimator.coef_, n_orient) + b)
        alphas.append(alpha)

        if abs(alphas[-2] - alphas[-1]).max() < 1e-2:
            print('Hyperparameter estimated: Convergence reached after'
                    ' %d iterations!' % i_hp)
            break

    active_set = np.linalg.norm(estimator.coef_, axis=-1) != 0
    X = estimator.coef_[active_set]

    return X, active_set
