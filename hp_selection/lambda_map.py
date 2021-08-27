import numpy as np
from hp_selection.solver_free_orient import MultiTaskLassoUnscaled
from hp_selection.utils import norm_l2_inf, groups_norm2


def solve_using_lambda_map(G, M, n_orient, b=1, hp_iter=10, n_mxne_iter=5,
                           random_state=None):
    def g(w, n_orient):
        if n_orient == 1:
            return np.sqrt(groups_norm2(w.copy(), n_orient))
        else:
            return np.sqrt(np.sqrt(groups_norm2(w.copy(), n_orient)))

    alpha_max = norm_l2_inf(np.dot(G.T, M), n_orient, copy=True)
    mode = alpha_max / 2
    a = mode * b + 1

    print("ALPHA MAX: ", alpha_max)

    k = 1 if n_orient == 1 else 0.5

    # Initial value of alpha
    alpha = alpha_max / 2

    alphas = []
    alphas.append(alpha)

    estimator = None

    for i_hp in range(hp_iter):
        print("[FITTING] Iteration %d :: alpha %.3f" % (i_hp, alpha))

        estimator = MultiTaskLassoUnscaled(alpha, n_orient=n_orient,
                                           active_set_size=10)
        estimator.fit(G, M)

        # Computation of new alpha
        alpha = (M.size / k + a - 1) / np.sum(g(estimator.coef_, n_orient) + b)
        print("NEW ALPHA: ", alpha)
        alphas.append(alpha)

        if abs(alphas[-2] - alphas[-1]).max() < 1e-2:
            print('Hyperparameter estimated: Convergence reached after'
                    ' %d iterations!' % i_hp)
            break

    as_ = np.linalg.norm(estimator.coef_, axis=-1) != 0
    X_ = estimator.coef_[as_]

    return X_, as_
