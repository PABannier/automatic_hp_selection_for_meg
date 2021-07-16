import numpy as np
from hp_selection.utils import compute_alpha_max, solve_irmxne_problem


def compute_log_likelihood(G, M_val, X, sigma=1):
    """
    Computes the Type-II log-likelihood criterion. See reference:
    https://www.biorxiv.org/content/10.1101/2020.08.10.243774v4.full.pdf
    """
    cov_M_val = np.cov(M_val)
    Gamma = np.diag(np.var(X, axis=-1))
    sigma_M_train = (sigma ** 2) * np.eye(G.shape[0]) + G @ Gamma @ G.T
    # must be invertible as a covariance matrix
    sigma_M_train_inv = np.linalg.inv(sigma_M_train)
    return np.trace(cov_M_val @ sigma_M_train_inv) + np.log(
        np.linalg.det(sigma_M_train)
    )

def solve_using_temporal_cv(G, M, n_orient, n_mxne_iter=5, grid_length=15, K=5,
                            random_state=0):
    """
    Solves the multi-task Lasso problem with a group l2,0.5 penalty with irMxNE.
    Regularization hyperparameter selection is done using (temporal) CV.
    """
    alpha_max = compute_alpha_max(G, M, n_orient)

    folds = np.array_split(range(M.shape[1]), K)
    loss_path = np.empty((K, grid_length))
    grid = np.geomspace(alpha_max, alpha_max * 0.1, grid_length)

    for i in range(len(folds)):
        train_folds = folds.copy()
        del train_folds[i]
        train_indices = np.concatenate(train_folds)
        val_indices = folds[i]

        # Contiguous split (time series) of the measurement matrix
        M_train, M_val = M[:, train_indices], M[:, val_indices]

        # Fitting on grid
        for j, alpha in enumerate(grid, total=len(grid)):
            X_ = solve_irmxne_problem(G, M_train, alpha, n_orient,
                                      n_mxne_iter)[0]
            loss_ = compute_log_likelihood(G, M_val, X_, sigma=1)
            loss_path[i, j] = loss_

    loss_path = loss_path.mean(axis=0)
    idx_selected_alpha = loss_path.argmin()
    best_alpha = grid[idx_selected_alpha]

    # Refitting
    best_X, best_as = solve_irmxne_problem(G, M, best_alpha, n_orient,
                                           n_mxne_iter=5)

    return best_X, best_as
