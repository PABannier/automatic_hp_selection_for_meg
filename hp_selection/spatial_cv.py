import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from hp_selection.utils import (compute_alpha_max, solve_irmxne_problem,
                                build_full_coefficient_matrix)


def solve_using_spatial_cv(G, M, n_orient, n_mxne_iter=5, grid_length=15, K=5,
                           random_state=0):
    """
    Solves the multi-task Lasso problem with a group l2,0.5 penalty with
    irMxNE. Regularization hyperparameter selection is done using (spatial) CV.
    """
    kf = KFold(K, shuffle=True, random_state=random_state)
    loss_path = np.empty((K, grid_length))
    alpha_max = compute_alpha_max(G, M, n_orient)
    grid = np.geomspace(alpha_max, alpha_max * 0.1, grid_length)

    for i, (trn_indices, val_indices) in enumerate(kf.split(G, M)):
        G_train, G_val = G[trn_indices, :], G[val_indices, :]
        M_train, M_val = M[trn_indices, :], M[val_indices, :]

        # Fitting on grid
        for j, alpha in enumerate(grid):
            X_, as_ = solve_irmxne_problem(G_train, M_train, alpha, n_orient,
                                           n_mxne_iter)
            X = build_full_coefficient_matrix(as_, M.shape[1], X_)
            loss_ = mean_squared_error(M_val, G_val @ X)
            loss_path[i, j] = loss_

    loss_path = loss_path.mean(axis=0)
    idx_selected_alpha = loss_path.argmin()
    best_alpha = grid[idx_selected_alpha]

    # Refitting
    best_X, best_as = solve_irmxne_problem(G, M, best_alpha, n_orient,
                                           n_mxne_iter=5)
    return best_X, best_as
