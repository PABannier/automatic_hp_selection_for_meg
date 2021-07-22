from hp_selection.ll_warm_start import LLForReweightedMTL
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


def solve_using_temporal_cv(G, M, n_orient, n_mxne_iter=5, grid_length=50, K=5,
                            random_state=None):
    """
    Solves the multi-task Lasso problem with a group l2,0.5 penalty with
    irMxNE. Regularization hyperparameter selection is done using (temporal)
    CV.
    """
    # Scaling alpha by number of samples (LLForReweightedMTL expects scaled
    # alpha)
    alpha_max = compute_alpha_max(G, M, n_orient) / G.shape[0]
    grid = np.geomspace(alpha_max, alpha_max * 0.1, grid_length)
    # Sigma = 1 because data are already pre-whitened
    criterion = LLForReweightedMTL(1, grid, n_orient=n_orient,
                                   random_state=random_state)
    best_alpha = criterion.get_val(G, M)[1]

    # Refitting
    # Re-scaling alpha by multiplying it by n_samples since
    # solve_irmxne_problem expects an unnormalized alpha.
    best_X, best_as = solve_irmxne_problem(G, M, best_alpha * G.shape[0],
                                           n_orient, n_mxne_iter=n_mxne_iter)
    return best_X, best_as
