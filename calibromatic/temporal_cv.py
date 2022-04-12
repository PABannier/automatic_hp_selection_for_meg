from calibromatic.ll_warm_start import LLForReweightedMTL
import numpy as np
from calibromatic.utils import compute_alpha_max, solve_irmxne_problem


def solve_using_temporal_cv(G, M, n_orient, n_mxne_iter=5, grid_length=15, K=5,
                            random_state=None):
    """
    Solves the multi-task Lasso problem with a group l2,0.5 penalty with
    irMxNE. Regularization hyperparameter selection is done using (temporal)
    CV.
    """
    alpha_max = compute_alpha_max(G, M, n_orient)
    grid = np.linspace(alpha_max, alpha_max * 0.1, grid_length)
    # Sigma = 1 because data are already pre-whitened
    criterion = LLForReweightedMTL(1, grid, n_orient=n_orient,
                                   random_state=random_state)
    best_alpha = criterion.get_val(G, M)[1]
    best_coef_ = criterion.best_coef_
    best_as_ = np.linalg.norm(best_coef_, axis=1) != 0
    best_X = best_coef_[best_as_]
    return best_X, best_as_
