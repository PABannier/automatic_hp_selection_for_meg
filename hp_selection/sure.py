import numpy as np
from mne.inverse_sparse.mxne_inverse import mixed_norm


def solve_using_sure(evoked, forward, noise_cov, depth=0.9, loose=0.9,
                     n_mxne_iter=5, random_state=0):
    """
    Solves the multi-task Lasso problem with a group l2,0.5 penalty
    with irMxNE. Regularization hyperparameter selection is done with the
    SURE criterion.
    """
    return mixed_norm(evoked, forward, noise_cov, depth=depth, loose=loose,
                      n_mxne_iter=n_mxne_iter, random_state=random_state,
                      debias=False, sure_alpha_grid=np.linspace(100, 10, 15))
