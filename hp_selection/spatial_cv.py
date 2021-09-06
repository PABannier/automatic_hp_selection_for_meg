import numpy as np
from numpy.linalg import norm

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import f1_score, jaccard_score

from hp_selection.solver_free_orient import MultiTaskLassoUnscaled
from hp_selection.utils import (compute_alpha_max, solve_irmxne_problem,
                                build_full_coefficient_matrix)



class ReweightedMultiTaskLassoCV(BaseEstimator, RegressorMixin):
    """Cross-validate the regularization penalty constant `alpha`
    for a reweighted multi-task LASSO regression.

    Parameters
    ----------
    alpha_grid : list or np.ndarray
        Values of `alpha` to test.

    criterion : Callable, default=mean_squared_error
        Cross-validation metric (e.g. MSE, SURE).

    n_folds : int, default=5
        Number of folds.

    n_iterations : int, default=5
        Number of reweighting iterations performed during fitting.

    random_state : int or None, default=None
        Seed for reproducible experiments.

    penalty : callable, default=None
        See docs of ReweightedMultiTaskLasso for more details.

    n_orient : int, default=1
        Number of orientations for a dipole on the scalp surface. Choose 1 for
        fixed orientation and 3 for free orientation.
    """
    def __init__(self, alpha_grid: list, criterion = mean_squared_error,
                 n_folds = 5, n_iterations = 5, random_state = None,
                 penalty = None, n_orient = 1):
        if not isinstance(alpha_grid, (list, np.ndarray)):
            raise TypeError(
                "The parameter grid must be a list or a Numpy array."
            )

        self.alpha_grid = alpha_grid
        self.criterion = criterion
        self.n_folds = n_folds
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.n_orient = n_orient

        self.best_estimator_ = None
        self.best_cv_, self.best_alpha_ = np.inf, None

        self.n_alphas = len(self.alpha_grid)
        self.mse_path_ = np.full((self.n_alphas, n_folds), np.inf)

        if penalty:
            self.penalty = penalty
        else:
            self.penalty = lambda u: 1 / (
                2 * np.sqrt(norm(u, axis=1)) + np.finfo(float).eps
            )

    @property
    def coef_(self):
        return self.best_estimator_.coef_

    def fit(self, X, Y):
        """Fits the cross-validation error estimator
        on X and Y.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix.

        Y : np.ndarray of shape (n_samples, n_tasks)
            Target matrix.
        """
        X, Y = check_X_y(X, Y, multi_output=True)

        scores_per_alpha_ = [np.inf for _ in range(self.n_alphas)]
        Y_oofs_ = [np.zeros(Y.shape) for _ in range(self.n_alphas)]

        kf = KFold(self.n_folds, random_state=self.random_state, shuffle=True)

        for i, (trn_idx, val_idx) in enumerate(kf.split(X, Y)):
            print(f"Fitting fold {i+1}...")
            X_train, Y_train = X[trn_idx, :], Y[trn_idx, :]
            X_valid, Y_valid = X[val_idx, :], Y[val_idx, :]

            coefs_ = self._fit_reweighted_with_grid(X_train, Y_train, X_valid,
                                                    Y_valid, i)
            predictions_ = [X_valid @ coefs_[j] for j in range(self.n_alphas)]

            for i in range(len(Y_oofs_)):
                Y_oofs_[i][val_idx, :] = predictions_[i]

        for i in range(len(Y_oofs_)):
            scores_per_alpha_[i] = self.criterion(Y, Y_oofs_[i])

        self.best_cv_ = np.min(scores_per_alpha_)
        self.best_alpha_ = self.alpha_grid[np.argmin(scores_per_alpha_)]

        print("\n")
        print(f"Best criterion: {self.best_cv_}")
        print(f"Best alpha: {self.best_alpha_}")

    def _fit_reweighted_with_grid(self, X_train, Y_train, X_valid, Y_valid,
                                  idx_fold):
        n_features, n_tasks = X_train.shape[1], Y_train.shape[1]
        coef_0 = np.empty((self.n_alphas, n_features, n_tasks))

        regressor = MultiTaskLassoUnscaled(np.nan, warm_start=True,
                                           n_orient=self.n_orient,
                                           accelerated=True)

        # Copy grid of first iteration (leverages convexity)
        for j, alpha in enumerate(self.alpha_grid):
            regressor.alpha = alpha
            coef_0[j] = regressor.fit(X_train, Y_train).coef_

        regressor.warm_start = False
        coefs = coef_0.copy()

        for j, alpha in enumerate(self.alpha_grid):
            regressor.alpha = alpha
            w = self.penalty(coef_0[j])

            for _ in range(self.n_iterations - 1):
                mask = w != 1.0 / np.finfo(float).eps
                coefs[j][~mask] = 0.0

                if mask.sum():
                    coefs[j][mask], w[mask] = self._reweight_op(
                        regressor, X_train[:, mask], Y_train, w[mask]
                    )

                    self.mse_path_[j, idx_fold] = mean_squared_error(
                        Y_valid, X_valid @ coefs[j]
                    )
                else:
                    self.mse_path_[j, idx_fold] = mean_squared_error(
                        Y_valid, np.zeros_like(Y_valid))

        return coefs

    def _reweight_op(self, regressor, X, Y, w):
        X_w = X / w[np.newaxis, :]
        regressor.fit(X_w, Y)

        w = np.expand_dims(w, axis=-1)
        coef = regressor.coef_ / w
        w = self.penalty(coef)

        return coef, w

    def predict(self, X: np.ndarray):
        """Predicts data with the fitted coefficients.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix for inference.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.best_estimator_.predict(X)


def solve_using_spatial_cv(G, M, n_orient, n_mxne_iter=5, grid_length=14, K=5,
                           random_state=0):
    """
    Solves the multi-task Lasso problem with a group l2,0.5 penalty with
    irMxNE. Regularization hyperparameter selection is done using (spatial) CV.
    """
    alpha_max = compute_alpha_max(G, M, n_orient)
    grid = np.linspace(alpha_max, alpha_max * 0.1, grid_length)

    criterion = ReweightedMultiTaskLassoCV(grid, n_folds=K,
                                           n_iterations=n_mxne_iter,
                                           random_state=random_state,
                                           n_orient=n_orient)
    criterion.fit(G, M)
    best_alpha = criterion.best_alpha_

    # Refitting
    best_X, best_as = solve_irmxne_problem(G, M, best_alpha, n_orient,
                                           n_mxne_iter=n_mxne_iter)
    return best_X, best_as
