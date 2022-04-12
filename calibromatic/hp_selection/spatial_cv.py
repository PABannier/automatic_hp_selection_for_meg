import numpy as np
from numpy.linalg import norm

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.base import BaseEstimator, RegressorMixin

from calibromatic.sparse_solver import MixedNorm
from calibromatic.utils import compute_alpha_max, solve_irmxne_problem



class SpatialCV(BaseEstimator, RegressorMixin):
    r"""Calibrates mixed norm with spatial cross-validation.

    Parameters
    ----------
    alpha_grid : array, shape (n_alphas,)
        Grid of regularization parameter to test.

    criterion : callable, optional
        Cross-validation metric.

    n_folds : int, optional
        Number of folds.

    n_reweighting : int, optional
        Number of penalty reweighing.

    penalty : callable, default=None
        See docs of ReweightedMultiTaskLasso for more details.

    n_orient: int, optional
        Number of orientation for a dipole. 1 for fixed orientation, > 1 for free.

    random_state : int or None, optional
        Seed for reproducible experiments.

    Attributes
    ----------
    coef_ : array, shape (n_sources, n_times)
        The coefficient matrix corresponding to `best_estimator_`.

    best_estimator_ : instance of MixedNorm
        The estimator minimizing the cross-validation criterion.

    best_cv_ : float
        The lowest cross-validation score.

    best_alpha_ : float
        The alpha corresponding to `best_cv_`.

    mse_path_ : array, shape (n_alphas,)
        The MSE value along the path.
    """

    def __init__(self, alpha_grid, criterion=mean_squared_error, n_folds=5,
                 n_reweighting=5, random_state=None, penalty=None, n_orient=1):
        if not isinstance(alpha_grid, (list, np.ndarray)):
            raise TypeError("The parameter grid must be a list or a Numpy array.")

        self.alpha_grid = alpha_grid
        self.criterion = criterion
        self.n_folds = n_folds
        self.n_reweighting = n_reweighting
        self.random_state = random_state
        self.n_orient = n_orient

        self.coef_ = None
        self.best_cv_, self.best_alpha_ = np.inf, None
        self.n_alphas = len(self.alpha_grid)
        self.mse_path_ = np.full((self.n_alphas, n_folds), np.inf)

        if penalty:
            self.penalty = penalty
        else:
            self.penalty = lambda u: 1 / (
                2 * np.sqrt(norm(u, axis=1)) + np.finfo(float).eps)

    def fit(self, X, Y):
        """Fit cross-validation to select best `alpha`.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        Y : array, shape (n_samples, n_tasks)
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
        self.coef_ = None  # TODO: register best coef

        print("\n")
        print(f"Best criterion: {self.best_cv_}")
        print(f"Best alpha: {self.best_alpha_}")

        return self

    def _fit_reweighted_with_grid(self, X_train, Y_train, X_valid, Y_valid, idx_fold):
        """Fit an iteratively reweighted Mixed Norm to the data.

        Parameters
        ----------
        X_train : array, shape (n_train_samples, n_features)
            The training design matrix.

        Y_train : array, shape (n_train_samples, n_tasks)
            The training measurement matrix.

        X_valid : array, shape (n_valid_samples, n_features)
            The validation design matrix.

        Y_valid : array, shape (n_valid_samples, n_tasks)
            The validation measurement matrix.

        idx_fold : int
            The fold index currently fitted.

        Returns
        -------
        coefs : array, shape (n_features, n_tasks)
            The coefficient matrix.
        """
        n_features, n_tasks = X_train.shape[1], Y_train.shape[1]
        coef_0 = np.empty((self.n_alphas, n_features, n_tasks))

        regressor = MixedNorm(np.nan, warm_start=True, n_orient=self.n_orient)

        # Copy grid of first iteration (leverages convexity)
        for j, alpha in enumerate(self.alpha_grid):
            regressor.alpha = alpha
            coef_0[j] = regressor.fit(X_train, Y_train).coef_

        regressor.warm_start = False
        coefs = coef_0.copy()

        for j, alpha in enumerate(self.alpha_grid):
            regressor.alpha = alpha
            w = self.penalty(coef_0[j])

            for _ in range(self.n_reweighting - 1):
                mask = w != 1.0 / np.finfo(float).eps
                coefs[j][~mask] = 0.0

                if mask.sum():
                    coefs[j][mask], w[mask] = self._reweight_op(
                        regressor, X_train[:, mask], Y_train, w[mask])

                    self.mse_path_[j, idx_fold] = mean_squared_error(
                        Y_valid, X_valid @ coefs[j])
                else:
                    self.mse_path_[j, idx_fold] = mean_squared_error(
                        Y_valid, np.zeros_like(Y_valid))
        return coefs

    def _reweight_op(self, regressor, X, Y, w):
        """Reweight design matrix by the weight (computing trick).

        Parameters
        ----------
        regressor : instance of MixedNorm
            The mixed norm estimator.

        X : array, shape (n_samples, n_features)
            The design matrix.

        Y : array, shape (n_samples, n_tasks)
            The measurement matrix.

        w : array, shape (n_features)
            A weight vector applied to the columns of X.

        Returns
        -------
        coef : array, shape (n_features, n_tasks)
            The coefficient matrix.

        w : array, shape (n_features)
            The updated weight vector.
        """
        X_w = X / w[np.newaxis, :]
        regressor.fit(X_w, Y)
        w = np.expand_dims(w, axis=-1)
        coef = regressor.coef_ / w
        w = self.penalty(coef)
        return coef, w


def spatial_cv(G, M, n_orient, n_mxne_iter=5, grid_length=15, n_folds=5, 
               random_state=0):
    """Calibrate Lasso model with a cross-validation splitted along the sensors.

    Parameters
    ----------
    G : array, shape (n_sensors, n_sources)
        The gain matrix.

    M : array, shape (n_sensors, n_times)
        The measurement matrix.

    n_orient : int
        Number of orientations. 1 if fixed orientation, otherwise free.

    n_mxne_iter : int
        Number of reweighting iterations of the mixed norm estimate.
    
    grid_length : int
        The grid length.
    
    n_folds : int  
        The number of folds.
    
    random_state : int
        The random state.

    Returns
    -------
    X : array, shape (n_sensors, ws_size)
        The coefficient matrix restricted to the active set.

    active_set : array, shape (n_sources)
        Boolean array containing the activated sources.
    """
    alpha_max = compute_alpha_max(G, M, n_orient)
    grid = np.linspace(alpha_max, alpha_max * 0.1, grid_length)

    criterion = SpatialCV(grid, n_folds=n_folds, n_iterations=n_mxne_iter,
                          n_orient=n_orient, random_state=random_state)
    criterion.fit(G, M)
    X_ = criterion.coef_
    as_ = norm(X_, axis=0) != 0
    return X_, as_
