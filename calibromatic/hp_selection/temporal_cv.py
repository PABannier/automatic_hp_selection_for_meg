import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import MultiTaskLasso
from sklearn.utils import check_X_y

from calibromatic.sparse_solver import MixedNorm
from calibromatic.utils import compute_alpha_max


class LLForReweightedMTL:
    r"""The Type-II Log-Likelihood criterion evaluated with temporal cross-validation.

    Parameters
    ----------
    sigma : float
        The noise estimate.

    alpha_grid : array, shape (n_alphas,)
        Grid of regularization parameter to test.

    n_reweighting : int, optional
        Number of penalty reweighting.

    penalty : callable, optional
        Non-convex penalty used to reweight the design matrix.

    n_orient: int, optional
        Number of orientation for a dipole. 1 for fixed orientation, > 1 for free.

    random_state : int or None, optional
        Seed for reproducible experiments.

    Attributes
    ----------
    ll_path_ : array, shape (n_alphas,)
        The Log-likehood criterion value along the path.

    trace_path_ : array, shape (n_alphas,)
        The trace term contribution to the LL criterion along the path.

    log_det_path_ : array, shape (n_alphas,)
        The Log determinant term contribution to the LL criterion along the path.
        It can be interpreted as a regularization constraint on the objective.

    best_coef_ : array, shape (n_sources, n_times)
        The coefficient matrix minimizing the criterion.

    References
    ----------
    .. [1] A. Hashemi et al.
    "Unification of sparse Bayesian learning algorithms for electromagnetic brain
    imaging with the Majorization Minimization framework",
    https://www.biorxiv.org/content/10.1101/2020.08.10.243774v4.full.pdf
    """

    def __init__(self, sigma, alpha_grid, n_reweighting=5, penalty=None,
                 n_orient=1, random_state=None):
        if not isinstance(alpha_grid, (list, np.ndarray)):
            raise TypeError("The parameter grid must be a list or a Numpy array.")

        self.sigma = sigma
        self.alpha_grid = alpha_grid
        self.n_reweighting = n_reweighting
        self.n_orient = n_orient
        self.random_state = random_state

        self.n_alphas = len(self.alpha_grid)

        self.ll_path_ = np.empty(self.n_alphas)
        self.trace_path_ = np.empty(self.n_alphas)
        self.log_det_path_ = np.empty(self.n_alphas)

        self.best_coef_ = None

        if self.n_orient <= 0:
            raise ValueError("Number of orientations can't be negative.")

        self.penalty = penalty if penalty else self._penalty

    def fit(self, X, Y):
        """Fit temporal cross-validation to select best `alpha`.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        Y : array, shape (n_samples, n_tasks)
            Target matrix.
        """
        X, Y = check_X_y(X, Y, multi_output=True)
        coefs_grid = self._fit_reweighted_with_grid(X, Y)

        for i, coef in enumerate(coefs_grid):
            trace_term, log_det_term, loss_ = self._compute_ll_val(X, coef, Y)
            self.ll_path_[i] = loss_
            self.trace_path_[i] = trace_term
            self.log_det_path_[i] = log_det_term

        best_ll_ = np.min(self.ll_path_)
        best_idx = np.argmin(self.ll_path_)
        best_alpha_ = self.alpha_grid[best_idx]
        self.coefs_grid_ = coefs_grid
        self.best_coef_ = coefs_grid[best_idx]
        return best_ll_, best_alpha_

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
        X_w = X / np.repeat(w[np.newaxis, :], self.n_orient)
        regressor.fit(X_w, Y)

        if self.n_orient == 1:
            coef = (regressor.coef_ / w).T
        else:
            coef = (regressor.coef_.T / np.repeat(w[np.newaxis, :],
                    self.n_orient)).T
        w = self.penalty(coef)
        return coef, w

    def _compute_ll_val(self, X, W, Y):
        """Compute the Type-II Log-Likelihood criterion.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The design matrix.

        W : array, shape (n_features, n_tasks)
            The coefficient matrix.

        Y : array, shape (n_samples, n_tasks)
            The measurement matrix.
        """
        cov_Y_val = np.cov(Y)  # (n_samples, n_samples)
        Gamma = self._compute_groupwise_var(W)
        sigma_Y_train = (self.sigma ** 2) * np.eye(X.shape[0]) + (X * Gamma) @ X.T
        sigma_Y_train_inv = np.linalg.inv(sigma_Y_train)
        trace = np.trace(cov_Y_val @ sigma_Y_train_inv)
        log_det = np.linalg.slogdet(sigma_Y_train)[1]
        return trace, log_det, trace + log_det

    def _fit_reweighted_with_grid(self, X, Y):
        """Fit an iteratively reweighted Mixed Norm to the data.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The design matrix.

        Y : array, shape (n_samples, n_tasks)
            The measurement matrix.

        Returns
        -------
        coefs : array, shape (n_features, n_tasks)
            The coefficient matrix.
        """
        n_features, n_tasks = X.shape[1], Y.shape[1]
        n_alphas = len(self.alpha_grid)

        coef_0 = np.empty((n_alphas, n_features, n_tasks))

        # Warm start first iteration
        if self.n_orient == 1:
            regressor = MultiTaskLasso(np.nan, fit_intercept=False, warm_start=True)
        else:
            regressor = MixedNorm(np.nan, warm_start=True, n_orient=self.n_orient)

        # Copy grid of first iteration (leverages convexity)
        print("First iteration")
        for j, alpha in enumerate(self.alpha_grid):
            regressor.alpha = alpha

            if self.n_orient == 1:
                coef_0[j] = regressor.fit(X, Y).coef_.T
            else:
                coef_0[j] = regressor.fit(X, Y).coef_

        regressor.warm_start = False

        coefs_ = coef_0.copy()

        print("Next iterations...")
        for j, alpha in enumerate(self.alpha_grid):
            regressor.alpha = alpha
            w = self.penalty(coef_0[j])

            for _ in range(self.n_iterations - 1):
                mask = w != 1.0 / np.finfo(float).eps
                mask_full = np.repeat(mask, self.n_orient)

                coefs_[j][~mask_full] = 0.0

                if mask.sum():
                    coefs_[j][mask_full], w[mask] = self._reweight_op(
                        regressor, X[:, mask_full], Y, w[mask]
                    )
        return coefs_

    def _compute_groupwise_var(self, X):
        """Group-wise variance along the temporal axis of the coefficient matrix.

        Parameters
        ----------
        X : array, shape (n_features, n_tasks)
            The coefficient matrix

        Returns
        -------
        variances : array, shape (n_positions)
            The variances for each block of coordinates.
        """
        n_positions = X.shape[0] // self.n_orient
        variances = np.zeros(X.shape[0], dtype=np.float32)

        for j in range(n_positions):
            idx = slice(j * self.n_orient, (j + 1) * self.n_orient)
            variances[idx] = (np.linalg.norm(X[idx, :], 'fro') ** 2
                              / X[idx, :].size)
            # variances[idx] = np.var(X[idx, :])
        return variances

    def _penalty(self, coef):
        """Non-convex penalty for reweighting the design matrix from the coefficients.

        Parameters
        ----------
        coef : array, shape (n_features, n_times)
            Coefficient matrix.

        Returns
        -------
        penalty : array, shape (n_positions,)
            Penalty vector.
        """
        n_positions = coef.shape[0] // self.n_orient
        coef = coef.reshape(n_positions, -1)
        m_norm = np.sqrt(norm(coef, axis=1))
        return 1 / (2 * m_norm + np.finfo(float).eps)


def temporal_cv(G, M, n_orient, n_mxne_iter=5, grid_length=15, random_state=None):
    """Calibrate Lasso model with a cross-validation with temporal splits.

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
    # Sigma = 1 because data are already pre-whitened
    criterion = LLForReweightedMTL(1, grid, n_orient=n_orient, n_iterations=n_mxne_iter,
                                   random_state=random_state)
    best_coef_ = criterion.best_coef_
    as_ = np.linalg.norm(best_coef_, axis=1) != 0
    X_ = best_coef_[as_]
    return X_, as_
