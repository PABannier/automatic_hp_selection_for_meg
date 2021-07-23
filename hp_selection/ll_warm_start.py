from tqdm import tqdm

import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_X_y

from celer import MultiTaskLasso

from hp_selection.solver_free_orient import MultiTaskLassoUnscaled


class LLForReweightedMTL:
    def __init__(self, sigma, alpha_grid, n_iterations=5, penalty=None,
                 n_orient=1, random_state=None):
        self.sigma = sigma
        self.alpha_grid = alpha_grid
        self.n_iterations = n_iterations
        self.n_orient = n_orient
        self.random_state = random_state

        self.n_alphas = len(self.alpha_grid)

        self.ll_path_ = np.empty(self.n_alphas)
        self.trace_path_ = np.empty(self.n_alphas)
        self.log_det_path_ = np.empty(self.n_alphas)

        if self.n_orient <= 0:
            raise ValueError("Number of orientations can't be negative.")

        if penalty:
            self.penalty = penalty
        else:
            self.penalty = self._penalty

    def get_val(self, X, Y):
        X, Y = check_X_y(X, Y, multi_output=True)
        coefs_grid = self._fit_reweighted_with_grid(X, Y)

        for i, coef in enumerate(coefs_grid):
            trace_term, log_det_term, loss_ = self._compute_ll_val(X, coef, Y)
            self.ll_path_[i] = loss_
            self.trace_path_[i] = trace_term
            self.log_det_path_[i] = log_det_term

        best_ll_ = np.min(self.ll_path_)
        best_alpha_ = self.alpha_grid[np.argmin(self.ll_path_)]
        return best_ll_, best_alpha_

    def _reweight_op(self, regressor, X, Y, w):
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
        """
        Computes the Type-II log-likelihood criterion. See reference:
        https://www.biorxiv.org/content/10.1101/2020.08.10.243774v4.full.pdf
        """
        cov_Y_val = np.cov(Y)  # (n_samples, n_samples)
        Gamma = self._compute_groupwise_var(W)
        sigma_Y_train = ((self.sigma ** 2) * np.eye(X.shape[0])
                         + (X * Gamma) @ X.T)
        # must be invertible as a covariance matrix
        sigma_Y_train_inv = np.linalg.inv(sigma_Y_train)
        trace = np.trace(cov_Y_val @ sigma_Y_train_inv)
        log_det = np.linalg.slogdet(sigma_Y_train)[1]
        return trace, log_det, trace + log_det

    def _fit_reweighted_with_grid(self, X, Y):
        _, n_features = X.shape
        _, n_tasks = Y.shape
        n_alphas = len(self.alpha_grid)

        coef_0 = np.empty((n_alphas, n_features, n_tasks))

        # Warm start first iteration
        if self.n_orient == 1:
            regressor = MultiTaskLasso(np.nan, fit_intercept=False,
                                       warm_start=True)
        else:
            regressor = MultiTaskLassoUnscaled(np.nan, warm_start=True,
                                               n_orient=self.n_orient,
                                               accelerated=True)

        # Copy grid of first iteration (leverages convexity)
        print("First iteration")
        for j, alpha in tqdm(enumerate(self.alpha_grid), total=self.n_alphas):
            regressor.alpha = alpha

            if self.n_orient == 1:
                coef_0[j] = regressor.fit(X, Y).coef_.T
            else:
                coef_0[j] = regressor.fit(X, Y).coef_

        regressor.warm_start = False

        coefs_ = coef_0.copy()

        print("Next iterations...")
        for j, alpha in tqdm(enumerate(self.alpha_grid), total=self.n_alphas):
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
        """
        Computes the group-wise variance along the temporal axis (axis=-1)
        of the coefficient matrix.
        The groups are typically blocks of sources (free orientation).
        """
        n_positions = X.shape[0] // self.n_orient
        variances = np.zeros(X.shape[0], dtype=np.float32)

        for j in range(n_positions):
            idx = slice(j * self.n_orient, (j + 1) * self.n_orient)
            variances[idx] = np.var(X[idx, :])
        return variances

    def _penalty(self, coef):
        """Defines a non-convex penalty for reweighting
        the design matrix from the regression coefficients.

        Takes into account the number of orientations
        of the problem.

        Parameters
        ----------
        coef : array of shape (n_features, n_times)
            Coefficient matrix.

        Returns
        -------
        penalty : array of shape (n_positions,)
            Penalty vector.
        """
        n_positions = coef.shape[0] // self.n_orient
        coef = coef.reshape(n_positions, -1)
        m_norm = np.sqrt(norm(coef, axis=1))
        return 1 / (2 * m_norm + np.finfo(float).eps)
