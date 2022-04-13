from tqdm import tqdm

import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state, check_X_y

from celer import MultiTaskLasso

from ..sparse_solver import NormalizedMixedNorm


class MCFD_SURE:
    r"""Monte-Carlo Finite Difference (MCFD) Stein's Unbiased Risk Estimate (SURE).

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

    n_orient : int, optional
        Number of orientation for a dipole. 1 for fixed orientation, > 1 for free.

    random_state : int or None, optional
        Seed for reproducible experiments.

    Attributes
    ----------
    sure_path : array, shape (n_alphas,)
        The SURE value along the path.

    dof_path_ : array, shape (n_alphas,)
        The degree of freedom contribution to SURE along the path.

    df_path_ : array, shape (n_alphas,)
        The data fitting contribution to SURE along the path.

    References
    ----------

    """
    def __init__(self, sigma, alpha_grid, n_reweighting=5, penalty=None, n_orient=1,
                 random_state=None):
        self.sigma = sigma
        self.alpha_grid = alpha_grid
        self.n_reweighting = n_reweighting
        self.n_orient = n_orient
        self.random_state = random_state

        self.n_alphas = len(self.alpha_grid)

        self.sure_path_ = np.empty(self.n_alphas)
        self.dof_path_ = np.empty(self.n_alphas)
        self.df_path_ = np.empty(self.n_alphas)

        self.eps = None
        self.delta = None

        if self.n_orient <= 0:
            raise ValueError("Number of orientations can't be negative.")

        self.penalty = penalty if penalty else self._penalty

    def fit(self, X, Y):
        """Fit SURE to select best `alpha`.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        Y : array, shape (n_samples, n_tasks)
            Target matrix.
        """
        n_samples, n_tasks = Y.shape

        if self.eps is None or self.delta is None:
            self._init_eps_and_delta(n_samples, n_tasks)

        X, Y = check_X_y(X, Y, multi_output=True)

        coefs_grid_1, coefs_grid_2 = self._fit_reweighted_with_grid(X, Y)

        for i, (coef1, coef2) in enumerate(zip(coefs_grid_1, coefs_grid_2)):
            sure_val, dof_term, data_fitting_term = self._compute_sure_val(coef1, coef2,
                                                                           X, Y)
            self.sure_path_[i] = sure_val
            self.dof_path_[i] = dof_term
            self.df_path_[i] = data_fitting_term

        best_sure_ = np.min(self.sure_path_)
        best_alpha_ = self.alpha_grid[np.argmin(self.sure_path_)]
        return best_sure_, best_alpha_

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
            coef = (regressor.coef_.T / np.repeat(w[np.newaxis, :], self.n_orient)).T
        w = self.penalty(coef)
        return coef, w

    def _compute_sure_val(self, coef1, coef2, X, Y):
        """Evaluate the DOF term and compute the MCFD SURE value.
        
        Parameters
        ----------
        coef1 : array, shape (n_features, n_tasks)
            The coefficient matrix.
        
        coef2 : array, shape (n_features, n_tasks)
            The noise-disturbed coefficient matrix.
        
        X : array, shape (n_samples, n_features)
            The design matrix.
        
        Y : array, shape (n_samples, n_tasks)
            The measurement matrix.
        
        Returns
        -------
        sure : float
            The SURE approximation using Monte-Carlo Finite Difference.
        
        dof : float
            The degrees of freedom term.
        
        df_term : float
            The data fitting term.
        """
        n_samples, n_tasks = X.shape[0], Y.shape[1]

        # Note: Celer returns the transpose of the coefficient matrix
        if coef1.shape[0] != X.shape[1]:
            coef1 = coef1.T
            coef2 = coef2.T

        # dof
        dof = ((X @ (coef2 - coef1)) * self.delta).sum() / self.eps
        # SURE
        df_term = norm(Y - X @ coef1) ** 2
        sure = df_term - n_samples * n_tasks * self.sigma ** 2
        sure += 2 * dof * self.sigma ** 2

        return sure, dof, df_term

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
        _, n_features = X.shape
        _, n_tasks = Y.shape
        n_alphas = len(self.alpha_grid)

        coef1_0 = np.empty((n_alphas, n_features, n_tasks))
        coef2_0 = np.empty((n_alphas, n_features, n_tasks))

        Y_eps = Y + self.eps * self.delta

        # Warm start first iteration
        if self.n_orient == 1:
            regressor1 = MultiTaskLasso(np.nan, fit_intercept=False, warm_start=True)
            regressor2 = MultiTaskLasso(np.nan, fit_intercept=False, warm_start=True)
        else:
            regressor1 = NormalizedMixedNorm(np.nan, warm_start=True,
                                             n_orient=self.n_orient)
            regressor2 = NormalizedMixedNorm(np.nan, warm_start=True,
                                             n_orient=self.n_orient)

        # Copy grid of first iteration (leverages convexity)
        print("First iteration")
        for j, alpha in tqdm(enumerate(self.alpha_grid), total=self.n_alphas):
            regressor1.alpha = alpha
            regressor2.alpha = alpha
            if self.n_orient == 1:
                coef1_0[j] = regressor1.fit(X, Y).coef_.T
                coef2_0[j] = regressor2.fit(X, Y_eps).coef_.T
            else:
                coef1_0[j] = regressor1.fit(X, Y).coef_
                coef2_0[j] = regressor2.fit(X, Y_eps).coef_

        regressor1.warm_start = False
        regressor2.warm_start = False

        coefs_1_ = coef1_0.copy()
        coefs_2_ = coef2_0.copy()

        print("Next iterations...")
        for j, alpha in tqdm(enumerate(self.alpha_grid), total=self.n_alphas):
            regressor1.alpha = alpha
            regressor2.alpha = alpha

            w1 = self.penalty(coef1_0[j])
            w2 = self.penalty(coef2_0[j])

            for _ in range(self.n_iterations - 1):
                mask1 = w1 != 1.0 / np.finfo(float).eps
                mask2 = w2 != 1.0 / np.finfo(float).eps

                mask1_full = np.repeat(mask1, self.n_orient)
                mask2_full = np.repeat(mask2, self.n_orient)

                coefs_1_[j][~mask1_full] = 0.0
                coefs_2_[j][~mask2_full] = 0.0

                if mask1.sum():
                    coefs_1_[j][mask1_full], w1[mask1] = self._reweight_op(
                        regressor1, X[:, mask1_full], Y, w1[mask1])
                if mask2.sum():
                    coefs_2_[j][mask2_full], w2[mask2] = self._reweight_op(
                        regressor2, X[:, mask2_full], Y_eps, w2[mask2])

        return coefs_1_, coefs_2_

    def _init_eps_and_delta(self, n_samples, n_tasks):
        """Initializes epsilon and delta for DOF term computation.
        
        Parameters
        ----------
        n_samples : int
            The number of samples.
        
        n_tasks : int
            The number of tasks.
        """
        rng = check_random_state(self.random_state)
        self.eps = 2 * self.sigma / (n_samples ** 0.3)
        self.delta = rng.randn(n_samples, n_tasks)

    def _penalty(self, coef):
        """Defines a non-convex penalty for reweighting
        the design matrix from the regression coefficients.

        Takes into account the number of orientations
        of the problem.

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
