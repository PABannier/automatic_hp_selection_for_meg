import numpy as np
from numpy.linalg import norm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y

from calibromatic.utils import (
    get_duality_gap_mtl, primal_mtl, groups_norm2, sum_squared, get_dgemm)


class MixedNorm(BaseEstimator, RegressorMixin):
    r"""Lasso solver for neuroscience inverse problem.

    The optimization objective for Lasso is::

    (1 / 2) * ||M - GX||^2_F + alpha * ||X||_2,1

    Parameters
    ----------
    alpha: float
        Regularization parameter.

    n_orient: int, optional
        Number of orientation for a dipole. 1 for fixed orientation, > 1 for free.

    max_iter: int, optional
        The maximum number of iterations for coordinate descent.

    tol: float, optional
        Stopping criterion for the optimization.

    p0: int, optional
        First active set size.

    warm_start: bool, optional (default=True)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    verbose: bool, optional (default=False)
        Verbosity.

    Attributes
    ----------
    coef_: array, shape (n_features, n_times)
        Parameter matrix.

    gap_history_: list
        The duality gap along the path.

    Notes
    -----
    This solver supports fixed and free dipole orientations.
    """

    def __init__(self, alpha, n_orient=3, max_iter=100, tol=1e-5, p0=100,
                 warm_start=True, verbose=False):
        super(MixedNorm, self).__init__()
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.n_orient = n_orient
        self.p0 = p0
        self.verbose = verbose

        self.gap_history_ = []
        self.primal_history_ = []
        self.coef_ = None
        self.active_set_ = None

        # Number of past iterates used to construct extrapolated point
        self.K = 5

    def fit(self, X, Y, alpha=None):
        """Compute Lasso fit.

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            Design matrix.

        Y: array, shape (n_samples, n_times)
            Target matrix.

        alpha: float, default=None
            Regularization parameter. If None, the solver uses self.alpha.
        """
        _alpha = alpha if alpha is not None else self.alpha
        X, Y = check_X_y(X, Y, multi_output=True)

        n_features, n_times = X.shape[1], Y.shape[1]
        n_positions = n_features // self.n_orient
        lipschitz = np.zeros(n_positions)

        for j in range(n_positions):
            idx = slice(j * self.n_orient, (j + 1) * self.n_orient)
            lipschitz[j] = norm(X[:, idx], ord=2) ** 2

        if self.active_set_ is None or not self.warm_start:
            active_set = np.zeros(n_features, dtype=bool)
        else:
            # Useful for warm starting active set
            active_set = self.active_set_

        idx_large_corr = np.argsort(groups_norm2(np.dot(X.T, Y), self.n_orient))
        new_active_idx = idx_large_corr[-self.p0:]

        if self.n_orient > 1:
            new_active_idx = (
                self.n_orient * new_active_idx[:, None]
                + np.arange(self.n_orient)[None, :]
            ).ravel()

        active_set[new_active_idx] = True
        as_size = np.sum(active_set)

        highest_d_obj = -np.inf

        if self.warm_start and self.coef_ is not None:
            if self.coef_.shape != (n_features, n_times):
                raise ValueError("Wrong dimension for initialized "
                                 "coefficients. Got %s. "
                                 "Expected {(%s, %s)}" %
                                 (self.coef_.shape, n_features, n_times))
            coef_init = self.coef_[active_set]
        else:
            coef_init = None

        for k in range(self.max_iter):
            lipschitz_ = lipschitz[active_set[:: self.n_orient]]
            coef, as_ = self._block_coordinate_descent(X[:, active_set], Y, lipschitz_,
                                                       coef_init, _alpha)
            active_set[active_set] = as_.copy()
            idx_old_active_set = np.where(active_set)[0]

            _, p_obj, d_obj = get_duality_gap_mtl(X, Y, coef, active_set, _alpha,
                                                  self.n_orient)
            highest_d_obj = max(highest_d_obj, d_obj)
            gap = p_obj - highest_d_obj

            self.gap_history_.append(gap)
            self.primal_history_.append(p_obj)

            if self.verbose:
                print(f"[{k+1}/{self.max_iter}] p_obj {p_obj:.5f} :: d_obj " +
                      f"{d_obj:.5f} :: d_gap {gap:.5f} :: n_active_start " +
                      f"{as_size // self.n_orient} :: n_active_end " +
                      f"{np.sum(active_set) // self.n_orient}")

            if gap < self.tol:
                if self.verbose:
                    print("Convergence reached!")
                break

            if k < (self.max_iter - 1):
                R = Y - X[:, active_set] @ coef
                idx_large_corr = np.argsort(groups_norm2(np.dot(X.T, R), self.n_orient))
                new_active_idx = idx_large_corr[-self.p0:]

                if self.n_orient > 1:
                    new_active_idx = (
                        self.n_orient * new_active_idx[:, None]
                        + np.arange(self.n_orient)[None, :]
                    )
                    new_active_idx = new_active_idx.ravel()

                active_set[new_active_idx] = True
                idx_active_set = np.where(active_set)[0]
                as_size = np.sum(active_set)
                coef_init = np.zeros((as_size, n_times), dtype=coef.dtype)
                idx = np.searchsorted(idx_active_set, idx_old_active_set)
                coef_init[idx] = coef

        # Building full coefficient matrix and filling active set with
        # non-zero coefficients
        final_coef_ = np.zeros((len(active_set), n_times))
        if coef is not None:
            final_coef_[active_set] = coef

        self.coef_ = final_coef_
        self.active_set_ = active_set

        return self

    def _block_coordinate_descent(self, X, Y, lipschitz, init, _alpha):
        """Solve subproblems restricted to active set.

        Parameters
        ----------
        X: array, (n_samples, n_features)
            Design matrix.

        Y: array, (n_samples, n_times)
            Target matrix.

        lipschitz: array, (ws_size,)
            Lipschitz constants for every block.

        init: array (ws_size,)
            Coefficient initialized from previous iteration.
            If None, coefficients are initialized with zeros.

        _alpha: float
            Regularization parameter.

        Returns
        -------
        coef: array, shape (n_features, n_times)
            Coefficient matrix.

        active_set: array, shape (n_features,)
            The active set.
        """
        n_times = Y.shape[1]
        n_features = X.shape[1]
        n_positions = n_features // self.n_orient

        if init is None:
            coef = np.zeros((n_features, n_times))
            R = Y.copy()
        else:
            coef = init
            R = Y - X @ coef

        X = np.asfortranarray(X)

        last_K_coef = np.empty((self.K + 1, n_features, n_times))
        U = np.zeros((self.K, n_features * n_times))

        highest_d_obj = -np.inf
        active_set = np.zeros(n_features, dtype=bool)

        for iter_idx in range(self.max_iter):
            coef_j_new = np.zeros_like(coef[: self.n_orient, :], order="C")
            # Call to Fortran BLAS subroutine
            dgemm = get_dgemm()

            for j in range(n_positions):
                idx = slice(j * self.n_orient, (j + 1) * self.n_orient)
                coef_j = coef[idx]
                X_j = X[:, idx]

                # coef_j_new = X_j.T @ R / L[j]
                dgemm(alpha=1 / lipschitz[j], beta=0.0, a=R.T, b=X_j,
                      c=coef_j_new.T, overwrite_c=True)

                if coef_j[0, 0] != 0:
                    # R += X_j @ coef_j
                    dgemm(alpha=1.0, beta=1.0, a=coef_j.T, b=X_j.T, c=R.T,
                          overwrite_c=True)
                    coef_j_new += coef_j

                block_norm = np.sqrt(sum_squared(coef_j_new))
                alpha_lc = _alpha / lipschitz[j]

                if block_norm <= alpha_lc:
                    coef_j.fill(0.0)
                    active_set[idx] = False
                else:
                    shrink = max(1.0 - alpha_lc / block_norm, 0.0)
                    coef_j_new *= shrink

                    # R -= np.dot(X_j, coef_j_new)
                    dgemm(alpha=-1.0, beta=1.0, a=coef_j_new.T, b=X_j.T, c=R.T,
                          overwrite_c=True)
                    coef_j[:] = coef_j_new
                    active_set[idx] = True

            _, p_obj, d_obj = get_duality_gap_mtl(X, Y, coef[active_set], active_set,
                                                  _alpha, self.n_orient)
            highest_d_obj = max(d_obj, highest_d_obj)
            gap = p_obj - highest_d_obj

            if self.verbose:
                print(f"[{iter_idx+1}/{self.max_iter}] p_obj {p_obj:.5f} :: "
                      + f"d_obj {d_obj:.5f} :: d_gap {gap:.5f}")

            last_K_coef[iter_idx % (self.K + 1)] = coef

            if iter_idx % (self.K + 1) == self.K:
                for k in range(self.K):
                    U[k] = last_K_coef[k + 1].ravel() - last_K_coef[k].ravel()

                C = U @ U.T

                try:
                    z = np.linalg.solve(C, np.ones(self.K))
                    # When C is ill-conditioned, z can take very large finite
                    # positive and negative values (1e35 and -1e35), which leads
                    # to z.sum() being null.
                    if z.sum() == 0:
                        raise np.linalg.LinAlgError
                except np.linalg.LinAlgError:
                    if self.verbose:
                        print("LinAlg Error")
                else:
                    c = z / z.sum()
                    coef_acc = np.sum(
                        last_K_coef[:-1] * c[:, None, None], axis=0)
                    active_set_acc = norm(coef_acc, axis=1) != 0
                    p_obj_acc = primal_mtl(X, Y, coef_acc[active_set_acc],
                                           active_set_acc, _alpha, self.n_orient)
                    if p_obj_acc < p_obj:
                        coef = coef_acc
                        active_set = active_set_acc
                        R = Y - X[:, active_set] @ coef[active_set]

            if gap < self.tol:
                if self.verbose:
                    print(f"Fitting ended after iteration {iter_idx + 1}.")
                break

        coef = coef[active_set]
        return coef, active_set


class NormalizedMixedNorm(MixedNorm):
    r"""Lasso solver for neuroscience inverse problem.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||M - GX||^2_F + alpha * ||X||_2,1

    See Also
    --------
    MixedNorm : Unscaled Lasso solver for neuroscience inverse problem.
    """

    def __init__(self, alpha, **kwargs):
        super(NormalizedMixedNorm, self).__init__(alpha, **kwargs)

    def fit(self, X, Y):
        alpha_scaled = self.alpha * len(X)
        super().fit(X, Y, alpha_scaled)
        return self
