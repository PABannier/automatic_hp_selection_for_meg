import pytest
import numpy as np

from celer import MultiTaskLasso

from calibromatic.utils import norm_l2_inf, simulate_data
from calibromatic.mixed_norm import NormalizedMixedNorm

n_orients = [1, 3]
tol = 1e-10

n_samples = 30
n_features = 90
n_tasks = 10

X, Y = simulate_data(n_samples=n_samples, n_features=n_features, n_tasks=n_tasks,
                     nnz=3, random_state=0)[:2]


@pytest.mark.parametrize("n_orient", n_orients)
def test_gap_decreasing(n_orient):
    alpha_max = norm_l2_inf(X.T @ Y, n_orient, copy=False)
    alpha = alpha_max * 0.1
    clf = NormalizedMixedNorm(alpha, n_orient=n_orient)
    clf.fit(X, Y)
    XR = X.T @ (Y - X @ clf.coef_)

    assert np.all(np.abs(XR) <= alpha * len(X) + 1e-12), "KKT check"
    assert clf.gap_history_[0] >= clf.gap_history_[-1]
    assert clf.gap_history_[-1] <= tol
    assert np.all(np.diff(clf.gap_history_) <= 0)
    assert np.all(np.diff(clf.primal_history_) <= 0)


def test_mixed_norm():
    alpha_max = norm_l2_inf(X.T @ Y, 1, copy=False)
    alpha = alpha_max * 0.1
    clf = NormalizedMixedNorm(alpha, n_orient=1, tol=tol)
    clf_cel = MultiTaskLasso(alpha, fit_intercept=False, tol=tol)
    clf.fit(X, Y); clf_cel.fit(X, Y)
    np.testing.assert_allclose(clf.coef_, clf_cel.coef_.T, rtol=1e-5)
