import pytest
import numpy as np

from celer import MultiTaskLasso

from calibromatic.utils import norm_l2_inf, simulate_data
from calibromatic.mixed_norm import NormalizedMixedNorm

n_orients = [1, 3]
tol = 1e-10

X, Y = simulate_data(n_samples=10, n_features=15, n_tasks=7, nnz=3, random_state=0)


@pytest.mark.parametrize("n_orient", n_orients)
def test_loss_decreasing_every_iteration(n_orient):
    alpha_max = norm_l2_inf(X.T @ Y, n_orient, copy=False)
    alpha = alpha_max * 0.1
    estimator = NormalizedMixedNorm(alpha, n_orient)
    estimator.fit(X, Y)
    diffs = np.diff(estimator.gap_history_)
    assert np.all(diffs < 1e-3)


def test_mixed_norm():
    alpha_max = norm_l2_inf(X.T @ Y, 1, copy=False)
    alpha = alpha_max * 0.1
    clf = NormalizedMixedNorm(alpha, 1, tol=tol)
    clf_cel = MultiTaskLasso(alpha / len(X), fit_intercept=False, tol=tol)
    clf.fit(X, Y); clf_cel.fit(X, Y)
    np.testing.assert_allclose(clf.coef_, clf_cel.coef_.T, rtol=1e-5)
