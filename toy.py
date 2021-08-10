import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from hp_selection.utils import compute_alpha_max
from hp_selection.ll_warm_start import LLForReweightedMTL

random_state = 0
corr = 0.99
n_samples = 30
n_features = 90
n_tasks = 15
nnz = 2
snr = 2

grid_length = 15

n_orient = 3

rng = np.random.RandomState(random_state)
sigma = np.sqrt(1 - corr ** 2)
U = rng.randn(n_samples)

X = np.empty([n_samples, n_features], order="F")
X[:, 0] = U
for j in range(1, n_features):
    U *= corr
    U += sigma * rng.randn(n_samples)
    X[:, j] = U

support = rng.choice(n_features, nnz, replace=False)
W = np.zeros((n_features, n_tasks))

for k in support:
    W[k, :] = rng.normal(size=(n_tasks))

Y = np.dot(X, W)

noise = rng.randn(n_samples, n_tasks)
sigma = 1 / norm(noise) * norm(Y) / snr

Y += sigma * noise


alpha_max = compute_alpha_max(X, Y, n_orient)
grid = np.geomspace(alpha_max, alpha_max / 10, grid_length)
criterion = LLForReweightedMTL(1, grid, n_orient=n_orient, random_state=0)
best_alpha = criterion.get_val(X, Y)[1]


fig = plt.figure()
plt.plot(grid, criterion.ll_path_, label="LL")
plt.plot(grid, criterion.log_det_path_, label="Log det")
plt.plot(grid, criterion.trace_path_, label="Trace")
plt.legend()
plt.title("LL - Simulated")
fig.show()
