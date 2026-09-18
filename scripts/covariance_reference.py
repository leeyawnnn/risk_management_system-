#!/usr/bin/env python3
"""Reference covariance values for the C++ test cross-check.

Produces a small fixed synthetic return matrix X (T x N) and computes:
  - numpy.cov sample covariance (unbiased, ddof=1)
  - RiskMetrics EWMA covariance (lambda = 0.94)
  - Ledoit-Wolf constant-correlation shrinkage (our own reference impl)

Run this to regenerate the literals hardcoded in tests/test_covariance.cpp.
The dataset is intentionally tiny and deterministic so the test is exact.
"""

import numpy as np

np.set_printoptions(precision=15, suppress=False)

# Fixed synthetic return matrix: T = 8 observations, N = 3 assets.
X = np.array(
    [
        [0.012, -0.004, 0.006],
        [-0.008, 0.011, -0.002],
        [0.015, -0.009, 0.004],
        [0.003, 0.002, -0.007],
        [-0.011, 0.006, 0.010],
        [0.007, -0.003, -0.001],
        [-0.005, 0.008, 0.003],
        [0.010, -0.006, -0.004],
    ]
)
T, N = X.shape

# --- Sample covariance (numpy, ddof=1) ---
S = np.cov(X, rowvar=False, ddof=1)
print("# sample_covariance (numpy ddof=1):")
print(repr(S))

# --- EWMA covariance (lambda = 0.94), weighted about the weighted mean ---
lam = 0.94
ages = np.arange(T - 1, -1, -1)  # oldest..newest -> age T-1..0
w = lam**ages
w /= w.sum()
wmean = w @ X
Xc = X - wmean
EW = Xc.T @ np.diag(w) @ Xc
print("\n# ewma_covariance (lambda=0.94):")
print(repr(EW))

# --- Ledoit-Wolf constant-correlation shrinkage (reference impl) ---
mean = X.mean(axis=0)
Xc = X - mean
S_mle = (Xc.T @ Xc) / T
var = np.diag(S_mle)
sd = np.sqrt(var)

# average correlation r-bar
pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
rbar = np.mean([S_mle[i, j] / (sd[i] * sd[j]) for i, j in pairs])

F = rbar * np.outer(sd, sd)
np.fill_diagonal(F, var)

# pi
pi_mat = np.zeros((N, N))
for t in range(T):
    d = np.outer(Xc[t], Xc[t]) - S_mle
    pi_mat += d * d
pi_mat /= T
pi_hat = pi_mat.sum()

# rho
rho = np.trace(pi_mat)
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        theta_ii = np.mean(
            (Xc[:, i] ** 2 - S_mle[i, i]) * (Xc[:, i] * Xc[:, j] - S_mle[i, j])
        )
        theta_jj = np.mean(
            (Xc[:, j] ** 2 - S_mle[j, j]) * (Xc[:, i] * Xc[:, j] - S_mle[i, j])
        )
        rho += rbar * 0.5 * (sd[j] / sd[i] * theta_ii + sd[i] / sd[j] * theta_jj)

gamma = np.sum((F - S_mle) ** 2)
delta = max(0.0, min(1.0, (pi_hat - rho) / gamma / T))
LW = delta * F + (1 - delta) * S_mle
print("\n# ledoit_wolf delta* =", delta)
print("# ledoit_wolf rbar   =", rbar)
print("# ledoit_wolf_covariance:")
print(repr(LW))
