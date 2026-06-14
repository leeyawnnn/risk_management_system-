#pragma once

#include <Eigen/Dense>
#include <vector>

#include "risk/return_series.hpp"

namespace risk {

// ---------------------------------------------------------------------------
// Matrix sanity checks (used in asserts and in tests).
// ---------------------------------------------------------------------------

// True if M is square and symmetric to within `tol` (max abs of M - M^T).
bool is_symmetric(const Eigen::MatrixXd& M, double tol = 1e-10);

// True if M is symmetric and positive-semi-definite: its smallest eigenvalue
// is >= -tol. A tiny negative tolerance absorbs floating-point round-off; a
// genuinely indefinite matrix (e.g. a corrupted covariance) fails.
bool is_psd(const Eigen::MatrixXd& M, double tol = 1e-10);

// ---------------------------------------------------------------------------
// Data assembly
// ---------------------------------------------------------------------------

// Stack per-asset return series into a (T x N) matrix: one row per time step,
// one column per asset. All series must have identical length, else throws.
Eigen::MatrixXd to_return_matrix(const std::vector<ReturnSeries>& series);

// ---------------------------------------------------------------------------
// Covariance estimators. Input X is (T x N): rows = observations, cols = assets.
// All outputs are daily covariances (annualize by multiplying by the factor).
// Each estimator asserts symmetry + PSD in debug builds before returning.
// ---------------------------------------------------------------------------

// Unbiased sample covariance: demean columns, divide by (T-1).
// Matches numpy.cov(X, rowvar=False). Requires T >= 2.
Eigen::MatrixXd sample_covariance(const Eigen::MatrixXd& X);

// EWMA weights for T observations under RiskMetrics decay `lambda`.
// Element t (0 = oldest, T-1 = most recent) is proportional to lambda^(age),
// normalized to sum to 1. The most recent observation carries the most weight.
Eigen::VectorXd ewma_weights(Eigen::Index T, double lambda);

// Exponentially-weighted covariance (RiskMetrics). `lambda` default 0.94 is the
// RiskMetrics daily decay. Returns a weighted covariance about the weighted
// mean; PSD by construction (weighted sum of outer products). Requires T >= 2.
Eigen::MatrixXd ewma_covariance(const Eigen::MatrixXd& X, double lambda = 0.94);

// Result of Ledoit-Wolf shrinkage toward a constant-correlation target.
struct LedoitWolf {
  Eigen::MatrixXd cov;   // the shrunk covariance estimate
  double shrinkage;      // optimal intensity delta* in [0, 1]
  double avg_correlation;  // r-bar, the constant correlation of the target
};

// Ledoit-Wolf (2004) shrinkage toward the constant-correlation target F:
//   Sigma_hat = delta* * F + (1 - delta*) * S
// where S is the (1/T) sample covariance and F shares S's variances but uses a
// single average correlation for every off-diagonal. delta* is estimated
// optimally (clipped to [0,1]). Requires T >= 2 and N >= 2.
LedoitWolf ledoit_wolf_covariance(const Eigen::MatrixXd& X);

// ---------------------------------------------------------------------------
// Correlation
// ---------------------------------------------------------------------------

// Convert a covariance matrix to a correlation matrix (unit diagonal).
Eigen::MatrixXd cov_to_correlation(const Eigen::MatrixXd& cov);

}  // namespace risk
