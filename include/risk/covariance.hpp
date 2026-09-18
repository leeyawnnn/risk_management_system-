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
// Conditioning
// ---------------------------------------------------------------------------

// The eigen-spectrum of a covariance or correlation matrix, and what it says
// about whether the matrix can safely be inverted or factorised.
//
// PSD is necessary but nowhere near sufficient. A matrix can pass is_psd and
// still be numerically useless: with eleven instruments including three
// highly correlated equity indices and three points on one yield curve, the
// sample correlation matrix here has a condition number in the hundreds. Any
// operation that effectively inverts Sigma -- optimisation, a Cholesky factor
// used for simulation, Mahalanobis distance in reverse stress -- amplifies
// estimation error in the smallest eigenvalue by that factor.
//
// This is the concrete reason shrinkage exists, and reporting the number is
// how a reader can see it rather than take it on faith.
struct MatrixDiagnostics {
  double min_eigenvalue = 0.0;
  double max_eigenvalue = 0.0;
  double condition_number = 0.0;  // max / min; infinity if min <= 0
  bool symmetric = false;
  bool psd = false;
  // Above this the matrix should not be inverted without regularisation.
  // 1e8 is roughly where double precision starts losing half its digits.
  bool ill_conditioned = false;
};

MatrixDiagnostics diagnose_matrix(const Eigen::MatrixXd& M,
                                  double condition_threshold = 1e8);

// ---------------------------------------------------------------------------
// Data assembly
// ---------------------------------------------------------------------------

// Stack per-asset return series into a (T x N) matrix: one row per time step,
// one column per asset. All series must have identical length, else throws.
Eigen::MatrixXd to_return_matrix(const std::vector<ReturnSeries>& series);

// ---------------------------------------------------------------------------
// Covariance estimators. Input X is (T x N): rows = observations, cols =
// assets. All outputs are daily covariances (annualize by multiplying by the
// factor). Each estimator asserts symmetry + PSD in debug builds before
// returning.
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
  Eigen::MatrixXd cov;     // the shrunk covariance estimate
  double shrinkage;        // optimal intensity delta* in [0, 1]
  double avg_correlation;  // r-bar, the constant correlation of the target
};

// Ledoit-Wolf shrinkage toward the constant-correlation target F.
//
// Reference: Olivier Ledoit and Michael Wolf, "Honey, I Shrunk the Sample
// Covariance Matrix", Journal of Portfolio Management 30(4), 2004, 110-119.
// The estimator is their equation (2),
//
//     Sigma_shrink = delta* F + (1 - delta*) S
//
// and the intensity is equation (5) of Appendix B,
//
//     delta* = max{0, min{kappa/T, 1}},   kappa = (pi - rho) / gamma
//
// with pi the summed asymptotic variances of the sample covariance entries,
// gamma = sum_ij (f_ij - s_ij)^2 the squared Frobenius distance from target to
// sample, and rho the summed asymptotic covariances between the two. S here is
// the maximum-likelihood (1/T) sample covariance the paper uses, not the
// unbiased 1/(T-1) one returned by sample_covariance().
//
// tests/test_covariance_reference.cpp recomputes pi, rho and gamma from those
// definitions and checks they reproduce the intensity this function returns.
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
