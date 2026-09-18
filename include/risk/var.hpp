#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace risk {

// ===========================================================================
// Sign / units convention (used across VaR and CVaR)
// ===========================================================================
// VaR is reported as a POSITIVE loss magnitude expressed as a fraction of
// portfolio value. A 1-day 95% VaR of 0.021 means "we expect to lose no more
// than 2.1% of the book on 95% of days; on the worst 5% of days the loss is at
// least 2.1%". Multiply by notional for a dollar figure.
//
// Horizon scaling:
//   * Historical and Parametric scale a 1-day figure by sqrt(horizon)
//     (the square-root-of-time rule; assumes i.i.d. zero-drift increments).
//   * Monte Carlo simulates the full multi-period path, so no extra scaling.
// Every annualization/horizon factor is explicit at the call site.

// ---------------------------------------------------------------------------
// Standard normal helpers
// ---------------------------------------------------------------------------
double normal_pdf(double x);  // phi(x), standard normal density
double normal_cdf(double x);  // Phi(x)
double normal_ppf(double p);  // Phi^{-1}(p), p in (0,1); z_alpha

// ---------------------------------------------------------------------------
// Empirical quantile (no interpolation): the k-th order statistic with
// k = floor(q * n), clamped to [0, n-1]. Uses std::nth_element, not a full
// sort. `data` is partitioned in place (pass a copy if you need it intact).
// ---------------------------------------------------------------------------
double empirical_quantile(std::vector<double>& data, double q);

// ---------------------------------------------------------------------------
// Historical VaR
// ---------------------------------------------------------------------------
// nth-percentile loss of the historical portfolio return sample. confidence is
// e.g. 0.95 or 0.99. The 1-day figure is sqrt-time scaled to `horizon_days`.
double historical_var(const Eigen::VectorXd& portfolio_returns,
                      double confidence, int horizon_days = 1);

// ---------------------------------------------------------------------------
// Parametric (Gaussian) VaR
// ---------------------------------------------------------------------------
// VaR = z_alpha * sigma - mu  (1-day), then mu -> mu*h, sigma -> sigma*sqrt(h).
// Inputs are the 1-day mean and standard deviation of portfolio returns.
double parametric_var(double mean, double stdev, double confidence,
                      int horizon_days = 1);

// ---------------------------------------------------------------------------
// Monte Carlo VaR
// ---------------------------------------------------------------------------
// Simulate `draws` portfolio returns over `horizon_days` from a multivariate
// normal N(mean, cov) using the Cholesky factor of cov, then take the empirical
// quantile. `mean` and `cov` are 1-day asset moments; `weights` the book.
Eigen::VectorXd simulate_portfolio_returns(const Eigen::VectorXd& mean,
                                           const Eigen::MatrixXd& cov,
                                           const Eigen::VectorXd& weights,
                                           int horizon_days, int draws,
                                           std::uint64_t seed = 42);

double monte_carlo_var(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov,
                       const Eigen::VectorXd& weights, double confidence,
                       int horizon_days = 1, int draws = 100000,
                       std::uint64_t seed = 42);

}  // namespace risk
