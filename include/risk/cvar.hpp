#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace risk {

// ===========================================================================
// CVaR / Expected Shortfall (ES)
// ===========================================================================
// CVaR is the AVERAGE loss in the tail beyond VaR: the mean of the worst
// (1 - confidence) fraction of outcomes. Like VaR it is reported as a POSITIVE
// loss fraction, and by construction CVaR >= VaR in magnitude (the tail mean is
// at least as deep as the tail quantile). Horizon scaling matches the VaR
// layer: Historical & Parametric sqrt-time scale the 1-day figure; Monte Carlo
// simulates the full path.

// Mean of the worst `tail_prob` fraction of `data` (the smallest values). Uses
// std::nth_element to isolate the tail, then averages it. At least one
// observation (the single worst) is always included. `data` is partitioned in
// place. Returns a value in the same units as the data (typically negative).
double tail_mean(std::vector<double>& data, double tail_prob);

// Historical CVaR: mean of portfolio returns at or below the (1-confidence)
// quantile, as a positive loss, sqrt-time scaled to the horizon.
double historical_cvar(const Eigen::VectorXd& portfolio_returns,
                       double confidence, int horizon_days = 1);

// Parametric (Gaussian) CVaR, closed form:
//   ES = -mu + sigma * phi(z_alpha) / (1 - confidence)
// with mu -> mu*h and sigma -> sigma*sqrt(h) for the horizon.
double parametric_cvar(double mean, double stdev, double confidence,
                       int horizon_days = 1);

// Monte Carlo CVaR: tail mean of simulated portfolio returns (Cholesky-based
// multivariate normal, multi-period). `mean`/`cov` are 1-day asset moments.
double monte_carlo_cvar(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov,
                        const Eigen::VectorXd& weights, double confidence,
                        int horizon_days = 1, int draws = 100000,
                        std::uint64_t seed = 42);

}  // namespace risk
