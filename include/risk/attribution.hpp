#pragma once

#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

#include "risk/portfolio.hpp"

namespace risk {

// Portfolio concentration metrics derived from the percentage risk shares.
struct ConcentrationMetrics {
  double herfindahl = 0.0;          // sum_i pct_i^2  (risk-share HHI)
  double effective_num_bets = 0.0;  // 1 / HHI; n for n equal independent bets
  double max_contribution = 0.0;    // largest single-name pct risk share
  std::string max_contributor;      // its asset name
};

// Full risk attribution for a portfolio under a given covariance.
struct RiskAttribution {
  std::vector<std::string> names;
  Eigen::VectorXd weights;
  Eigen::VectorXd marginal;    // MCR_i = (Sigma w)_i / sigma_p
  Eigen::VectorXd component;   // CCR_i = w_i * MCR_i  (sums to sigma_p)
  Eigen::VectorXd percent;     // CCR_i / sigma_p      (sums to 1)
  double portfolio_vol = 0.0;  // sigma_p

  ConcentrationMetrics concentration;

  // Optional sector roll-up (empty if no sector metadata supplied). Each maps
  // sector label -> summed component / percent risk contribution.
  std::map<std::string, double> sector_component;
  std::map<std::string, double> sector_percent;
};

// Compute the attribution. `sectors` maps asset name -> sector label; pass an
// empty map to skip the sector roll-up.
RiskAttribution compute_attribution(
    const Portfolio& portfolio, const Eigen::MatrixXd& cov,
    const std::map<std::string, std::string>& sectors = {});

}  // namespace risk
