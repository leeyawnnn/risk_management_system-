#pragma once

#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

#include "risk/portfolio.hpp"

namespace risk {

// Per-asset factor sensitivities: betas[asset][factor] = d(asset return) per
// unit move of `factor`. Used to translate factor shocks into asset shocks.
using FactorBetas = std::map<std::string, std::map<std::string, double>>;

// A stress scenario. Shocks can be specified two (combinable) ways:
//   * asset_shocks[asset]  = a direct return shock applied to that asset.
//   * factor_shocks[factor] = a factor move, translated to asset shocks via the
//                             supplied FactorBetas.
// The effective shock for asset i is:
//     s_i = asset_shocks[i] + sum_f betas[i][f] * factor_shocks[f]
struct StressScenario {
  std::string name;
  std::string type;  // "historical" | "synthetic" (informational)
  std::string description;
  std::map<std::string, double> asset_shocks;
  std::map<std::string, double> factor_shocks;
};

// Outcome of applying one scenario to a portfolio.
struct StressResult {
  std::string name;
  double pct_pnl = 0.0;                  // P&L as a fraction of notional
  double dollar_pnl = 0.0;               // pct_pnl * notional
  std::vector<std::string> asset_names;  // portfolio order
  Eigen::VectorXd asset_shock;           // effective per-asset shock
  Eigen::VectorXd asset_pnl;  // per-asset dollar P&L (sums to dollar_pnl)
};

// Apply a single scenario. `betas` is only needed for factor_shocks; pass an
// empty map for pure asset-shock scenarios. Assets in the scenario but not in
// the portfolio are ignored; portfolio assets absent from the scenario take a
// zero shock.
StressResult apply_scenario(const Portfolio& portfolio,
                            const StressScenario& scenario,
                            const FactorBetas& betas = {});

// Apply many scenarios.
std::vector<StressResult> apply_scenarios(
    const Portfolio& portfolio, const std::vector<StressScenario>& scenarios,
    const FactorBetas& betas = {});

// Load scenarios from a JSON file with a top-level "scenarios" array.
std::vector<StressScenario> load_scenarios(const std::string& path);

// Load factor betas from the JSON written by scripts/estimate_factor_betas.py:
// { "_meta": {...}, "betas": { "ASSET": { "factor": beta, ... } }, ... }.
// A bare { "ASSET": { "factor": beta } } object is also accepted.
FactorBetas load_factor_betas(const std::string& path);

// ===========================================================================
// Reverse stress testing
// ===========================================================================
// Forward stress asks "what does this scenario cost?". Reverse stress asks the
// question a board actually cares about: "what would it take to lose this
// much, and how far-fetched is it?"
//
// Among all factor moves f that produce exactly the target loss, we want the
// most plausible one -- the one closest to the centre of the factor
// distribution. Minimising the Mahalanobis distance f' Sigma_F^-1 f subject to
// b'f = -L, where b = B'w are the portfolio's factor exposures, has the
// closed-form solution
//
//     f* = -L * Sigma_F b / (b' Sigma_F b)
//
// and the distance is L / sqrt(b' Sigma_F b): the loss measured in units of
// the book's own factor-model volatility. That number is the answer. A
// scenario at 2.5 sigma is a bad quarter; one at 9 sigma means the loss cannot
// be reached through the factors and something outside the model has to break.
struct ReverseStressResult {
  double target_loss_pct = 0.0;  // as a positive fraction of notional
  double target_loss_dollar = 0.0;
  std::vector<std::string> factors;
  Eigen::VectorXd factor_move;  // f*, the most likely move reaching the loss
  Eigen::VectorXd asset_shock;  // B f*, the implied per-asset shock
  std::vector<std::string> asset_names;
  double mahalanobis_distance = 0.0;  // in standard deviations
  double factor_model_vol = 0.0;      // sqrt(b' Sigma_F b), one period
  // One-sided Gaussian probability of a move at least this extreme. This is a
  // model-implied number under a Gaussian factor assumption, and real factor
  // distributions have fatter tails, so treat it as a lower bound on the
  // likelihood rather than a forecast.
  double gaussian_probability = 0.0;
};

// `target_loss_pct` is a positive fraction of notional (0.10 for "lose 10%").
ReverseStressResult reverse_stress(const Portfolio& portfolio,
                                   const FactorBetas& betas,
                                   const std::vector<std::string>& factors,
                                   const Eigen::MatrixXd& factor_cov,
                                   double target_loss_pct);

}  // namespace risk
