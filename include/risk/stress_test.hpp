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
  std::string type;         // "historical" | "synthetic" (informational)
  std::string description;
  std::map<std::string, double> asset_shocks;
  std::map<std::string, double> factor_shocks;
};

// Outcome of applying one scenario to a portfolio.
struct StressResult {
  std::string name;
  double pct_pnl = 0.0;     // P&L as a fraction of notional
  double dollar_pnl = 0.0;  // pct_pnl * notional
  std::vector<std::string> asset_names;  // portfolio order
  Eigen::VectorXd asset_shock;           // effective per-asset shock
  Eigen::VectorXd asset_pnl;             // per-asset dollar P&L (sums to dollar_pnl)
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

// Load factor betas from a JSON file: { "ASSET": { "factor": beta, ... }, ... }.
FactorBetas load_factor_betas(const std::string& path);

}  // namespace risk
