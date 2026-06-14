#include "risk/stress_test.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace risk {

namespace {
// Effective shock for one asset: direct asset shock plus factor contributions.
double effective_shock(const std::string& asset,
                       const StressScenario& scenario,
                       const FactorBetas& betas) {
  double s = 0.0;
  auto it = scenario.asset_shocks.find(asset);
  if (it != scenario.asset_shocks.end()) s += it->second;

  if (!scenario.factor_shocks.empty()) {
    auto bit = betas.find(asset);
    if (bit != betas.end()) {
      for (const auto& [factor, shock] : scenario.factor_shocks) {
        auto fit = bit->second.find(factor);
        if (fit != bit->second.end()) s += fit->second * shock;
      }
    }
  }
  return s;
}
}  // namespace

StressResult apply_scenario(const Portfolio& portfolio,
                            const StressScenario& scenario,
                            const FactorBetas& betas) {
  const auto& names = portfolio.names();
  const auto& w = portfolio.weights();
  const double notional = portfolio.notional();
  const Eigen::Index n = portfolio.size();

  StressResult res;
  res.name = scenario.name;
  res.asset_names = names;
  res.asset_shock.resize(n);
  res.asset_pnl.resize(n);

  double pct = 0.0;
  for (Eigen::Index i = 0; i < n; ++i) {
    const double s = effective_shock(names[static_cast<std::size_t>(i)],
                                     scenario, betas);
    res.asset_shock(i) = s;
    // First-order P&L: position value * shock. weight_i * notional = position.
    res.asset_pnl(i) = notional * w(i) * s;
    pct += w(i) * s;
  }
  res.pct_pnl = pct;
  res.dollar_pnl = notional * pct;
  return res;
}

std::vector<StressResult> apply_scenarios(
    const Portfolio& portfolio, const std::vector<StressScenario>& scenarios,
    const FactorBetas& betas) {
  std::vector<StressResult> out;
  out.reserve(scenarios.size());
  for (const auto& sc : scenarios) out.push_back(apply_scenario(portfolio, sc, betas));
  return out;
}

std::vector<StressScenario> load_scenarios(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::invalid_argument("load_scenarios: cannot open " + path);
  }
  nlohmann::json j;
  in >> j;
  if (!j.contains("scenarios") || !j["scenarios"].is_array()) {
    throw std::invalid_argument(
        "load_scenarios: missing top-level \"scenarios\" array in " + path);
  }

  std::vector<StressScenario> scenarios;
  for (const auto& js : j["scenarios"]) {
    StressScenario sc;
    sc.name = js.value("name", std::string{});
    sc.type = js.value("type", std::string{});
    sc.description = js.value("description", std::string{});
    if (js.contains("asset_shocks")) {
      for (auto& [k, v] : js["asset_shocks"].items())
        sc.asset_shocks[k] = v.get<double>();
    }
    if (js.contains("factor_shocks")) {
      for (auto& [k, v] : js["factor_shocks"].items())
        sc.factor_shocks[k] = v.get<double>();
    }
    if (sc.name.empty()) {
      throw std::invalid_argument("load_scenarios: a scenario is missing \"name\"");
    }
    scenarios.push_back(std::move(sc));
  }
  return scenarios;
}

FactorBetas load_factor_betas(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::invalid_argument("load_factor_betas: cannot open " + path);
  }
  nlohmann::json j;
  in >> j;
  FactorBetas betas;
  for (auto& [asset, factors] : j.items()) {
    for (auto& [factor, beta] : factors.items())
      betas[asset][factor] = beta.get<double>();
  }
  return betas;
}

}  // namespace risk
