#include "risk/stress_test.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "risk/var.hpp"

namespace risk {

namespace {
// Effective shock for one asset: direct asset shock plus factor contributions.
double effective_shock(const std::string& asset, const StressScenario& scenario,
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
    const double s =
        effective_shock(names[static_cast<std::size_t>(i)], scenario, betas);
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
  for (const auto& sc : scenarios)
    out.push_back(apply_scenario(portfolio, sc, betas));
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
      for (const auto& [k, v] : js["asset_shocks"].items())
        sc.asset_shocks[k] = v.get<double>();
    }
    if (js.contains("factor_shocks")) {
      for (const auto& [k, v] : js["factor_shocks"].items())
        sc.factor_shocks[k] = v.get<double>();
    }
    if (sc.name.empty()) {
      throw std::invalid_argument(
          "load_scenarios: a scenario is missing \"name\"");
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

  // The generated file nests the loadings under "betas" alongside "_meta" and
  // "diagnostics". Reading the top level directly would treat those two as
  // assets. A plain { asset: { factor: beta } } file still works.
  const nlohmann::json& src = j.contains("betas") ? j.at("betas") : j;

  FactorBetas betas;
  for (const auto& [asset, factors] : src.items()) {
    if (!factors.is_object()) continue;
    for (const auto& [factor, beta] : factors.items()) {
      if (beta.is_number()) betas[asset][factor] = beta.get<double>();
    }
  }
  return betas;
}

ReverseStressResult reverse_stress(const Portfolio& portfolio,
                                   const FactorBetas& betas,
                                   const std::vector<std::string>& factors,
                                   const Eigen::MatrixXd& factor_cov,
                                   double target_loss_pct) {
  const auto N = portfolio.size();
  const auto K = static_cast<Eigen::Index>(factors.size());
  if (factor_cov.rows() != K || factor_cov.cols() != K) {
    throw std::invalid_argument("reverse_stress: factor_cov must be K x K");
  }
  if (!(target_loss_pct > 0.0)) {
    throw std::invalid_argument("reverse_stress: target_loss_pct must be > 0");
  }

  const auto& names = portfolio.names();
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N, K);
  for (Eigen::Index i = 0; i < N; ++i) {
    auto ait = betas.find(names[static_cast<std::size_t>(i)]);
    if (ait == betas.end()) continue;
    for (Eigen::Index k = 0; k < K; ++k) {
      auto fit = ait->second.find(factors[static_cast<std::size_t>(k)]);
      if (fit != ait->second.end()) B(i, k) = fit->second;
    }
  }

  const Eigen::VectorXd b = B.transpose() * portfolio.weights();
  const Eigen::VectorXd sigma_b = factor_cov * b;
  const double denom = b.dot(sigma_b);

  ReverseStressResult r;
  r.factors = factors;
  r.asset_names = names;
  r.target_loss_pct = target_loss_pct;
  r.target_loss_dollar = target_loss_pct * portfolio.notional();

  if (!(denom > 0.0)) {
    // The book has no factor exposure at all, so no factor move reaches the
    // loss. Report that honestly instead of dividing by zero.
    r.factor_move = Eigen::VectorXd::Zero(K);
    r.asset_shock = Eigen::VectorXd::Zero(N);
    r.mahalanobis_distance = std::numeric_limits<double>::infinity();
    r.gaussian_probability = 0.0;
    return r;
  }

  r.factor_move = (-target_loss_pct / denom) * sigma_b;
  r.asset_shock = B * r.factor_move;
  r.factor_model_vol = std::sqrt(denom);
  r.mahalanobis_distance = target_loss_pct / r.factor_model_vol;
  r.gaussian_probability = normal_cdf(-r.mahalanobis_distance);
  return r;
}

}  // namespace risk
