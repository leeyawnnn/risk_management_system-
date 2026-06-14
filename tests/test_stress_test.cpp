#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>

#include "risk/stress_test.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace risk;

namespace {
std::string cfg(const std::string& name) {
  return std::string(RISK_TEST_DATA_DIR) + "/../../config/" + name;
}
Portfolio single_equity(double weight, double notional) {
  Eigen::VectorXd w(1);
  w << weight;
  return Portfolio({"SPY"}, w, notional);
}
}  // namespace

TEST_CASE("equity shock of -20% yields -20% * weight * notional", "[stress]") {
  // The headline test: single-equity portfolio, direct -20% asset shock.
  const double notional = 1'000'000.0;
  Portfolio p = single_equity(1.0, notional);

  StressScenario sc;
  sc.name = "Equity -20%";
  sc.asset_shocks = {{"SPY", -0.20}};

  StressResult r = apply_scenario(p, sc);
  CHECK_THAT(r.pct_pnl, WithinAbs(-0.20, 1e-12));
  CHECK_THAT(r.dollar_pnl, WithinAbs(-0.20 * 1.0 * notional, 1e-6));
  CHECK_THAT(r.asset_pnl(0), WithinAbs(-0.20 * 1.0 * notional, 1e-6));
}

TEST_CASE("shock scales with weight", "[stress]") {
  const double notional = 2'000'000.0;
  Portfolio p = single_equity(0.5, notional);  // half the book in SPY
  StressScenario sc{"e", "synthetic", "", {{"SPY", -0.20}}, {}};
  StressResult r = apply_scenario(p, sc);
  CHECK_THAT(r.dollar_pnl, WithinAbs(-0.20 * 0.5 * notional, 1e-6));
}

TEST_CASE("factor shock applied through beta equals direct equity shock",
          "[stress][factor]") {
  Portfolio p = single_equity(1.0, 1'000'000.0);
  FactorBetas betas = {{"SPY", {{"equity", 1.0}}}};
  StressScenario sc;
  sc.name = "Equity factor -20%";
  sc.factor_shocks = {{"equity", -0.20}};

  StressResult r = apply_scenario(p, sc, betas);
  CHECK_THAT(r.asset_shock(0), WithinAbs(-0.20, 1e-12));
  CHECK_THAT(r.pct_pnl, WithinAbs(-0.20, 1e-12));
}

TEST_CASE("multi-asset P&L is the weighted sum of shocks", "[stress]") {
  Eigen::VectorXd w(3);
  w << 0.6, 0.3, 0.1;
  Portfolio p({"SPY", "TLT", "GLD"}, w, 1'000'000.0);

  StressScenario sc;
  sc.name = "GFC-like";
  sc.asset_shocks = {{"SPY", -0.30}, {"TLT", 0.15}, {"GLD", 0.05}};

  StressResult r = apply_scenario(p, sc);
  const double expected = 0.6 * -0.30 + 0.3 * 0.15 + 0.1 * 0.05;
  CHECK_THAT(r.pct_pnl, WithinRel(expected, 1e-12));
  // Per-asset P&L sums to the total.
  CHECK_THAT(r.asset_pnl.sum(), WithinAbs(r.dollar_pnl, 1e-6));
}

TEST_CASE("assets absent from a scenario take zero shock", "[stress]") {
  Eigen::VectorXd w(2);
  w << 0.5, 0.5;
  Portfolio p({"SPY", "GLD"}, w, 1'000'000.0);
  StressScenario sc;
  sc.name = "only SPY";
  sc.asset_shocks = {{"SPY", -0.10}};  // GLD unmentioned
  StressResult r = apply_scenario(p, sc);
  CHECK_THAT(r.asset_shock(1), WithinAbs(0.0, 1e-15));
  CHECK_THAT(r.pct_pnl, WithinAbs(0.5 * -0.10, 1e-12));
}

TEST_CASE("scenarios load from JSON config", "[stress][json]") {
  auto scenarios = load_scenarios(cfg("stress_scenarios.json"));
  REQUIRE(scenarios.size() == 6);

  // Find the 1987 single-day scenario and check its SPY shock.
  bool found = false;
  for (const auto& sc : scenarios) {
    if (sc.name == "1987 Black Monday") {
      found = true;
      CHECK(sc.type == "historical");
      CHECK_THAT(sc.asset_shocks.at("SPY"), WithinAbs(-0.205, 1e-12));
    }
  }
  CHECK(found);
}

TEST_CASE("factor betas load and drive the synthetic scenarios",
          "[stress][json]") {
  auto scenarios = load_scenarios(cfg("stress_scenarios.json"));
  auto betas = load_factor_betas(cfg("factor_betas.json"));

  Eigen::VectorXd w(3);
  w << 0.6, 0.3, 0.1;
  Portfolio p({"SPY", "TLT", "GLD"}, w, 1'000'000.0);

  // Find "USD +10%" and verify GLD's effective shock = beta(-0.60) * 0.10.
  for (const auto& sc : scenarios) {
    if (sc.name == "USD +10%") {
      StressResult r = apply_scenario(p, sc, betas);
      // GLD is index 2.
      CHECK_THAT(r.asset_shock(2), WithinAbs(-0.60 * 0.10, 1e-12));
      CHECK_THAT(r.asset_shock(0), WithinAbs(-0.30 * 0.10, 1e-12));
    }
  }
}

TEST_CASE("missing scenario file throws", "[stress][errors]") {
  CHECK_THROWS_AS(load_scenarios("/nonexistent/scenarios.json"),
                  std::invalid_argument);
}
