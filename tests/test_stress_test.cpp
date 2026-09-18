#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>

#include "risk/stress_test.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace risk;

namespace {
std::string fixture(const std::string& name) {
  return std::string(RISK_TEST_DATA_DIR) + "/" + name;
}
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

TEST_CASE("scenarios load from JSON", "[stress][json]") {
  // Against a fixture, not the shipped config. Pinning exact shock sizes to
  // config/stress_scenarios.json would make every re-measurement of the
  // historical episodes look like a test failure.
  auto scenarios = load_scenarios(fixture("fixture_scenarios.json"));
  REQUIRE(scenarios.size() == 3);

  bool found = false;
  for (const auto& sc : scenarios) {
    if (sc.name == "Direct asset shock") {
      found = true;
      CHECK(sc.type == "historical");
      CHECK_THAT(sc.asset_shocks.at("AAA"), WithinAbs(-0.205, 1e-12));
      CHECK(sc.factor_shocks.empty());
    }
  }
  CHECK(found);
}

TEST_CASE("factor betas load from the nested generated shape",
          "[stress][json]") {
  // The generated file puts loadings under "betas" beside "_meta" and
  // "diagnostics". Reading the top level would invent two phantom assets.
  auto betas = load_factor_betas(fixture("fixture_betas.json"));
  CHECK(betas.size() == 3);
  CHECK(betas.count("_meta") == 0);
  CHECK(betas.count("diagnostics") == 0);
  CHECK_THAT(betas.at("CCC").at("usd"), WithinAbs(-0.60, 1e-12));
}

TEST_CASE("factor shocks drive asset shocks through the betas",
          "[stress][json]") {
  auto scenarios = load_scenarios(fixture("fixture_scenarios.json"));
  auto betas = load_factor_betas(fixture("fixture_betas.json"));

  Eigen::VectorXd w(3);
  w << 0.6, 0.3, 0.1;
  Portfolio p({"AAA", "BBB", "CCC"}, w, 1'000'000.0);

  bool checked = false;
  for (const auto& sc : scenarios) {
    if (sc.name == "Pure factor shock") {
      StressResult r = apply_scenario(p, sc, betas);
      CHECK_THAT(r.asset_shock(0), WithinAbs(-0.30 * 0.10, 1e-12));
      CHECK_THAT(r.asset_shock(2), WithinAbs(-0.60 * 0.10, 1e-12));
      checked = true;
    }
    if (sc.name == "Combined") {
      // Asset and factor contributions add for the same asset.
      StressResult r = apply_scenario(p, sc, betas);
      CHECK_THAT(r.asset_shock(0), WithinAbs(-0.01 + 1.00 * -0.20, 1e-12));
    }
  }
  CHECK(checked);
}

TEST_CASE("the shipped config loads and is internally consistent",
          "[stress][json][config]") {
  // Checks shape and provenance, not values: the historical shocks are
  // measured from FRED and are expected to move when the data is refreshed.
  auto scenarios = load_scenarios(cfg("stress_scenarios.json"));
  auto betas = load_factor_betas(cfg("factor_betas.json"));
  REQUIRE(!scenarios.empty());
  REQUIRE(!betas.empty());

  for (const auto& sc : scenarios) {
    CHECK(!sc.name.empty());
    CHECK(!sc.description.empty());
    CHECK((sc.type == "historical" || sc.type == "hypothetical"));
    CHECK((!sc.asset_shocks.empty() || !sc.factor_shocks.empty()));
    // Every factor a scenario shocks must be one some instrument loads on,
    // otherwise the scenario silently does nothing.
    for (const auto& [factor, shock] : sc.factor_shocks) {
      bool loaded = false;
      for (const auto& [asset, loadings] : betas) {
        if (loadings.count(factor) > 0) loaded = true;
      }
      INFO("scenario " << sc.name << " shocks unknown factor " << factor);
      CHECK(loaded);
    }
  }
}

TEST_CASE("missing scenario file throws", "[stress][errors]") {
  CHECK_THROWS_AS(load_scenarios("/nonexistent/scenarios.json"),
                  std::invalid_argument);
}
