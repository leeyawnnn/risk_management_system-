#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdio>
#include <fstream>
#include <string>

#include "risk/config.hpp"
#include "risk/covariance.hpp"

using Catch::Matchers::WithinAbs;
using namespace risk;

namespace {

// Writes a config to a temporary path and removes it on scope exit, so a
// failing assertion cannot leave a stray file behind for the next run.
class TempConfig {
 public:
  explicit TempConfig(const std::string& body) {
    path_ = std::string(RISK_TEST_DATA_DIR) + "/tmp_config_" +
            std::to_string(counter_++) + ".json";
    std::ofstream out(path_);
    out << body;
  }
  TempConfig(const TempConfig&) = delete;
  TempConfig& operator=(const TempConfig&) = delete;
  TempConfig(TempConfig&&) = delete;
  TempConfig& operator=(TempConfig&&) = delete;
  ~TempConfig() {
    // Best effort: a failure to unlink a scratch file should not abort a
    // test run, but the result is checked so it is not silently discarded.
    if (std::remove(path_.c_str()) != 0) {
      // Nothing useful to do here; the next run overwrites it anyway.
    }
  }

  [[nodiscard]] const std::string& path() const { return path_; }

 private:
  std::string path_;
  static int counter_;
};

int TempConfig::counter_ = 0;

const char* kMinimal = R"({
  "name": "Tiny",
  "notional": 1000000,
  "positions": [
    { "asset": "A", "weight": 0.6, "sector": "Equity" },
    { "asset": "B", "weight": 0.4 }
  ]
})";

}  // namespace

TEST_CASE("config defaults are applied when keys are absent", "[config]") {
  const TempConfig cfg(kMinimal);
  const EngineConfig c = load_engine_config(cfg.path());

  CHECK(c.name == "Tiny");
  CHECK(c.notional == 1000000);
  // Everything the file did not say must fall back to a documented default
  // rather than to whatever was left in the struct.
  CHECK(c.return_type == ReturnType::Log);
  CHECK(c.missing_policy == MissingPolicy::Skip);
  CHECK(c.annualization_factor == 252.0);
  CHECK(c.covariance_method == "sample");
  CHECK(c.ewma_lambda == 0.94);
  CHECK(c.confidences == std::vector<double>{0.95, 0.99});
  CHECK(c.mc_draws == 100000);
  CHECK(c.seed == 42);

  REQUIRE(c.assets.size() == 2);
  CHECK(c.assets[0] == "A");
  CHECK(c.weights[1] == 0.4);
  // A position without a sector is simply absent from the map, not mapped to
  // an empty string that would then become its own bucket in the roll-up.
  CHECK(c.sectors.size() == 1);
  CHECK(c.sectors.at("A") == "Equity");
}

TEST_CASE("config overrides are read", "[config]") {
  const TempConfig cfg(R"({
    "name": "Full",
    "notional": 5000000,
    "return_type": "simple",
    "missing_policy": "fill_forward",
    "annualization_factor": 260,
    "covariance": "ledoit_wolf",
    "ewma_lambda": 0.97,
    "confidences": [0.9, 0.975],
    "horizons": [1, 5, 20],
    "monte_carlo": { "draws": 1234, "seed": 99 },
    "stress_scenarios": "s.json",
    "factor_betas": "b.json",
    "positions": [ { "asset": "X", "weight": 1.0 } ]
  })");
  const EngineConfig c = load_engine_config(cfg.path());

  CHECK(c.return_type == ReturnType::Simple);
  CHECK(c.missing_policy == MissingPolicy::FillForward);
  CHECK(c.annualization_factor == 260.0);
  CHECK(c.covariance_method == "ledoit_wolf");
  CHECK(c.ewma_lambda == 0.97);
  CHECK(c.confidences == std::vector<double>{0.9, 0.975});
  CHECK(c.horizons == std::vector<int>{1, 5, 20});
  CHECK(c.mc_draws == 1234);
  CHECK(c.seed == 99);
  CHECK(c.stress_scenarios_path == "s.json");
  CHECK(c.factor_betas_path == "b.json");
}

TEST_CASE("make_portfolio carries names, weights and notional", "[config]") {
  const TempConfig cfg(kMinimal);
  const Portfolio p = make_portfolio(load_engine_config(cfg.path()));
  CHECK(p.size() == 2);
  CHECK(p.names()[1] == "B");
  CHECK_THAT(p.weights()(0), WithinAbs(0.6, 1e-15));
  CHECK(p.notional() == 1000000);
}

TEST_CASE("config rejects malformed input rather than guessing", "[config]") {
  CHECK_THROWS(load_engine_config("/nonexistent/portfolio.json"));

  const TempConfig no_positions(R"({"name": "X", "positions": []})");
  CHECK_THROWS(load_engine_config(no_positions.path()));

  const TempConfig missing_key(R"({"name": "X"})");
  CHECK_THROWS(load_engine_config(missing_key.path()));

  const TempConfig bad_return_type(
      R"({"return_type": "continuous",
          "positions": [{"asset": "A", "weight": 1.0}]})");
  CHECK_THROWS(load_engine_config(bad_return_type.path()));

  const TempConfig bad_policy(
      R"({"missing_policy": "interpolate",
          "positions": [{"asset": "A", "weight": 1.0}]})");
  CHECK_THROWS(load_engine_config(bad_policy.path()));

  const TempConfig no_weight(R"({"positions": [{"asset": "A"}]})");
  CHECK_THROWS(load_engine_config(no_weight.path()));
}

TEST_CASE("estimate_covariance dispatches and rejects unknown names",
          "[config]") {
  Eigen::MatrixXd X(8, 2);
  X << 0.012, -0.004, -0.008, 0.011, 0.015, -0.009, 0.003, 0.002, -0.011, 0.006,
      0.007, -0.003, -0.005, 0.008, 0.010, -0.006;

  CHECK(estimate_covariance("sample", X, 0.94) == sample_covariance(X));
  CHECK(estimate_covariance("ewma", X, 0.94) == ewma_covariance(X, 0.94));
  CHECK(estimate_covariance("ledoit_wolf", X, 0.94) ==
        ledoit_wolf_covariance(X).cov);

  // An unknown name must throw, never quietly fall back to the sample
  // estimator: a typo in the config would otherwise change the reported VaR
  // with nothing in the output saying so.
  CHECK_THROWS(estimate_covariance("shrinkage", X, 0.94));
  CHECK_THROWS(estimate_covariance("", X, 0.94));
}

TEST_CASE("the shipped portfolio config loads and is self-consistent",
          "[config][shipped]") {
  const std::string path =
      std::string(RISK_TEST_DATA_DIR) + "/../../config/portfolio.json";
  const EngineConfig c = load_engine_config(path);

  REQUIRE(!c.assets.empty());
  CHECK(c.assets.size() == c.weights.size());

  double total = 0.0;
  for (double w : c.weights) total += w;
  CHECK_THAT(total, WithinAbs(1.0, 1e-12));

  // Every position carries an asset class, or the sector roll-up silently
  // grows an "Unclassified" bucket.
  for (const auto& asset : c.assets) {
    INFO("asset " << asset);
    CHECK(c.sectors.count(asset) == 1);
  }

  for (double conf : c.confidences) {
    CHECK(conf > 0.0);
    CHECK(conf < 1.0);
  }
  for (int h : c.horizons) CHECK(h >= 1);
  CHECK(c.notional > 0.0);
}
