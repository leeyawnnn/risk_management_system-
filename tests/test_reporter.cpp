#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <map>
#include <random>
#include <string>

#include "risk/covariance.hpp"
#include "risk/decomposition.hpp"
#include "risk/portfolio.hpp"
#include "risk/reporter.hpp"
#include "risk/stress_test.hpp"
#include "test_support.hpp"

using Catch::Matchers::WithinAbs;
using namespace risk;

namespace {

// A return sample long enough for the backtests to have something to chew on.
Eigen::MatrixXd sample_returns(int T = 600) {
  std::mt19937_64 gen(4242);
    Eigen::MatrixXd L(3, 3);
  L << 0.010, 0.0, 0.0, 0.004, 0.006, 0.0, -0.002, 0.001, 0.015;
  Eigen::MatrixXd X(T, 3);
  for (int t = 0; t < T; ++t) {
    Eigen::VectorXd z(3);
    for (int i = 0; i < 3; ++i) z(i) = risk_test::normal(gen);
    X.row(t) = (L * z).transpose();
  }
  return X;
}

Portfolio book() {
  Eigen::VectorXd w(3);
  w << 0.5, 0.3, 0.2;
  return Portfolio({"A", "B", "C"}, w, 10'000'000.0);
}

ReportInputs base_inputs(const Eigen::MatrixXd& X, const Eigen::MatrixXd& cov) {
  ReportInputs in;
  in.asset_returns = &X;
  in.cov = &cov;
  in.cov_method = "sample";
  in.annualization_factor = 252.0;
  in.confidences = {0.95, 0.99};
  in.horizons = {1, 10};
  in.sectors = {{"A", "Equity"}, {"B", "Rates"}, {"C", "Commodity"}};
  in.mc_draws = 20000;
  in.seed = 7;
  in.bootstrap_resamples = 300;
  in.as_of = "2026-09-11";
  in.git_commit = "abcdef1234";
  return in;
}

}  // namespace

TEST_CASE("report assembles the VaR table and serialises", "[reporter]") {
  const Eigen::MatrixXd X = sample_returns();
  const Eigen::MatrixXd cov = sample_covariance(X);
  const Portfolio p = book();

  StressScenario sc{.name = "Equity -20%",
                    .type = "hypothetical",
                    .description = "test",
                    .asset_shocks = {{"A", -0.20}},
                    .factor_shocks = {}};
  ReportInputs in = base_inputs(X, cov);
  in.stress = apply_scenarios(p, {sc});

  const RiskReport rep = build_report("Test Book", p, in);

  // 3 methods x 2 confidences x 2 horizons.
  CHECK(rep.var_cvar.size() == 12);
  // CVaR is the mean beyond the quantile, so it can never be shallower.
  for (const auto& r : rep.var_cvar) CHECK(r.cvar >= r.var);

  const std::string js = to_json(rep);
  CHECK(js.find("\"portfolio_name\": \"Test Book\"") != std::string::npos);
  CHECK(js.find("\"var_cvar\"") != std::string::npos);
  CHECK(js.find("\"backtests\"") != std::string::npos);
  CHECK(js.find("\"var_decomposition\"") != std::string::npos);

  const std::string md = to_markdown(rep);
  CHECK(md.find("# Risk report: Test Book") != std::string::npos);
  CHECK(md.find("Value at Risk") != std::string::npos);
  CHECK(md.find("## Backtests") != std::string::npos);
}

TEST_CASE("every report artifact is stamped with its provenance",
          "[reporter][provenance]") {
  // A number with no as-of date, commit and seed attached cannot be traced
  // back to a run, which is the whole point of committing the report.
  const Eigen::MatrixXd X = sample_returns();
  const Eigen::MatrixXd cov = sample_covariance(X);
  const RiskReport rep = build_report("Stamped", book(), base_inputs(X, cov));

  CHECK(rep.as_of == "2026-09-11");
  CHECK(rep.git_commit == "abcdef1234");
  CHECK(rep.seed == 7);

  const std::string js = to_json(rep);
  CHECK(js.find("\"as_of\": \"2026-09-11\"") != std::string::npos);
  CHECK(js.find("\"git_commit\": \"abcdef1234\"") != std::string::npos);

  const std::string md = to_markdown(rep);
  CHECK(md.find("2026-09-11") != std::string::npos);
  CHECK(md.find("abcdef1234") != std::string::npos);
}

TEST_CASE("report carries backtests for every method and confidence",
          "[reporter][backtest]") {
  const Eigen::MatrixXd X = sample_returns();
  const Eigen::MatrixXd cov = sample_covariance(X);
  const RiskReport rep =
      build_report("Backtested", book(), base_inputs(X, cov));

  // 3 methods x 2 confidences.
  REQUIRE(rep.backtests.size() == 6);
  for (const auto& b : rep.backtests) {
    CHECK(b.kupiec.observations == X.rows());
    CHECK(b.kupiec.p_value >= 0.0);
    CHECK(b.kupiec.p_value <= 1.0);
    CHECK(b.christoffersen.p_value_independence >= 0.0);
    CHECK(b.basel.window == 250);
    CHECK(b.es_test.simulations > 0);
  }

  // The data is Gaussian and the parametric model is correct for it, so
  // nothing should reject at 95%. If this starts failing, either the test
  // sample or a test statistic has changed.
  for (const auto& b : rep.backtests) {
    if (b.method != "Parametric") continue;
    INFO("confidence " << b.confidence);
    CHECK_FALSE(b.kupiec.reject_at_95);
  }
}

TEST_CASE("report VaR decomposition satisfies Euler exactly",
          "[reporter][decomposition]") {
  const Eigen::MatrixXd X = sample_returns();
  const Eigen::MatrixXd cov = sample_covariance(X);
  const RiskReport rep = build_report("Euler", book(), base_inputs(X, cov));

  REQUIRE(rep.var_decomposition.size() == 2);
  for (const auto& d : rep.var_decomposition) {
    CHECK(std::abs(d.euler_residual) < 1e-15);
    CHECK_THAT(d.component.sum(), WithinAbs(d.var, 1e-15));
  }
}

TEST_CASE("asset statistics carry bootstrap intervals around the volatility",
          "[reporter][uncertainty]") {
  const Eigen::MatrixXd X = sample_returns();
  const Eigen::MatrixXd cov = sample_covariance(X);
  const RiskReport rep = build_report("CI", book(), base_inputs(X, cov));

  REQUIRE(rep.asset_stats.size() == 3);
  double risk_sum = 0.0;
  for (const auto& s : rep.asset_stats) {
    CHECK(s.annual_vol > 0.0);
    // The point estimate must sit inside its own interval, and the interval
    // must have positive width -- a zero-width interval means the bootstrap
    // silently did not run.
    CHECK(s.annual_vol_ci_lower < s.annual_vol);
    CHECK(s.annual_vol > 0.0);
    CHECK(s.annual_vol_ci_upper > s.annual_vol);
    risk_sum += s.pct_risk;
  }
  CHECK_THAT(risk_sum, WithinAbs(1.0, 1e-10));
}

TEST_CASE("estimator comparison reports conditioning and shrinkage",
          "[reporter][estimators]") {
  const Eigen::MatrixXd X = sample_returns();
  const Eigen::MatrixXd cov = sample_covariance(X);
  const RiskReport rep =
      build_report("Estimators", book(), base_inputs(X, cov));

  REQUIRE(rep.estimator_comparison.size() == 3);
  for (const auto& e : rep.estimator_comparison) {
    CHECK(e.var99 > e.var95);
    CHECK(e.condition_number > 0.0);
    CHECK(e.correlation_condition_number > 0.0);
  }
  // Only Ledoit-Wolf reports a shrinkage intensity; the others must not
  // pretend to have one.
  for (const auto& e : rep.estimator_comparison) {
    if (e.method == "Ledoit-Wolf") {
      CHECK(e.shrinkage > 0.0);
      CHECK(e.shrinkage <= 1.0);
    } else {
      CHECK(e.shrinkage == 0.0);
    }
  }
}

TEST_CASE("factor sections appear only when factor inputs are supplied",
          "[reporter][factors]") {
  const Eigen::MatrixXd X = sample_returns();
  const Eigen::MatrixXd cov = sample_covariance(X);

  // Without factors the section is absent rather than built from defaults.
  const RiskReport bare = build_report("Bare", book(), base_inputs(X, cov));
  CHECK_FALSE(bare.has_factor_decomposition);
  CHECK(bare.reverse_stress.empty());
  CHECK(to_json(bare).find("factor_decomposition") == std::string::npos);

  FactorSeries fs;
  fs.names = {"f1", "f2"};
  fs.values = X.leftCols(2);
  FactorBetas betas = {{"A", {{"f1", 1.0}}},
                       {"B", {{"f2", 0.8}}},
                       {"C", {{"f1", 0.2}, {"f2", -0.3}}}};

  ReportInputs in = base_inputs(X, cov);
  in.factors = &fs;
  in.betas = &betas;
  in.reverse_stress_targets = {0.05, 0.10};

  const RiskReport full = build_report("Full", book(), in);
  CHECK(full.has_factor_decomposition);
  REQUIRE(full.reverse_stress.size() == 2);
  // A bigger loss must sit further out in the factor distribution.
  CHECK(full.reverse_stress[1].mahalanobis_distance >
        full.reverse_stress[0].mahalanobis_distance);
  CHECK(to_json(full).find("factor_decomposition") != std::string::npos);
  CHECK(to_markdown(full).find("Reverse stress") != std::string::npos);
}

TEST_CASE("incremental VaR rows appear for requested trades",
          "[reporter][incremental]") {
  const Eigen::MatrixXd X = sample_returns();
  const Eigen::MatrixXd cov = sample_covariance(X);
  ReportInputs in = base_inputs(X, cov);
  in.trades = {{"A", 1'000'000.0}, {"C", -500'000.0}};

  const RiskReport rep = build_report("Trades", book(), in);
  REQUIRE(rep.incremental.size() == 2);
  CHECK(rep.incremental[0].asset == "A");
  CHECK(rep.incremental[0].incremental > 0.0);
  CHECK(to_markdown(rep).find("Incremental VaR") != std::string::npos);
}

TEST_CASE("build_report rejects missing required inputs",
          "[reporter][errors]") {
  ReportInputs in;
  CHECK_THROWS(build_report("Nope", book(), in));
}
