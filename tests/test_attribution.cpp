#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>

#include "risk/attribution.hpp"
#include "risk/covariance.hpp"
#include "risk/figures.hpp"
#include "risk/reporter.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace risk;

namespace {
Portfolio equal_weight3() {
  Eigen::VectorXd w(3);
  w << 1.0 / 3, 1.0 / 3, 1.0 / 3;
  return Portfolio({"A", "B", "C"}, w, 1'000'000.0);
}
Eigen::MatrixXd diag_cov() {
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(3, 3);
  S.diagonal() << 0.04, 0.04, 0.04;  // identical, independent
  return S;
}
Eigen::MatrixXd correlated_cov() {
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(3, 3);
  S.diagonal() << 0.04, 0.09, 0.16;
  S(0, 1) = S(1, 0) = 0.006;
  return S;
}
}  // namespace

TEST_CASE("component contributions sum to portfolio volatility (1e-10)",
          "[attribution]") {
  auto p = equal_weight3();
  auto a = compute_attribution(p, correlated_cov());
  CHECK_THAT(a.component.sum(), WithinAbs(a.portfolio_vol, 1e-10));
  CHECK_THAT(a.percent.sum(), WithinAbs(1.0, 1e-10));
}

TEST_CASE("equal independent positions give ENB = n", "[attribution][conc]") {
  // 3 equal-weight, equal-variance, independent assets -> each pct = 1/3,
  // HHI = 3*(1/9) = 1/3, ENB = 3.
  auto p = equal_weight3();
  auto a = compute_attribution(p, diag_cov());
  for (Eigen::Index i = 0; i < 3; ++i)
    CHECK_THAT(a.percent(i), WithinAbs(1.0 / 3.0, 1e-12));
  CHECK_THAT(a.concentration.herfindahl, WithinAbs(1.0 / 3.0, 1e-12));
  CHECK_THAT(a.concentration.effective_num_bets, WithinAbs(3.0, 1e-10));
}

TEST_CASE("concentration identifies the largest contributor",
          "[attribution][conc]") {
  // With the correlated cov, asset C (highest variance) dominates risk.
  auto p = equal_weight3();
  auto a = compute_attribution(p, correlated_cov());
  CHECK(a.concentration.max_contributor == "C");
  CHECK(a.concentration.max_contribution > 1.0 / 3.0);
  CHECK(a.concentration.effective_num_bets < 3.0);
}

TEST_CASE("sector roll-up sums position contributions", "[attribution][sector]") {
  auto p = equal_weight3();
  std::map<std::string, std::string> sectors = {
      {"A", "Equity"}, {"B", "Equity"}, {"C", "Bonds"}};
  auto a = compute_attribution(p, correlated_cov(), sectors);

  // Equity sector % = pct(A) + pct(B); total across sectors = 1.
  CHECK_THAT(a.sector_percent.at("Equity"),
             WithinRel(a.percent(0) + a.percent(1), 1e-12));
  double total = 0.0;
  for (auto& [s, v] : a.sector_percent) total += v;
  CHECK_THAT(total, WithinAbs(1.0, 1e-10));
}

TEST_CASE("report builds and serializes to JSON and Markdown", "[reporter]") {
  // Two assets, simple synthetic returns.
  Eigen::MatrixXd X(6, 2);
  X << 0.01, -0.005,
       -0.002, 0.004,
       0.003, 0.001,
       -0.006, 0.002,
       0.004, -0.003,
       0.002, 0.0;
  Eigen::VectorXd w(2);
  w << 0.5, 0.5;
  Portfolio p({"X", "Y"}, w, 1'000'000.0);

  Eigen::MatrixXd cov(2, 2);
  cov << 0.0001, 0.00002, 0.00002, 0.00015;

  StressScenario sc{"Equity -20%", "synthetic", "", {{"X", -0.20}}, {}};
  auto stress = apply_scenarios(p, {sc});

  RiskReport rep = build_report("Test Book", p, X, cov, "sample", 252.0,
                                {0.95, 0.99}, {1, 10}, stress, {}, 20000, 7);

  // Table has 3 methods x 2 confidences x 2 horizons = 12 rows.
  CHECK(rep.var_cvar.size() == 12);

  std::string js = to_json(rep);
  CHECK(js.find("\"portfolio_name\": \"Test Book\"") != std::string::npos);
  CHECK(js.find("\"var_cvar\"") != std::string::npos);
  CHECK(js.find("\"attribution\"") != std::string::npos);

  std::string md = to_markdown(rep);
  CHECK(md.find("# Risk Report: Test Book") != std::string::npos);
  CHECK(md.find("Value at Risk") != std::string::npos);

  // Every VaR row should have CVaR >= VaR.
  for (const auto& r : rep.var_cvar) CHECK(r.cvar >= r.var);
}

TEST_CASE("report includes asset stats, estimator comparison, and backtest",
          "[reporter][detail]") {
  Eigen::MatrixXd X(8, 3);
  X << 0.012, -0.004, 0.006,
      -0.008, 0.011, -0.002,
       0.015, -0.009, 0.004,
       0.003, 0.002, -0.007,
      -0.011, 0.006, 0.010,
       0.007, -0.003, -0.001,
      -0.005, 0.008, 0.003,
       0.010, -0.006, -0.004;
  Eigen::VectorXd w(3);
  w << 0.5, 0.3, 0.2;
  Portfolio p({"A", "B", "C"}, w, 1'000'000.0);
  Eigen::MatrixXd cov = sample_covariance(X);
  std::map<std::string, std::string> sectors = {
      {"A", "Equity"}, {"B", "Rates"}, {"C", "Commodity"}};

  RiskReport rep = build_report("Detail", p, X, cov, "sample", 252.0, {0.95, 0.99},
                                {1}, {}, sectors, 20000, 7);

  // Per-asset stats: one per asset, pct_risk sums to 1.
  REQUIRE(rep.asset_stats.size() == 3);
  double risk_sum = 0.0;
  for (const auto& a : rep.asset_stats) {
    CHECK(a.annual_vol > 0.0);
    risk_sum += a.pct_risk;
  }
  CHECK_THAT(risk_sum, WithinAbs(1.0, 1e-10));

  // Three estimators compared.
  REQUIRE(rep.estimator_comparison.size() == 3);
  for (const auto& e : rep.estimator_comparison) CHECK(e.var99 > e.var95);

  // One backtest per confidence; exceedance rate is a valid fraction.
  REQUIRE(rep.backtests.size() == 2);
  for (const auto& b : rep.backtests) {
    CHECK(b.observations == 8);
    CHECK(b.actual_rate >= 0.0);
    CHECK(b.actual_rate <= 1.0);
  }

  // JSON + MD carry the new sections.
  std::string js = to_json(rep);
  CHECK(js.find("\"asset_stats\"") != std::string::npos);
  CHECK(js.find("\"estimator_comparison\"") != std::string::npos);
  CHECK(js.find("\"var_backtest\"") != std::string::npos);
  std::string md = to_markdown(rep);
  CHECK(md.find("Asset comparison") != std::string::npos);
  CHECK(md.find("Covariance estimator comparison") != std::string::npos);
  CHECK(md.find("VaR backtest") != std::string::npos);
}

TEST_CASE("SVG figures are well-formed documents", "[figures]") {
  auto p = equal_weight3();
  auto cov = correlated_cov();
  auto a = compute_attribution(p, cov);
  Eigen::MatrixXd corr = cov_to_correlation(cov);

  std::string heat = svg_correlation_heatmap(corr, {"A", "B", "C"});
  CHECK(heat.rfind("<svg", 0) == 0);
  CHECK(heat.find("</svg>") != std::string::npos);

  std::string bars = svg_risk_contribution_bars(a);
  CHECK(bars.find("<svg") != std::string::npos);

  Eigen::VectorXd r(1000);
  for (Eigen::Index i = 0; i < r.size(); ++i)
    r(i) = 0.001 * std::sin(static_cast<double>(i)) - 0.0005;
  std::string hist = svg_return_histogram(r, 0.02, 0.025, 0.95);
  CHECK(hist.find("VaR") != std::string::npos);
  CHECK(hist.find("CVaR") != std::string::npos);
}

TEST_CASE("new comparison figures are well-formed", "[figures][detail]") {
  std::vector<std::string> names = {"SPY", "TLT", "GLD"};
  std::vector<std::string> sectors = {"Equity", "Rates", "Commodity"};
  std::vector<double> vols = {0.16, 0.14, 0.15};
  std::vector<double> weights = {0.6, 0.3, 0.1};
  std::vector<double> risk = {0.89, 0.06, 0.05};

  std::string vol = svg_asset_volatility(names, vols, sectors);
  CHECK(vol.rfind("<svg", 0) == 0);
  CHECK(vol.find("Equity") != std::string::npos);

  std::string wvr = svg_weight_vs_risk(names, weights, risk);
  CHECK(wvr.find("risk") != std::string::npos);

  std::vector<std::string> methods = {"Sample", "EWMA", "Ledoit-Wolf"};
  std::vector<double> v95 = {0.009, 0.011, 0.009};
  std::vector<double> v99 = {0.013, 0.016, 0.013};
  std::string ec = svg_estimator_var_comparison(methods, v95, v99);
  CHECK(ec.find("EWMA") != std::string::npos);

  Eigen::VectorXd rr(500);
  for (Eigen::Index i = 0; i < rr.size(); ++i)
    rr(i) = 0.002 * std::sin(static_cast<double>(i) * 0.3) - 0.0003;
  std::string bt = svg_var_backtest(rr, 0.02, 0.95);
  CHECK(bt.find("breaches") != std::string::npos);
}
