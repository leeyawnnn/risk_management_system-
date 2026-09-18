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

TEST_CASE("sector roll-up sums position contributions",
          "[attribution][sector]") {
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
