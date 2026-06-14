#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>

#include "risk/portfolio.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using risk::Portfolio;

namespace {
// Equal-weight 3-asset portfolio used throughout.
Portfolio equal_weight() {
  Eigen::VectorXd w(3);
  w << 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0;
  return Portfolio({"A", "B", "C"}, w, 1'000'000.0);
}

// Diagonal covariance: vols 0.2, 0.3, 0.4 (variances 0.04, 0.09, 0.16).
Eigen::MatrixXd diagonal_cov() {
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(3, 3);
  S.diagonal() << 0.04, 0.09, 0.16;
  return S;
}

// Correlated covariance: as above but cov(A,B) = 0.006 (corr 0.1).
Eigen::MatrixXd correlated_cov() {
  Eigen::MatrixXd S = diagonal_cov();
  S(0, 1) = S(1, 0) = 0.006;
  return S;
}
}  // namespace

TEST_CASE("portfolio variance equals w^T Sigma w (hand-computed)",
          "[portfolio]") {
  auto p = equal_weight();

  // Diagonal: w^T Sigma w = (0.04 + 0.09 + 0.16) / 9 = 0.29/9.
  CHECK_THAT(p.variance(diagonal_cov()), WithinRel(0.29 / 9.0, 1e-12));
  CHECK_THAT(p.volatility(diagonal_cov()),
             WithinRel(std::sqrt(0.29 / 9.0), 1e-12));

  // Correlated: sum of all Sigma entries / 9 = (0.29 + 2*0.006)/9 = 0.302/9.
  CHECK_THAT(p.variance(correlated_cov()), WithinRel(0.302 / 9.0, 1e-12));
}

TEST_CASE("marginal contributions equal (Sigma w)/sigma_p", "[portfolio]") {
  auto p = equal_weight();
  auto S = correlated_cov();
  const double vol = p.volatility(S);

  Eigen::VectorXd sigma_w = S * p.weights();
  Eigen::VectorXd mcr = p.marginal_contributions(S);
  for (Eigen::Index i = 0; i < 3; ++i)
    CHECK_THAT(mcr(i), WithinRel(sigma_w(i) / vol, 1e-12));
}

TEST_CASE("component contributions sum to portfolio volatility",
          "[portfolio]") {
  auto p = equal_weight();
  for (const auto& S : {diagonal_cov(), correlated_cov()}) {
    Eigen::VectorXd ccr = p.component_contributions(S);
    CHECK_THAT(ccr.sum(), WithinAbs(p.volatility(S), 1e-12));
  }
}

TEST_CASE("percent contributions sum to 1", "[portfolio]") {
  auto p = equal_weight();
  Eigen::VectorXd pct = p.percent_contributions(correlated_cov());
  CHECK_THAT(pct.sum(), WithinAbs(1.0, 1e-12));
}

TEST_CASE("diagonal covariance: component risk ranks with variance",
          "[portfolio]") {
  // For equal weights and a diagonal cov, CCR_i = w_i^2 * var_i / sigma_p, so
  // the highest-variance asset (C) contributes the most risk.
  auto p = equal_weight();
  auto S = diagonal_cov();
  Eigen::VectorXd ccr = p.component_contributions(S);

  // Hand value: CCR_i = (1/9) var_i / sigma_p.
  const double sigma_p = p.volatility(S);
  CHECK_THAT(ccr(0), WithinRel((1.0 / 9.0) * 0.04 / sigma_p, 1e-12));
  CHECK_THAT(ccr(2), WithinRel((1.0 / 9.0) * 0.16 / sigma_p, 1e-12));
  CHECK(ccr(2) > ccr(1));
  CHECK(ccr(1) > ccr(0));
}

TEST_CASE("portfolio return series is the weighted sum of asset returns",
          "[portfolio]") {
  Eigen::VectorXd w(2);
  w << 0.6, 0.4;
  Portfolio p({"X", "Y"}, w);

  Eigen::MatrixXd X(3, 2);
  X << 0.01, -0.02,
       0.00, 0.03,
       -0.01, 0.01;
  Eigen::VectorXd r = p.return_series(X);

  REQUIRE(r.size() == 3);
  CHECK_THAT(r(0), WithinRel(0.6 * 0.01 + 0.4 * -0.02, 1e-12));
  CHECK_THAT(r(1), WithinRel(0.6 * 0.00 + 0.4 * 0.03, 1e-12));
  CHECK_THAT(r(2), WithinRel(0.6 * -0.01 + 0.4 * 0.01, 1e-12));
}

TEST_CASE("dimension mismatches are rejected", "[portfolio][errors]") {
  Eigen::VectorXd w(3);
  w << 0.3, 0.3, 0.4;
  Portfolio p({"A", "B", "C"}, w);

  Eigen::MatrixXd bad = Eigen::MatrixXd::Identity(2, 2);
  CHECK_THROWS_AS(p.variance(bad), std::invalid_argument);

  Eigen::MatrixXd bad_returns(4, 2);  // 2 cols != 3 assets
  bad_returns.setZero();
  CHECK_THROWS_AS(p.return_series(bad_returns), std::invalid_argument);

  // names/weights mismatch at construction.
  CHECK_THROWS_AS(Portfolio({"A"}, w), std::invalid_argument);
}
