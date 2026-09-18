#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "risk/decomposition.hpp"
#include "risk/portfolio.hpp"
#include "risk/stress_test.hpp"
#include "risk/var.hpp"

using Catch::Approx;
using namespace risk;

namespace {

Eigen::MatrixXd three_asset_cov() {
  // Daily covariance for three assets with vols 1.0%, 0.6%, 2.0% and
  // correlations 0.5, -0.2, 0.1.
  Eigen::VectorXd sd(3);
  sd << 0.010, 0.006, 0.020;
  Eigen::MatrixXd c(3, 3);
  c << 1.0, 0.5, -0.2, 0.5, 1.0, 0.1, -0.2, 0.1, 1.0;
  return sd.asDiagonal() * c * sd.asDiagonal();
}

Portfolio three_asset_book(double notional = 10'000'000.0) {
  Eigen::VectorXd w(3);
  w << 0.5, 0.3, 0.2;
  return Portfolio({"A", "B", "C"}, w, notional);
}

}  // namespace

// ---------------------------------------------------------------------------
// VaR decomposition
// ---------------------------------------------------------------------------

TEST_CASE("component VaR sums to total VaR (Euler)") {
  const Eigen::MatrixXd cov = three_asset_cov();
  const Portfolio p = three_asset_book();
  Eigen::VectorXd mean(3);
  mean << 0.0004, 0.0001, -0.0002;

  for (double conf : {0.95, 0.99}) {
    for (int h : {1, 10}) {
      const auto d = decompose_var(p, cov, mean, conf, h);
      CHECK(d.component.sum() == Approx(d.var).epsilon(1e-12));
      CHECK(std::abs(d.euler_residual) < 1e-15);
      CHECK(d.percent.sum() == Approx(1.0).epsilon(1e-12));
    }
  }
}

TEST_CASE("decomposed VaR equals the parametric VaR of the same book") {
  const Eigen::MatrixXd cov = three_asset_cov();
  const Portfolio p = three_asset_book();
  Eigen::VectorXd mean(3);
  mean << 0.0004, 0.0001, -0.0002;

  const auto d = decompose_var(p, cov, mean, 0.99, 1);
  const double direct =
      parametric_var(p.weights().dot(mean), p.volatility(cov), 0.99, 1);
  CHECK(d.var == Approx(direct).epsilon(1e-12));
}

TEST_CASE("marginal VaR matches a numerical derivative") {
  // The analytic gradient must agree with a central difference of the VaR
  // function itself. This is the check that catches a wrong chain rule.
  const Eigen::MatrixXd cov = three_asset_cov();
  const Portfolio p = three_asset_book();
  Eigen::VectorXd mean(3);
  mean << 0.0004, 0.0001, -0.0002;
  const auto d = decompose_var(p, cov, mean, 0.975, 1);

  const double eps = 1e-7;
  for (Eigen::Index i = 0; i < 3; ++i) {
    Eigen::VectorXd up = p.weights();
    Eigen::VectorXd dn = p.weights();
    up(i) += eps;
    dn(i) -= eps;
    auto var_of = [&](const Eigen::VectorXd& w) {
      return parametric_var(w.dot(mean), std::sqrt(w.dot(cov * w)), 0.975, 1);
    };
    const double numeric = (var_of(up) - var_of(dn)) / (2.0 * eps);
    CHECK(d.marginal(i) == Approx(numeric).epsilon(1e-6));
  }
}

TEST_CASE("a hedge takes a negative component VaR") {
  // Asset C is negatively correlated with the dominant position A. Held short
  // against it, its component VaR should come out negative: it is paying for
  // itself in risk terms.
  Eigen::VectorXd sd(2);
  sd << 0.02, 0.02;
  Eigen::MatrixXd c(2, 2);
  c << 1.0, -0.8, -0.8, 1.0;
  const Eigen::MatrixXd cov = sd.asDiagonal() * c * sd.asDiagonal();

  Eigen::VectorXd w(2);
  w << 0.9, 0.1;
  const Portfolio p({"RISK", "HEDGE"}, w, 1'000'000.0);
  const auto d = decompose_var(p, cov, Eigen::VectorXd::Zero(2), 0.99, 1);

  CHECK(d.component(1) < 0.0);
  CHECK(d.component.sum() == Approx(d.var).epsilon(1e-12));
}

// ---------------------------------------------------------------------------
// Incremental VaR
// ---------------------------------------------------------------------------

TEST_CASE("a zero trade has zero incremental VaR") {
  const auto r = incremental_var(three_asset_book(), three_asset_cov(),
                                 Eigen::VectorXd::Zero(3), "B", 0.0, 0.99);
  CHECK(r.incremental == Approx(0.0).margin(1e-9));
  CHECK(r.linear_estimate == Approx(0.0).margin(1e-12));
}

TEST_CASE("incremental VaR converges to the linear estimate for small trades") {
  const Eigen::MatrixXd cov = three_asset_cov();
  const Portfolio p = three_asset_book();
  const Eigen::VectorXd mean = Eigen::VectorXd::Zero(3);

  // The marginal-VaR linearisation is the first-order term, so the relative
  // gap must shrink as the trade does.
  double previous_gap = 1.0;
  for (double trade : {1'000'000.0, 100'000.0, 10'000.0, 1'000.0}) {
    const auto r = incremental_var(p, cov, mean, "C", trade, 0.99);
    const double gap = std::abs(r.incremental - r.linear_estimate) /
                       std::abs(r.linear_estimate);
    CHECK(gap < previous_gap);
    previous_gap = gap;
  }
  CHECK(previous_gap < 1e-3);
}

TEST_CASE("adding to the largest position raises VaR") {
  const auto r =
      incremental_var(three_asset_book(), three_asset_cov(),
                      Eigen::VectorXd::Zero(3), "A", 2'000'000.0, 0.99);
  CHECK(r.incremental > 0.0);
  CHECK(r.var_after > r.var_before);
}

TEST_CASE("incremental VaR rejects unknown assets and ruinous trades") {
  const Portfolio p = three_asset_book();
  const Eigen::MatrixXd cov = three_asset_cov();
  const Eigen::VectorXd mean = Eigen::VectorXd::Zero(3);
  CHECK_THROWS(incremental_var(p, cov, mean, "NOPE", 1.0, 0.99));
  CHECK_THROWS(incremental_var(p, cov, mean, "A", -20'000'000.0, 0.99));
}

// ---------------------------------------------------------------------------
// Factor decomposition
// ---------------------------------------------------------------------------

TEST_CASE("factor and specific variance sum to the model variance") {
  // Build a covariance that IS a factor model, so the decomposition must
  // reproduce it exactly and model_error must vanish.
  const std::vector<std::string> factors = {"f1", "f2"};
  Eigen::MatrixXd fcov(2, 2);
  fcov << 4e-4, 1e-4, 1e-4, 9e-4;

  std::map<std::string, std::map<std::string, double>> betas = {
      {"A", {{"f1", 1.0}, {"f2", 0.2}}},
      {"B", {{"f1", 0.5}, {"f2", -0.4}}},
      {"C", {{"f1", 0.0}, {"f2", 1.5}}},
  };
  Eigen::VectorXd specific(3);
  specific << 1e-5, 4e-6, 9e-6;

  Eigen::MatrixXd B(3, 2);
  B << 1.0, 0.2, 0.5, -0.4, 0.0, 1.5;
  const Eigen::MatrixXd cov =
      B * fcov * B.transpose() + Eigen::MatrixXd(specific.asDiagonal());

  const Portfolio p = three_asset_book();
  const auto d = decompose_factor_risk(p, betas, factors, fcov, specific, cov);

  CHECK(d.factor_variance + d.specific_variance ==
        Approx(d.model_variance).epsilon(1e-14));
  CHECK(d.variance_contribution.sum() ==
        Approx(d.factor_variance).epsilon(1e-14));
  // The covariance was built from exactly this model, so there is no gap.
  CHECK(d.model_error == Approx(0.0).margin(1e-18));
  CHECK(d.model_vol == Approx(p.volatility(cov)).epsilon(1e-12));
  CHECK(d.percent_of_total.sum() + d.specific_variance / d.model_variance ==
        Approx(1.0).epsilon(1e-12));
}

TEST_CASE("factor exposures are B transpose w") {
  const std::vector<std::string> factors = {"f1", "f2"};
  Eigen::MatrixXd fcov = Eigen::MatrixXd::Identity(2, 2) * 1e-4;
  std::map<std::string, std::map<std::string, double>> betas = {
      {"A", {{"f1", 1.0}}},
      {"B", {{"f2", 2.0}}},
      {"C", {{"f1", -1.0}, {"f2", 0.5}}},
  };
  const Portfolio p = three_asset_book();  // w = 0.5, 0.3, 0.2
  const auto d =
      decompose_factor_risk(p, betas, factors, fcov, Eigen::VectorXd::Zero(3),
                            Eigen::MatrixXd::Identity(3, 3) * 1e-4);
  CHECK(d.exposures(0) == Approx(0.5 * 1.0 + 0.2 * -1.0));
  CHECK(d.exposures(1) == Approx(0.3 * 2.0 + 0.2 * 0.5));
}

TEST_CASE("factor decomposition validates its shapes") {
  const Portfolio p = three_asset_book();
  const std::vector<std::string> factors = {"f1", "f2"};
  CHECK_THROWS(
      decompose_factor_risk(p, {}, factors, Eigen::MatrixXd::Identity(3, 3),
                            Eigen::VectorXd::Zero(3), three_asset_cov()));
  CHECK_THROWS(
      decompose_factor_risk(p, {}, factors, Eigen::MatrixXd::Identity(2, 2),
                            Eigen::VectorXd::Zero(2), three_asset_cov()));
}

// ---------------------------------------------------------------------------
// Reverse stress
// ---------------------------------------------------------------------------

TEST_CASE("reverse stress recovers exactly the target loss") {
  const std::vector<std::string> factors = {"f1", "f2"};
  Eigen::MatrixXd fcov(2, 2);
  fcov << 4e-4, 1e-4, 1e-4, 9e-4;
  FactorBetas betas = {
      {"A", {{"f1", 1.0}, {"f2", 0.2}}},
      {"B", {{"f1", 0.5}, {"f2", -0.4}}},
      {"C", {{"f1", 0.0}, {"f2", 1.5}}},
  };
  const Portfolio p = three_asset_book();

  for (double target : {0.05, 0.10, 0.25}) {
    const auto r = reverse_stress(p, betas, factors, fcov, target);
    // Applying the solved factor move must reproduce the requested loss.
    const double realised = p.weights().dot(r.asset_shock);
    CHECK(realised == Approx(-target).epsilon(1e-12));
    CHECK(r.mahalanobis_distance > 0.0);
    // Distance scales linearly with the target loss.
    CHECK(r.mahalanobis_distance ==
          Approx(target / r.factor_model_vol).epsilon(1e-12));
  }
}

TEST_CASE("reverse stress returns the minimum-norm solution") {
  // Any other factor move reaching the same loss must be further from the
  // centre of the distribution, measured in Mahalanobis distance.
  const std::vector<std::string> factors = {"f1", "f2"};
  Eigen::MatrixXd fcov(2, 2);
  fcov << 4e-4, 1e-4, 1e-4, 9e-4;
  FactorBetas betas = {
      {"A", {{"f1", 1.0}, {"f2", 0.2}}},
      {"B", {{"f1", 0.5}, {"f2", -0.4}}},
      {"C", {{"f1", 0.0}, {"f2", 1.5}}},
  };
  const Portfolio p = three_asset_book();
  const auto r = reverse_stress(p, betas, factors, fcov, 0.10);

  Eigen::MatrixXd B(3, 2);
  B << 1.0, 0.2, 0.5, -0.4, 0.0, 1.5;
  const Eigen::VectorXd b = B.transpose() * p.weights();
  const Eigen::MatrixXd fcov_inv = fcov.inverse();
  const double best = r.factor_move.dot(fcov_inv * r.factor_move);

  // Walk along the constraint surface: any move of the form f* + t*d with
  // b'd = 0 also produces the target loss.
  Eigen::VectorXd d(2);
  d << b(1), -b(0);  // orthogonal to b, so b'd = 0
  for (double t : {-2.0, -0.5, 0.5, 2.0}) {
    const Eigen::VectorXd alt = r.factor_move + t * d;
    CHECK(p.weights().dot(B * alt) == Approx(-0.10).epsilon(1e-10));
    CHECK(alt.dot(fcov_inv * alt) > best);
  }
}

TEST_CASE("reverse stress reports an unreachable loss honestly") {
  // A book with no factor exposure cannot lose anything through the factors.
  const std::vector<std::string> factors = {"f1"};
  const Eigen::MatrixXd fcov = Eigen::MatrixXd::Identity(1, 1) * 1e-4;
  const Portfolio p = three_asset_book();
  const auto r = reverse_stress(p, /*betas=*/{}, factors, fcov, 0.10);
  CHECK(std::isinf(r.mahalanobis_distance));
  CHECK(r.gaussian_probability == Approx(0.0));
}

TEST_CASE("reverse stress validates its inputs") {
  const Portfolio p = three_asset_book();
  const std::vector<std::string> factors = {"f1"};
  CHECK_THROWS(
      reverse_stress(p, {}, factors, Eigen::MatrixXd::Identity(2, 2), 0.1));
  CHECK_THROWS(
      reverse_stress(p, {}, factors, Eigen::MatrixXd::Identity(1, 1), 0.0));
}
