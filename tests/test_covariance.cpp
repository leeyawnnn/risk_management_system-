#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <random>

#include "risk/covariance.hpp"
#include "test_support.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace risk;

namespace {

// The exact synthetic dataset from scripts/covariance_reference.py (T=8, N=3).
Eigen::MatrixXd synthetic_X() {
  Eigen::MatrixXd X(8, 3);
  X << 0.012, -0.004, 0.006, -0.008, 0.011, -0.002, 0.015, -0.009, 0.004, 0.003,
      0.002, -0.007, -0.011, 0.006, 0.010, 0.007, -0.003, -0.001, -0.005, 0.008,
      0.003, 0.010, -0.006, -0.004;
  return X;
}

// Compare every entry of two matrices to a tight absolute tolerance. The C++
// and numpy algorithms are identical, so only summation-order round-off
// differs.
void expect_matrix_eq(const Eigen::MatrixXd& got, const Eigen::MatrixXd& ref,
                      double tol = 1e-13) {
  REQUIRE(got.rows() == ref.rows());
  REQUIRE(got.cols() == ref.cols());
  for (Eigen::Index i = 0; i < got.rows(); ++i)
    for (Eigen::Index j = 0; j < got.cols(); ++j)
      CHECK_THAT(got(i, j), WithinAbs(ref(i, j), tol));
}

}  // namespace

TEST_CASE("is_symmetric / is_psd behave correctly", "[covariance][checks]") {
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3, 3);
  CHECK(is_symmetric(I));
  CHECK(is_psd(I));

  // A symmetric but indefinite matrix (eigenvalues +1, -1) must fail is_psd.
  Eigen::MatrixXd indef(2, 2);
  indef << 0.0, 1.0, 1.0, 0.0;
  CHECK(is_symmetric(indef));
  CHECK_FALSE(is_psd(indef));

  // A non-symmetric matrix fails both.
  Eigen::MatrixXd asym(2, 2);
  asym << 1.0, 2.0, 0.0, 1.0;
  CHECK_FALSE(is_symmetric(asym));
  CHECK_FALSE(is_psd(asym));
}

TEST_CASE("sample covariance matches numpy.cov (ddof=1)",
          "[covariance][sample]") {
  Eigen::MatrixXd ref(3, 3);
  ref << 9.583928571428571e-05, -6.662499999999999e-05, -1.012500000000000e-05,
      -6.662499999999999e-05, 5.198214285714285e-05, 1.339285714285714e-06,
      -1.012500000000000e-05, 1.339285714285714e-06, 3.155357142857143e-05;

  Eigen::MatrixXd S = sample_covariance(synthetic_X());
  expect_matrix_eq(S, ref);
  CHECK(is_symmetric(S));
  CHECK(is_psd(S));
}

TEST_CASE("EWMA covariance matches the RiskMetrics reference",
          "[covariance][ewma]") {
  Eigen::MatrixXd ref(3, 3);
  ref << 8.104912065884224e-05, -5.650706832637623e-05, -1.282621742437049e-05,
      -5.650706832637623e-05, 4.429090660398489e-05, 4.197848622365670e-06,
      -1.282621742437049e-05, 4.197848622365672e-06, 2.743939393686789e-05;

  Eigen::MatrixXd EW = ewma_covariance(synthetic_X(), 0.94);
  expect_matrix_eq(EW, ref);
  CHECK(is_psd(EW));
}

TEST_CASE("EWMA weights decay: recent observations weigh more",
          "[covariance][ewma]") {
  const double lambda = 0.94;
  Eigen::VectorXd w = ewma_weights(10, lambda);

  // Sum to 1.
  CHECK_THAT(w.sum(), WithinAbs(1.0, 1e-15));
  // Strictly increasing from oldest (index 0) to newest (index T-1).
  for (Eigen::Index t = 1; t < w.size(); ++t) CHECK(w(t) > w(t - 1));
  // Newest carries strictly more weight than oldest.
  CHECK(w(w.size() - 1) > w(0));
  // Consecutive ratio equals lambda (older / newer).
  for (Eigen::Index t = 1; t < w.size(); ++t)
    CHECK_THAT(w(t - 1) / w(t), WithinRel(lambda, 1e-12));
}

TEST_CASE("Ledoit-Wolf matches the constant-correlation reference",
          "[covariance][ledoitwolf]") {
  Eigen::MatrixXd ref(3, 3);
  ref << 8.385937500000000e-05, -4.700396916829431e-05, -1.160820285428923e-05,
      -4.700396916829431e-05, 4.548437500000000e-05, -3.283443617676087e-06,
      -1.160820285428923e-05, -3.283443617676087e-06, 2.760937500000000e-05;

  LedoitWolf lw = ledoit_wolf_covariance(synthetic_X());
  expect_matrix_eq(lw.cov, ref);
  CHECK_THAT(lw.shrinkage, WithinRel(0.31584158774752125, 1e-10));
  CHECK_THAT(lw.avg_correlation, WithinRel(-0.36499232244088214, 1e-10));
  CHECK(lw.shrinkage >= 0.0);
  CHECK(lw.shrinkage <= 1.0);
  CHECK(is_psd(lw.cov));
}

TEST_CASE("all three estimators are symmetric and PSD", "[covariance][psd]") {
  Eigen::MatrixXd X = synthetic_X();
  for (const Eigen::MatrixXd& M :
       {sample_covariance(X), ewma_covariance(X, 0.94),
        ledoit_wolf_covariance(X).cov}) {
    CHECK(is_symmetric(M));
    CHECK(is_psd(M));
  }
}

TEST_CASE("cov_to_correlation has unit diagonal and correct entries",
          "[covariance][corr]") {
  Eigen::MatrixXd S = sample_covariance(synthetic_X());
  Eigen::MatrixXd C = cov_to_correlation(S);
  for (Eigen::Index i = 0; i < C.rows(); ++i)
    CHECK_THAT(C(i, i), WithinAbs(1.0, 1e-15));
  // Off-diagonal matches s_ij / (sd_i sd_j).
  const double expected01 = S(0, 1) / (std::sqrt(S(0, 0)) * std::sqrt(S(1, 1)));
  CHECK_THAT(C(0, 1), WithinRel(expected01, 1e-12));
  CHECK(std::abs(C(0, 1)) <= 1.0);
}

TEST_CASE("shrinkage strictly improves conditioning",
          "[covariance][ledoitwolf]") {
  // The shrunk matrix should have a condition number no worse than the sample
  // covariance (shrinkage pulls eigenvalues toward the target).
  Eigen::MatrixXd X = synthetic_X();
  Eigen::MatrixXd S = sample_covariance(X);
  Eigen::MatrixXd LW = ledoit_wolf_covariance(X).cov;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_s(S), es_lw(LW);
  const double cond_s =
      es_s.eigenvalues().maxCoeff() / es_s.eigenvalues().minCoeff();
  const double cond_lw =
      es_lw.eigenvalues().maxCoeff() / es_lw.eigenvalues().minCoeff();
  CHECK(cond_lw <= cond_s);
}

TEST_CASE("matrix diagnostics report the spectrum and conditioning",
          "[covariance][conditioning]") {
  // A matrix with a known spectrum: diag(4, 1, 0.01) has condition number 400.
  Eigen::MatrixXd d = Eigen::MatrixXd::Zero(3, 3);
  d.diagonal() << 4.0, 1.0, 0.01;
  const auto diag = risk::diagnose_matrix(d);
  CHECK(diag.symmetric);
  CHECK(diag.psd);
  CHECK_THAT(diag.min_eigenvalue, WithinRel(0.01, 1e-12));
  CHECK_THAT(diag.max_eigenvalue, WithinRel(4.0, 1e-12));
  CHECK_THAT(diag.condition_number, WithinRel(400.0, 1e-12));
  CHECK_FALSE(diag.ill_conditioned);
}

TEST_CASE("a singular matrix reports infinite condition number",
          "[covariance][conditioning]") {
  // Two identical assets: the covariance is rank 1 and cannot be inverted.
  Eigen::MatrixXd s(2, 2);
  s << 1e-4, 1e-4, 1e-4, 1e-4;
  const auto diag = risk::diagnose_matrix(s);
  CHECK(diag.psd);
  CHECK(std::isinf(diag.condition_number));
  CHECK(diag.ill_conditioned);
}

TEST_CASE("shrinkage improves conditioning on a near-singular sample",
          "[covariance][conditioning]") {
  // Two near-duplicate assets plus a third, with barely more observations
  // than assets: exactly the regime the sample estimator handles worst.
  // Ledoit-Wolf must pull the smallest eigenvalue up and the condition
  // number down. This is the numerical statement of why shrinkage exists.
  std::mt19937_64 gen(12345);
  const int T = 12;
  Eigen::MatrixXd X(T, 3);
  for (int t = 0; t < T; ++t) {
    const double base = risk_test::normal(gen, 0.0, 0.01);
    X(t, 0) = base;
    X(t, 1) =
        base + 0.001 * risk_test::normal(
                           gen, 0.0, 0.01);  // almost a duplicate of column 0
    X(t, 2) = risk_test::normal(gen, 0.0, 0.01);
  }

  const auto sample = risk::diagnose_matrix(risk::sample_covariance(X));
  const auto shrunk =
      risk::diagnose_matrix(risk::ledoit_wolf_covariance(X).cov);

  CHECK(shrunk.condition_number < sample.condition_number);
  CHECK(shrunk.min_eigenvalue > sample.min_eigenvalue);
}
