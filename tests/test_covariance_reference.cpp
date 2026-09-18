// Cross-check the C++ covariance estimators against an independent Python
// reference implementation.
//
// The literals below are the output of `python3
// scripts/covariance_reference.py` on the fixed 8x3 return matrix defined in
// that script, printed at full double precision. The sample covariance row is
// numpy.cov itself, so that one is a check against a third-party implementation
// rather than a second copy of our own reasoning.
//
// Regenerate with:
//     python3 scripts/covariance_reference.py
//
// A previous version of this suite only checked the estimators for
// self-consistency (symmetry, PSD, correct shape), which would have passed
// against a wrong formula.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>
#include <algorithm>

#include "risk/covariance.hpp"

using Catch::Matchers::WithinRel;
using namespace risk;

namespace {

// The same matrix as scripts/covariance_reference.py: T = 8, N = 3.
Eigen::MatrixXd reference_returns() {
  Eigen::MatrixXd x(8, 3);
  x << 0.012, -0.004, 0.006, -0.008, 0.011, -0.002, 0.015, -0.009, 0.004, 0.003,
      0.002, -0.007, -0.011, 0.006, 0.010, 0.007, -0.003, -0.001, -0.005, 0.008,
      0.003, 0.010, -0.006, -0.004;
  return x;
}

// Relative tolerance for a value that travelled through a print/parse round
// trip at 15 significant figures.
constexpr double kTol = 1e-12;

void expect_matches(const Eigen::MatrixXd& got, const Eigen::MatrixXd& want) {
  REQUIRE(got.rows() == want.rows());
  REQUIRE(got.cols() == want.cols());
  for (Eigen::Index i = 0; i < want.rows(); ++i) {
    for (Eigen::Index j = 0; j < want.cols(); ++j) {
      INFO("entry (" << i << ", " << j << ")");
      CHECK_THAT(got(i, j), WithinRel(want(i, j), kTol));
    }
  }
}

}  // namespace

TEST_CASE("sample covariance matches numpy.cov", "[covariance][reference]") {
  Eigen::MatrixXd want(3, 3);
  want << 9.583928571428571e-05, -6.662499999999999e-05, -1.012500000000000e-05,
      -6.662499999999999e-05, 5.198214285714285e-05, 1.339285714285714e-06,
      -1.012500000000000e-05, 1.339285714285714e-06, 3.155357142857143e-05;
  expect_matches(sample_covariance(reference_returns()), want);
}

TEST_CASE("EWMA covariance matches the reference implementation",
          "[covariance][reference]") {
  Eigen::MatrixXd want(3, 3);
  want << 8.104912065884224e-05, -5.650706832637623e-05, -1.282621742437049e-05,
      -5.650706832637623e-05, 4.429090660398489e-05, 4.197848622365670e-06,
      -1.282621742437049e-05, 4.197848622365672e-06, 2.743939393686789e-05;
  expect_matches(ewma_covariance(reference_returns(), 0.94), want);
}

TEST_CASE("Ledoit-Wolf matches the reference implementation",
          "[covariance][reference]") {
  const auto lw = ledoit_wolf_covariance(reference_returns());

  // Shrinkage intensity and average correlation, from the same script.
  CHECK_THAT(lw.shrinkage, WithinRel(0.31584158774752125, kTol));
  CHECK_THAT(lw.avg_correlation, WithinRel(-0.36499232244088214, kTol));

  Eigen::MatrixXd want(3, 3);
  want << 8.385937500000000e-05, -4.700396916829431e-05, -1.160820285428923e-05,
      -4.700396916829431e-05, 4.548437500000000e-05, -3.283443617676087e-06,
      -1.160820285428923e-05, -3.283443617676087e-06, 2.760937500000000e-05;
  expect_matches(lw.cov, want);
}

TEST_CASE("Ledoit-Wolf reproduces equation (5) of the paper from its parts",
          "[covariance][reference]") {
  // Ledoit & Wolf (2004), Appendix B: delta* = max{0, min{kappa/T, 1}} with
  // kappa = (pi - rho) / gamma. Recomputing pi, rho and gamma here from the
  // definitions and checking they reproduce the shrinkage the estimator
  // returned tests the formula rather than the code path that implements it.
  const Eigen::MatrixXd X = reference_returns();
  const auto lw = ledoit_wolf_covariance(X);

  const Eigen::Index T = X.rows();
  const Eigen::Index N = X.cols();
  const auto Td = static_cast<double>(T);

  const Eigen::RowVectorXd mean = X.colwise().mean();
  const Eigen::MatrixXd Xc = X.rowwise() - mean;
  const Eigen::MatrixXd S = (Xc.transpose() * Xc) / Td;  // the MLE, not /(T-1)
  const Eigen::VectorXd var = S.diagonal();
  const Eigen::VectorXd sd = var.cwiseSqrt();

  // r-bar: the average off-diagonal sample correlation.
  double sum_corr = 0.0;
  long pairs = 0;
  for (Eigen::Index i = 0; i < N; ++i) {
    for (Eigen::Index j = i + 1; j < N; ++j) {
      sum_corr += S(i, j) / (sd(i) * sd(j));
      ++pairs;
    }
  }
  const double rbar = sum_corr / static_cast<double>(pairs);
  CHECK_THAT(rbar, WithinRel(lw.avg_correlation, 1e-14));

  // The constant-correlation target F.
  Eigen::MatrixXd F = rbar * (sd * sd.transpose());
  F.diagonal() = var;

  // pi-hat: the summed asymptotic variances of the sample covariance entries.
  Eigen::MatrixXd pi_mat = Eigen::MatrixXd::Zero(N, N);
  for (Eigen::Index t = 0; t < T; ++t) {
    const Eigen::RowVectorXd xt = Xc.row(t);
    const Eigen::MatrixXd d = (xt.transpose() * xt) - S;
    pi_mat += d.cwiseProduct(d);
  }
  pi_mat /= Td;
  const double pi_hat = pi_mat.sum();

  // rho-hat: the diagonal of pi plus the paper's off-diagonal cross terms.
  double rho_hat = pi_mat.diagonal().sum();
  for (Eigen::Index i = 0; i < N; ++i) {
    for (Eigen::Index j = 0; j < N; ++j) {
      if (i == j) continue;
      double theta_ii = 0.0;
      double theta_jj = 0.0;
      for (Eigen::Index t = 0; t < T; ++t) {
        const double xi = Xc(t, i);
        const double xj = Xc(t, j);
        const double dij = xi * xj - S(i, j);
        theta_ii += (xi * xi - S(i, i)) * dij;
        theta_jj += (xj * xj - S(j, j)) * dij;
      }
      theta_ii /= Td;
      theta_jj /= Td;
      rho_hat +=
          rbar * 0.5 * (sd(j) / sd(i) * theta_ii + sd(i) / sd(j) * theta_jj);
    }
  }

  const double gamma_hat = (F - S).squaredNorm();
  const double kappa = (pi_hat - rho_hat) / gamma_hat;
  const double delta = std::max(0.0, std::min(kappa / Td, 1.0));

  CHECK_THAT(delta, WithinRel(lw.shrinkage, 1e-14));

  // And equation (2): Sigma_shrink = delta* F + (1 - delta*) S.
  const Eigen::MatrixXd rebuilt = delta * F + (1.0 - delta) * S;
  expect_matches(lw.cov, rebuilt);
}

TEST_CASE("shrinkage intensity stays inside the unit interval",
          "[covariance][reference]") {
  // The clip in equation (5) is not decoration: kappa/T runs outside [0, 1]
  // for short samples, and an unclipped delta would produce a matrix that is
  // neither the sample estimate nor the target.
  const auto lw = ledoit_wolf_covariance(reference_returns());
  CHECK(lw.shrinkage >= 0.0);
  CHECK(lw.shrinkage <= 1.0);
}
