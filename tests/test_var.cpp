#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>

#include "risk/var.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace risk;

namespace {
// Draw `n` standard-normal-scaled returns N(mu, sigma) with a fixed seed.
Eigen::VectorXd gaussian_sample(int n, double mu, double sigma,
                                std::uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::normal_distribution<double> nd(mu, sigma);
  Eigen::VectorXd v(n);
  for (int i = 0; i < n; ++i) v(i) = nd(gen);
  return v;
}

double sample_mean(const Eigen::VectorXd& v) {
  return v.mean();
}
double sample_std(const Eigen::VectorXd& v) {
  const double m = v.mean();
  return std::sqrt((v.array() - m).square().sum() / (v.size() - 1));
}
}  // namespace

TEST_CASE("normal_cdf and normal_ppf are accurate", "[var][normal]") {
  CHECK_THAT(normal_cdf(0.0), WithinAbs(0.5, 1e-12));
  CHECK_THAT(normal_cdf(1.959963985), WithinAbs(0.975, 1e-9));

  CHECK_THAT(normal_ppf(0.95), WithinAbs(1.6448536269514722, 1e-9));
  CHECK_THAT(normal_ppf(0.99), WithinAbs(2.3263478740408408, 1e-9));
  CHECK_THAT(normal_ppf(0.975), WithinAbs(1.959963984540054, 1e-9));

  // Round trip Phi(Phi^{-1}(p)) = p.
  for (double p : {0.01, 0.25, 0.5, 0.83, 0.99})
    CHECK_THAT(normal_cdf(normal_ppf(p)), WithinAbs(p, 1e-10));
}

TEST_CASE("parametric VaR formula and sqrt-time scaling", "[var][parametric]") {
  // mean 0, sigma 1% -> 95% VaR = z_0.95 * 0.01.
  CHECK_THAT(parametric_var(0.0, 0.01, 0.95),
             WithinRel(normal_ppf(0.95) * 0.01, 1e-12));
  // 99% > 95%.
  CHECK(parametric_var(0.0, 0.01, 0.99) > parametric_var(0.0, 0.01, 0.95));
  // 10-day / 1-day = sqrt(10) when mean is 0.
  const double v1 = parametric_var(0.0, 0.01, 0.99, 1);
  const double v10 = parametric_var(0.0, 0.01, 0.99, 10);
  CHECK_THAT(v10 / v1, WithinRel(std::sqrt(10.0), 1e-12));
  // Positive drift reduces VaR.
  CHECK(parametric_var(0.002, 0.01, 0.95) < parametric_var(0.0, 0.01, 0.95));
}

TEST_CASE("empirical quantile uses nth_element correctly", "[var][quantile]") {
  std::vector<double> data = {5, 1, 4, 2, 3, 0, 9, 7, 6, 8};  // 0..9
  CHECK_THAT(empirical_quantile(data, 0.0), WithinAbs(0.0, 1e-12));
  std::vector<double> d2 = {5, 1, 4, 2, 3, 0, 9, 7, 6, 8};
  // q=0.5 -> k = floor(0.5*10)=5 -> 5th smallest (0-indexed) = 5.
  CHECK_THAT(empirical_quantile(d2, 0.5), WithinAbs(5.0, 1e-12));
}

TEST_CASE("on Gaussian data the three VaR methods agree", "[var][gaussian]") {
  const int n = 200'000;
  const double mu = 0.0003, sigma = 0.012;
  Eigen::VectorXd r = gaussian_sample(n, mu, sigma, /*seed=*/12345);

  const double conf = 0.95;
  const double hist = historical_var(r, conf);
  const double para = parametric_var(sample_mean(r), sample_std(r), conf);

  Eigen::VectorXd mean1(1);
  mean1 << sample_mean(r);
  Eigen::MatrixXd cov1(1, 1);
  cov1 << sample_std(r) * sample_std(r);
  Eigen::VectorXd w1(1);
  w1 << 1.0;
  const double mc = monte_carlo_var(mean1, cov1, w1, conf, 1, 200'000, 999);

  // All three within 5% of each other.
  CHECK_THAT(hist, WithinRel(para, 0.05));
  CHECK_THAT(mc, WithinRel(para, 0.05));
  CHECK_THAT(hist, WithinRel(mc, 0.05));
}

TEST_CASE("on heavy-tailed data parametric underestimates the 99% tail",
          "[var][heavytail]") {
  // Student-t(5): finite variance, fat tails. Standardized 1% quantile (~-2.61)
  // is deeper than the Gaussian's (-2.326), so the empirical 99% VaR exceeds
  // the parametric one. (Monte Carlo here is Gaussian, so it tracks parametric;
  // the divergence is historical-vs-Gaussian, which is the whole point.)
  const int n = 500'000;
  std::mt19937_64 gen(2024);
  std::student_t_distribution<double> td(5.0);
  Eigen::VectorXd r(n);
  for (int i = 0; i < n; ++i) r(i) = 0.01 * td(gen);  // scale into return units

  const double conf = 0.99;
  const double hist = historical_var(r, conf);
  const double para = parametric_var(sample_mean(r), sample_std(r), conf);

  // Parametric underestimates the fat tail by a clear margin.
  CHECK(hist > para * 1.05);
}

TEST_CASE("Monte Carlo VaR converges to the analytic Gaussian value",
          "[var][montecarlo]") {
  Eigen::VectorXd mean(1);
  mean << 0.0005;
  Eigen::MatrixXd cov(1, 1);
  cov << 0.01 * 0.01;
  Eigen::VectorXd w(1);
  w << 1.0;
  const double conf = 0.99;
  const double analytic = parametric_var(0.0005, 0.01, conf);

  const double mc_small = monte_carlo_var(mean, cov, w, conf, 1, 1'000, 7);
  const double mc_large = monte_carlo_var(mean, cov, w, conf, 1, 400'000, 7);

  // Large-N estimate is close to the analytic value...
  CHECK_THAT(mc_large, WithinRel(analytic, 0.02));
  // ...and closer than the small-N estimate.
  CHECK(std::abs(mc_large - analytic) < std::abs(mc_small - analytic));
}

TEST_CASE("Monte Carlo multi-period sim respects sqrt-time for a linear book",
          "[var][montecarlo]") {
  Eigen::VectorXd mean(1);
  mean << 0.0;  // zero drift so sqrt-time is exact in expectation
  Eigen::MatrixXd cov(1, 1);
  cov << 0.01 * 0.01;
  Eigen::VectorXd w(1);
  w << 1.0;

  const double mc1 = monte_carlo_var(mean, cov, w, 0.99, 1, 400'000, 55);
  const double mc10 = monte_carlo_var(mean, cov, w, 0.99, 10, 400'000, 55);
  CHECK_THAT(mc10 / mc1, WithinRel(std::sqrt(10.0), 0.05));
}

TEST_CASE("VaR input validation", "[var][errors]") {
  Eigen::VectorXd r = Eigen::VectorXd::Ones(10);
  CHECK_THROWS_AS(historical_var(r, 1.5), std::invalid_argument);
  CHECK_THROWS_AS(historical_var(r, 0.95, 0), std::invalid_argument);
  CHECK_THROWS_AS(parametric_var(0.0, -1.0, 0.95), std::invalid_argument);
  CHECK_THROWS_AS(normal_ppf(0.0), std::invalid_argument);
  CHECK_THROWS_AS(normal_ppf(1.0), std::invalid_argument);
}
