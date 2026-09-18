#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>

#include "risk/cvar.hpp"
#include "risk/var.hpp"
#include "test_support.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace risk;

namespace {
Eigen::VectorXd gaussian_sample(int n, double mu, double sigma,
                                std::uint64_t seed) {
  std::mt19937_64 gen(seed);
    Eigen::VectorXd v(n);
  for (int i = 0; i < n; ++i) v(i) = risk_test::normal(gen, mu, sigma);
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

TEST_CASE("tail_mean averages the worst tail", "[cvar][tail]") {
  std::vector<double> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // tail_prob 0.3 -> k=3 worst = {0,1,2} -> mean 1.
  CHECK_THAT(tail_mean(data, 0.3), WithinAbs(1.0, 1e-12));
  std::vector<double> d2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // tail_prob 0 -> still at least the single worst (0).
  CHECK_THAT(tail_mean(d2, 0.0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("parametric CVaR matches the closed form and exceeds VaR",
          "[cvar][parametric]") {
  const double mu = 0.0, sigma = 0.01, conf = 0.95;
  const double z = normal_ppf(conf);
  const double expected = -mu + sigma * normal_pdf(z) / (1.0 - conf);

  CHECK_THAT(parametric_cvar(mu, sigma, conf), WithinRel(expected, 1e-12));
  // CVaR > VaR in magnitude.
  CHECK(parametric_cvar(mu, sigma, conf) > parametric_var(mu, sigma, conf));
}

TEST_CASE("parametric CVaR matches the empirical Gaussian tail mean",
          "[cvar][parametric]") {
  // Cross-check the closed form against a brute-force simulation tail mean.
  Eigen::VectorXd r = gaussian_sample(1'000'000, 0.0003, 0.012, 4242);
  const double conf = 0.95;
  const double closed = parametric_cvar(sample_mean(r), sample_std(r), conf);
  const double empirical = historical_cvar(r, conf);
  CHECK_THAT(empirical, WithinRel(closed, 0.03));
}

TEST_CASE("CVaR >= VaR for all three methods", "[cvar][ordering]") {
  Eigen::VectorXd r = gaussian_sample(200'000, 0.0002, 0.011, 777);
  const double conf = 0.99;

  // Historical.
  CHECK(historical_cvar(r, conf) >= historical_var(r, conf));

  // Parametric.
  const double m = sample_mean(r), s = sample_std(r);
  CHECK(parametric_cvar(m, s, conf) >= parametric_var(m, s, conf));

  // Monte Carlo (single synthetic asset).
  Eigen::VectorXd mean(1);
  mean << m;
  Eigen::MatrixXd cov(1, 1);
  cov << s * s;
  Eigen::VectorXd w(1);
  w << 1.0;
  CHECK(monte_carlo_cvar(mean, cov, w, conf, 1, 200'000, 9) >=
        monte_carlo_var(mean, cov, w, conf, 1, 200'000, 9));
}

TEST_CASE("on Gaussian data the three CVaR methods agree", "[cvar][gaussian]") {
  Eigen::VectorXd r = gaussian_sample(300'000, 0.0003, 0.012, 31337);
  const double conf = 0.95;
  const double m = sample_mean(r), s = sample_std(r);

  const double hist = historical_cvar(r, conf);
  const double para = parametric_cvar(m, s, conf);

  Eigen::VectorXd mean(1);
  mean << m;
  Eigen::MatrixXd cov(1, 1);
  cov << s * s;
  Eigen::VectorXd w(1);
  w << 1.0;
  const double mc = monte_carlo_cvar(mean, cov, w, conf, 1, 300'000, 5);

  CHECK_THAT(hist, WithinRel(para, 0.04));
  CHECK_THAT(mc, WithinRel(para, 0.04));
}

TEST_CASE("CVaR sqrt-time scaling", "[cvar][horizon]") {
  const double c1 = parametric_cvar(0.0, 0.01, 0.99, 1);
  const double c10 = parametric_cvar(0.0, 0.01, 0.99, 10);
  CHECK_THAT(c10 / c1, WithinRel(std::sqrt(10.0), 1e-12));
}

TEST_CASE("heavy tails widen the CVaR/VaR gap", "[cvar][heavytail]") {
  // Student-t(5): the tail-mean/quantile ratio is larger than Gaussian's,
  // so empirical CVaR/VaR exceeds the parametric (Gaussian) ratio of ~1.15.
  std::mt19937_64 gen(2025);
    Eigen::VectorXd r(500'000);
  for (Eigen::Index i = 0; i < r.size(); ++i) r(i) = 0.01 * risk_test::student_t(gen, 5);

  const double conf = 0.99;
  const double ratio = historical_cvar(r, conf) / historical_var(r, conf);
  const double gaussian_ratio =
      parametric_cvar(0.0, 1.0, conf) / parametric_var(0.0, 1.0, conf);
  CHECK(ratio > gaussian_ratio);
}
