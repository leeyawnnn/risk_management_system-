#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "risk/random.hpp"
#include "risk/var.hpp"
#include "test_support.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace risk;

namespace {
// Draw `n` standard-normal-scaled returns N(mu, sigma) with a fixed seed.
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
  Eigen::VectorXd r(n);
  for (int i = 0; i < n; ++i)
    r(i) = 0.01 * risk_test::student_t(gen, 5);  // scale into return units

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

// ---------------------------------------------------------------------------
// Reproducibility of the random path
// ---------------------------------------------------------------------------

// Square brackets are Catch2 tag syntax, so they must not appear in a test
// NAME: catch_discover_tests parses the name as tags and collapses the whole
// suite into one bogus entry. That is what broke the offline CI leg, whose
// packaged Catch2 ships an older discovery script.
TEST_CASE("uniform_unit covers the unit interval without reaching one",
          "[var][random][reproducibility]") {
  CHECK(risk::uniform_unit(0) == 0.0);
  CHECK(risk::uniform_unit(~std::uint64_t{0}) < 1.0);
  CHECK(risk::uniform_unit(~std::uint64_t{0}) > 1.0 - 1e-15);
  // The open variant must exclude BOTH endpoints: normal_ppf rejects 0 and 1
  // and Phi^{-1} is infinite at each. The all-ones case is the one that bit:
  // the natural half-ulp offset rounds up to exactly 1.0 in double.
  CHECK(risk::uniform_open_unit(0) > 0.0);
  CHECK(risk::uniform_open_unit(~std::uint64_t{0}) < 1.0);
  CHECK_NOTHROW(risk::normal_ppf(risk::uniform_open_unit(0)));
  CHECK_NOTHROW(risk::normal_ppf(risk::uniform_open_unit(~std::uint64_t{0})));
  // Sweep the extremes of the engine's output range for the same reason.
  for (std::uint64_t b : {std::uint64_t{0}, std::uint64_t{1}, ~std::uint64_t{0},
                          ~std::uint64_t{0} - 1, std::uint64_t{1} << 63}) {
    const double u = risk::uniform_open_unit(b);
    INFO("bits = " << b << " -> u = " << u);
    CHECK(u > 0.0);
    CHECK(u < 1.0);
  }
}

TEST_CASE("uniform_below is in range and unbiased",
          "[var][random][reproducibility]") {
  std::mt19937_64 gen(1234);
  std::vector<int> counts(7, 0);
  const int draws = 70000;
  for (int i = 0; i < draws; ++i) {
    const std::uint64_t v = risk::uniform_below(gen, 7);
    REQUIRE(v < 7);
    ++counts[static_cast<std::size_t>(v)];
  }
  // Each bucket should land near draws/7 = 10000; 5 sigma of a binomial with
  // p = 1/7 on 70000 draws is about 490.
  for (int c : counts) {
    CHECK(std::abs(c - 10000) < 600);
  }
  CHECK(risk::uniform_below(gen, 1) == 0);
  CHECK(risk::uniform_below(gen, 0) == 0);
}

TEST_CASE("standard_normal has the right moments and a fixed sequence",
          "[var][random][reproducibility]") {
  std::mt19937_64 gen(20260918);
  const int n = 200000;
  double sum = 0.0;
  double sumsq = 0.0;
  double sum4 = 0.0;
  for (int i = 0; i < n; ++i) {
    const double z = risk::standard_normal(gen);
    sum += z;
    sumsq += z * z;
    sum4 += z * z * z * z;
  }
  const double mean = sum / n;
  const double var = sumsq / n - mean * mean;
  CHECK_THAT(mean, WithinAbs(0.0, 0.01));
  CHECK_THAT(var, WithinRel(1.0, 0.02));
  // Excess kurtosis of a standard normal is 0.
  CHECK_THAT(sum4 / n - 3.0, WithinAbs(0.0, 0.1));
}

TEST_CASE("the same seed reproduces the same draws exactly",
          "[var][random][reproducibility]") {
  // This is the property the committed report depends on. std::normal_
  // distribution does NOT have it across standard libraries, which is why
  // this project does not use it.
  auto sequence = [](std::uint64_t seed, int n) {
    std::mt19937_64 gen(seed);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) out.push_back(risk::standard_normal(gen));
    return out;
  };
  CHECK(sequence(7, 64) == sequence(7, 64));
  CHECK(sequence(7, 64) != sequence(8, 64));

  // And the sequence is a pure function of the engine's bit stream, so it can
  // be reproduced from first principles without calling into this project.
  std::mt19937_64 gen(7);
  const std::uint64_t bits = gen();
  const double expected =
      risk::normal_ppf(static_cast<double>(2 * (bits >> 12) + 1) * 0x1.0p-53);
  std::mt19937_64 gen2(7);
  CHECK(risk::standard_normal(gen2) == expected);
}

TEST_CASE("Monte Carlo VaR is stable across repeated runs",
          "[var][random][reproducibility]") {
  Eigen::MatrixXd cov(2, 2);
  cov << 1e-4, 2e-5, 2e-5, 9e-5;
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd w(2);
  w << 0.6, 0.4;
  const double a = risk::monte_carlo_var(mean, cov, w, 0.99, 1, 50000, 11);
  const double b = risk::monte_carlo_var(mean, cov, w, 0.99, 1, 50000, 11);
  CHECK(a == b);
  CHECK(a != risk::monte_carlo_var(mean, cov, w, 0.99, 1, 50000, 12));
}
