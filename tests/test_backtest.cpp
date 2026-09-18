#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>

#include "risk/backtest.hpp"
#include "risk/var.hpp"

using Catch::Approx;
using namespace risk;

namespace {

// Build a return series with exceptions at exactly the requested indices.
// Non-exception days sit at -0.001, comfortably inside a VaR of 0.02.
Eigen::VectorXd series_with_exceptions(int n, const std::vector<int>& at) {
  Eigen::VectorXd r = Eigen::VectorXd::Constant(n, -0.001);
  for (int i : at) r(i) = -0.05;
  return r;
}

constexpr double kVar = 0.02;

}  // namespace

// ---------------------------------------------------------------------------
// Distribution helpers, against values that are known in closed form.
// ---------------------------------------------------------------------------

TEST_CASE("chi-square survival matches closed forms") {
  // chi2(2) has survival exp(-x/2) exactly.
  for (double x : {0.5, 1.0, 2.5, 6.0, 11.0}) {
    CHECK(chi_square_sf(x, 2) == Approx(std::exp(-x / 2.0)).epsilon(1e-12));
  }
  // chi2(1) has survival erfc(sqrt(x/2)) exactly.
  for (double x : {0.1, 1.0, 3.841458820694124, 10.0}) {
    CHECK(chi_square_sf(x, 1) ==
          Approx(std::erfc(std::sqrt(x / 2.0))).epsilon(1e-10));
  }
  // The conventional 95% critical values must land on p = 0.05.
  CHECK(chi_square_sf(3.841458820694124, 1) == Approx(0.05).epsilon(1e-9));
  CHECK(chi_square_sf(5.991464547107979, 2) == Approx(0.05).epsilon(1e-9));
  CHECK(chi_square_sf(0.0, 1) == Approx(1.0));
  CHECK_THROWS(chi_square_sf(1.0, 0));
}

TEST_CASE("binomial CDF matches a direct sum") {
  const int n = 20;
  const double p = 0.3;
  double cumulative = 0.0;
  for (int k = 0; k <= n; ++k) {
    const double pmf = std::exp(std::lgamma(n + 1.0) - std::lgamma(k + 1.0) -
                                std::lgamma(n - k + 1.0)) *
                       std::pow(p, k) * std::pow(1.0 - p, n - k);
    cumulative += pmf;
    CHECK(binomial_cdf(k, n, p) == Approx(cumulative).epsilon(1e-12));
  }
  CHECK(binomial_cdf(-1, 10, 0.5) == Approx(0.0));
  CHECK(binomial_cdf(10, 10, 0.5) == Approx(1.0));
}

// ---------------------------------------------------------------------------
// Kupiec
// ---------------------------------------------------------------------------

TEST_CASE("Kupiec statistic is zero when the exception rate is exact") {
  // 5 exceptions in 100 days at 95% is exactly the expected rate, so the
  // restricted and unrestricted likelihoods coincide and LR must be 0.
  const auto r = kupiec_pof(series_with_exceptions(100, {3, 20, 44, 61, 90}),
                            kVar, 0.95);
  CHECK(r.exceptions == 5);
  CHECK(r.observed_rate == Approx(0.05));
  CHECK(r.lr_statistic == Approx(0.0).margin(1e-12));
  CHECK(r.p_value == Approx(1.0));
  CHECK_FALSE(r.reject_at_95);
}

TEST_CASE("Kupiec matches a hand-computed likelihood ratio") {
  // 15 exceptions in 250 days against an expected 5% (12.5).
  std::vector<int> at;
  for (int i = 0; i < 15; ++i) at.push_back(i * 16);
  const auto r = kupiec_pof(series_with_exceptions(250, at), kVar, 0.95);

  const double p = 0.05;
  const double pi = 15.0 / 250.0;
  const double expected =
      -2.0 * ((235 * std::log(1 - p) + 15 * std::log(p)) -
              (235 * std::log(1 - pi) + 15 * std::log(pi)));
  CHECK(r.exceptions == 15);
  CHECK(r.lr_statistic == Approx(expected).epsilon(1e-12));
  // 6% against 5% on 250 days is well within noise.
  CHECK_FALSE(r.reject_at_95);
  CHECK(r.p_value > 0.05);
}

TEST_CASE("Kupiec rejects a badly miscalibrated VaR") {
  // 40 exceptions in 250 days at 99% is 16%, sixteen times the advertised rate.
  std::vector<int> at;
  for (int i = 0; i < 40; ++i) at.push_back(i * 6);
  const auto r = kupiec_pof(series_with_exceptions(250, at), kVar, 0.99);
  CHECK(r.exceptions == 40);
  CHECK(r.reject_at_95);
  CHECK(r.p_value < 1e-10);
}

TEST_CASE("Kupiec handles the zero-exception corner") {
  // No exceptions at all: the log-likelihood has a 0*log(0) term that must
  // resolve to 0 rather than NaN.
  const auto r = kupiec_pof(series_with_exceptions(250, {}), kVar, 0.99);
  CHECK(r.exceptions == 0);
  CHECK(std::isfinite(r.lr_statistic));
  CHECK(r.lr_statistic == Approx(-2.0 * 250 * std::log(0.99)).epsilon(1e-12));
}

// ---------------------------------------------------------------------------
// Christoffersen
// ---------------------------------------------------------------------------

TEST_CASE("Christoffersen independence is ~zero for spread-out exceptions") {
  // Evenly spaced exceptions never follow one another, which is close to the
  // i.i.d. null for a low exception rate.
  std::vector<int> at;
  for (int i = 0; i < 10; ++i) at.push_back(i * 25 + 5);
  const auto r = christoffersen(series_with_exceptions(250, at), kVar, 0.95);
  CHECK(r.n11 == 0);
  CHECK(r.pi11 == Approx(0.0));
  CHECK(r.lr_independence < 2.0);
  CHECK_FALSE(r.reject_independence_at_95);
}

TEST_CASE("Christoffersen catches clustered exceptions that Kupiec cannot") {
  // The same 12 exceptions, scattered versus in one consecutive run. Kupiec
  // sees an identical count and returns an identical statistic; only the
  // independence test distinguishes them. This is the whole point of the test.
  std::vector<int> spread;
  for (int i = 0; i < 12; ++i) spread.push_back(i * 20 + 3);
  std::vector<int> clustered;
  for (int i = 0; i < 12; ++i) clustered.push_back(100 + i);

  const auto ks = kupiec_pof(series_with_exceptions(250, spread), kVar, 0.95);
  const auto kc = kupiec_pof(series_with_exceptions(250, clustered), kVar, 0.95);
  CHECK(ks.lr_statistic == Approx(kc.lr_statistic));

  const auto cs = christoffersen(series_with_exceptions(250, spread), kVar, 0.95);
  const auto cc =
      christoffersen(series_with_exceptions(250, clustered), kVar, 0.95);
  CHECK(cc.n11 == 11);
  CHECK(cc.pi11 > 0.9);
  CHECK(cc.lr_independence > cs.lr_independence);
  CHECK(cc.reject_independence_at_95);
  CHECK(cc.reject_conditional_coverage_at_95);
}

TEST_CASE("conditional coverage is the sum of the two statistics") {
  std::vector<int> at;
  for (int i = 0; i < 9; ++i) at.push_back(i * 13);
  const Eigen::VectorXd r = series_with_exceptions(250, at);
  const auto k = kupiec_pof(r, kVar, 0.95);
  const auto c = christoffersen(r, kVar, 0.95);
  CHECK(c.lr_conditional_coverage ==
        Approx(k.lr_statistic + c.lr_independence).epsilon(1e-12));
}

// ---------------------------------------------------------------------------
// Basel
// ---------------------------------------------------------------------------

TEST_CASE("Basel zones reproduce the published 250-day boundaries") {
  // The supervisory table: green 0-4, yellow 5-9, red 10 and above, at 99%
  // over 250 days. Here it falls out of the binomial CDF rather than being
  // hardcoded, so this test is checking the derivation, not a lookup.
  auto zone_for = [](int exceptions) {
    std::vector<int> at;
    for (int i = 0; i < exceptions; ++i) at.push_back(i * 2);
    return basel_traffic_light(series_with_exceptions(250, at), kVar, 0.99)
        .zone;
  };
  for (int k = 0; k <= 4; ++k) CHECK(zone_for(k) == BaselZone::Green);
  for (int k = 5; k <= 9; ++k) CHECK(zone_for(k) == BaselZone::Yellow);
  for (int k = 10; k <= 14; ++k) CHECK(zone_for(k) == BaselZone::Red);
}

TEST_CASE("Basel looks at the most recent window only") {
  // 12 exceptions, all in the first 250 days of a 600-day sample. The trailing
  // 250-day window is clean, so the zone must be green.
  std::vector<int> at;
  for (int i = 0; i < 12; ++i) at.push_back(i * 5);
  const auto r = basel_traffic_light(series_with_exceptions(600, at), kVar, 0.99);
  CHECK(r.exceptions == 0);
  CHECK(r.zone == BaselZone::Green);
  CHECK(r.window_complete);
}

TEST_CASE("Basel flags an incomplete window") {
  const auto r = basel_traffic_light(series_with_exceptions(100, {1, 2}), kVar,
                                     0.99, 250);
  CHECK_FALSE(r.window_complete);
  CHECK(r.window == 250);
}

// ---------------------------------------------------------------------------
// Acerbi-Szekely
// ---------------------------------------------------------------------------

TEST_CASE("Acerbi-Szekely Z is near zero on data from the assumed model") {
  // Draw from exactly the Gaussian the test assumes, with VaR and ES set to
  // their analytic values. Both statistics should sit near zero and neither
  // p-value should be in the rejection region.
  const double sigma = 0.01;
  const double conf = 0.975;
  const double z = normal_ppf(conf);
  const double var_level = z * sigma;
  const double es_level = sigma * normal_pdf(z) / (1.0 - conf);

  std::mt19937_64 gen(20260918);
  std::normal_distribution<double> nd(0.0, sigma);
  Eigen::VectorXd r(4000);
  for (Eigen::Index i = 0; i < r.size(); ++i) r(i) = nd(gen);

  const auto out = acerbi_szekely(r, var_level, es_level, conf, 0.0, sigma,
                                  2000, 7);
  CHECK(out.z1_defined);
  CHECK(std::abs(out.z1) < 0.15);
  CHECK(std::abs(out.z2) < 0.15);
  CHECK(out.p_value_z1 > 0.01);
  CHECK(out.p_value_z2 > 0.01);
}

TEST_CASE("Acerbi-Szekely Z goes negative when the tail is understated") {
  // Data is Student-t with 3 degrees of freedom scaled to the same variance,
  // but VaR and ES are still the Gaussian numbers. The realised tail is much
  // deeper than the model claims, so both statistics must go negative and
  // reject.
  const double sigma = 0.01;
  const double conf = 0.975;
  const double z = normal_ppf(conf);
  const double var_level = z * sigma;
  const double es_level = sigma * normal_pdf(z) / (1.0 - conf);

  std::mt19937_64 gen(20260918);
  std::student_t_distribution<double> td(3.0);
  const double scale = sigma / std::sqrt(3.0 / (3.0 - 2.0));  // unit variance
  Eigen::VectorXd r(4000);
  for (Eigen::Index i = 0; i < r.size(); ++i) r(i) = td(gen) * scale;

  const auto out = acerbi_szekely(r, var_level, es_level, conf, 0.0, sigma,
                                  2000, 7);
  CHECK(out.z1 < 0.0);
  CHECK(out.z2 < 0.0);
  CHECK(out.p_value_z2 < 0.05);
}

TEST_CASE("backtest input validation") {
  const Eigen::VectorXd r = series_with_exceptions(100, {5});
  CHECK_THROWS(kupiec_pof(r, kVar, 0.0));
  CHECK_THROWS(kupiec_pof(r, kVar, 1.0));
  CHECK_THROWS(christoffersen(Eigen::VectorXd::Zero(1), kVar, 0.95));
  CHECK_THROWS(basel_traffic_light(r, kVar, 0.99, 0));
  CHECK_THROWS(acerbi_szekely(r, kVar, 0.0, 0.95, 0.0, 0.01));
  CHECK_THROWS(acerbi_szekely(r, kVar, 0.03, 0.95, 0.0, 0.01, 0));
}
