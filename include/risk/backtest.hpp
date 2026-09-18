#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <string>

namespace risk {

// ===========================================================================
// VaR and Expected Shortfall backtesting
// ===========================================================================
// A VaR number nobody has backtested is an opinion. These are the standard
// tests a risk desk and its regulator actually run.
//
// Sign convention matches the rest of the engine: `returns` are signed (a loss
// is negative), `var_level` and `es_level` are POSITIVE loss fractions. An
// exception on day t is `returns(t) < -var_level`.

// ---------------------------------------------------------------------------
// Distribution helpers
// ---------------------------------------------------------------------------

// Upper-tail probability of a chi-square with `df` degrees of freedom:
// P(X > x). Implemented via the regularized upper incomplete gamma function
// Q(df/2, x/2) with the usual series/continued-fraction split.
double chi_square_sf(double x, int df);

// P(X <= k) for X ~ Binomial(n, p). Used for the Basel zone boundaries, which
// are defined by binomial cumulative probability rather than by fixed counts.
double binomial_cdf(int k, int n, double p);

// ---------------------------------------------------------------------------
// Kupiec (1995) proportion-of-failures test — unconditional coverage
// ---------------------------------------------------------------------------
// Asks one question: did exceptions happen at the advertised rate? The
// likelihood ratio compares the observed failure rate against the expected
// (1 - confidence) under a binomial null. It is blind to *when* they happened.
struct KupiecResult {
  long observations = 0;
  long exceptions = 0;
  double expected_rate = 0.0;  // 1 - confidence
  double observed_rate = 0.0;
  double lr_statistic = 0.0;
  int degrees_of_freedom = 1;
  double critical_value_95 = 3.841458820694124;  // chi2(1) at 95%
  double p_value = 1.0;
  bool reject_at_95 = false;  // true => coverage is wrong
};

KupiecResult kupiec_pof(const Eigen::VectorXd& returns, double var_level,
                        double confidence);

// ---------------------------------------------------------------------------
// Christoffersen (1998) independence and conditional coverage
// ---------------------------------------------------------------------------
// Asks the question Kupiec cannot: do exceptions *cluster*? Five scattered
// breaches and five consecutive breaches give Kupiec the same answer, and only
// one of them empties the trading account. The independence test is a
// likelihood ratio against a first-order Markov alternative; the conditional
// coverage test is the sum of the two statistics, distributed chi2(2).
struct ChristoffersenResult {
  // Transition counts: n_ij = days following state i that were in state j,
  // where 1 = exception.
  long n00 = 0, n01 = 0, n10 = 0, n11 = 0;
  double pi01 = 0.0;  // P(exception | no exception yesterday)
  double pi11 = 0.0;  // P(exception | exception yesterday)
  double lr_independence = 0.0;
  double p_value_independence = 1.0;
  bool reject_independence_at_95 = false;

  double lr_conditional_coverage = 0.0;  // Kupiec LR + independence LR
  double p_value_conditional_coverage = 1.0;
  bool reject_conditional_coverage_at_95 = false;
};

ChristoffersenResult christoffersen(const Eigen::VectorXd& returns,
                                    double var_level, double confidence);

// ---------------------------------------------------------------------------
// Basel traffic-light zones
// ---------------------------------------------------------------------------
// The supervisory backstop: count exceptions over the most recent 250 trading
// days at 99% and read off a zone. Green means the model is accepted as is;
// yellow adds a capital multiplier and an explanation; red means the model is
// presumed broken.
//
// The boundaries are not magic numbers. A zone is defined by the binomial
// cumulative probability of the observed count: green below 95%, yellow below
// 99.99%, red at or above. At the regulatory window of 250 days and p = 1%
// that reproduces the published table (green 0-4, yellow 5-9, red 10+).
enum class BaselZone : std::uint8_t { Green, Yellow, Red };

struct BaselResult {
  long window = 250;
  long exceptions = 0;
  double cumulative_probability = 0.0;
  BaselZone zone = BaselZone::Green;
  // Scaling factor applied to the capital charge: 3.0 plus the supervisory
  // increment. The yellow zone is graduated by exception count rather than
  // flat -- 5 exceptions costs 0.40 and 9 costs 0.65 -- and reporting one
  // number for the whole zone understates a model at the top of it.
  // Defined only at the 250-day, 99% regulatory window; see
  // `plus_factor_applicable`.
  double capital_multiplier = 3.0;
  double plus_factor = 0.0;
  // False when the window or confidence is not the supervisory 250 days at
  // 99%, in which case the published increments do not apply and inventing
  // an interpolation would be worse than saying so.
  bool plus_factor_applicable = false;
  bool window_complete = true;  // false if fewer than `window` observations
};

BaselResult basel_traffic_light(const Eigen::VectorXd& returns,
                                double var_level, double confidence,
                                long window = 250);

const char* to_string(BaselZone zone);

// ---------------------------------------------------------------------------
// Acerbi-Szekely (2014) Expected Shortfall backtests
// ---------------------------------------------------------------------------
// ES is not elicitable: there is no scoring function it uniquely minimises, so
// it cannot be backtested by the direct hit-counting argument that makes VaR
// easy. Acerbi and Szekely's response is to test ES *conditionally on VaR*
// rather than on its own.
//
//   Z1 tests the magnitude of the losses that did breach VaR, given how many
//      breached. It says nothing about whether the breach count was right.
//   Z2 tests count and magnitude jointly against the unconditional tail.
//
// Both are centred at zero under the null. A NEGATIVE value means realised
// tail losses were worse than the model predicted, which is the direction that
// matters. Neither has a closed-form null distribution, so the p-value is
// simulated from the predictive distribution.
struct AcerbiSzekelyResult {
  long observations = 0;
  long exceptions = 0;
  double z1 = 0.0;
  double z2 = 0.0;
  double p_value_z1 = 1.0;
  double p_value_z2 = 1.0;
  int simulations = 0;
  bool z1_defined = false;  // false when there were no exceptions at all
};

// `var_level` and `es_level` are the levels the MODEL reported, and the
// realised statistic is computed against them. `mean` and `stdev` describe
// the Gaussian predictive distribution the null is simulated from, and the
// null statistic is computed against that distribution's OWN analytic VaR and
// ES. Keeping the two apart is what lets the p-value detect a reported ES
// that is inconsistent with the distribution it claims to come from.
AcerbiSzekelyResult acerbi_szekely(const Eigen::VectorXd& returns,
                                   double var_level, double es_level,
                                   double confidence, double mean, double stdev,
                                   int simulations = 10000,
                                   std::uint64_t seed = 42);

// ---------------------------------------------------------------------------
// One row of the combined backtest table: a method at a confidence level.
// ---------------------------------------------------------------------------
struct BacktestRow {
  std::string method;  // "Historical" | "Parametric" | "MonteCarlo"
  double confidence = 0.95;
  double var_level = 0.0;
  double es_level = 0.0;
  KupiecResult kupiec;
  ChristoffersenResult christoffersen;
  BaselResult basel;
  AcerbiSzekelyResult es_test;
};

}  // namespace risk
