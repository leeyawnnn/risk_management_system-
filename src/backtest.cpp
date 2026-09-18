#include "risk/backtest.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "risk/random.hpp"
#include "risk/var.hpp"

namespace risk {

namespace {

// Regularized lower incomplete gamma P(a, x) by series expansion. Converges
// quickly for x < a + 1.
double gamma_p_series(double a, double x) {
  constexpr int kMaxIter = 1000;
  constexpr double kEps = 1e-15;
  double ap = a;
  double sum = 1.0 / a;
  double del = sum;
  for (int n = 0; n < kMaxIter; ++n) {
    ap += 1.0;
    del *= x / ap;
    sum += del;
    if (std::abs(del) < std::abs(sum) * kEps) break;
  }
  return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

// Regularized upper incomplete gamma Q(a, x) by the modified Lentz continued
// fraction. Converges quickly for x >= a + 1.
double gamma_q_continued_fraction(double a, double x) {
  constexpr int kMaxIter = 1000;
  constexpr double kEps = 1e-15;
  constexpr double kTiny = 1e-300;
  double b = x + 1.0 - a;
  double c = 1.0 / kTiny;
  double d = 1.0 / b;
  double h = d;
  for (int i = 1; i <= kMaxIter; ++i) {
    const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
    b += 2.0;
    d = an * d + b;
    if (std::abs(d) < kTiny) d = kTiny;
    c = b + an / c;
    if (std::abs(c) < kTiny) c = kTiny;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::abs(del - 1.0) < kEps) break;
  }
  return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

double gamma_q(double a, double x) {
  if (x < 0.0 || a <= 0.0) {
    throw std::invalid_argument("gamma_q: require a > 0 and x >= 0");
  }
  if (x == 0.0) return 1.0;
  if (x < a + 1.0) return 1.0 - gamma_p_series(a, x);
  return gamma_q_continued_fraction(a, x);
}

// The log-likelihood ratio terms below all take the form n * log(p), which is
// 0 * log(0) whenever a transition never occurred. The limit is 0, but the
// expression is NaN, so every call routes through here.
double n_log_p(long n, double p) {
  if (n == 0) return 0.0;
  if (p <= 0.0) return 0.0;
  return static_cast<double>(n) * std::log(p);
}

std::vector<bool> exception_flags(const Eigen::VectorXd& returns,
                                  double var_level) {
  std::vector<bool> hits(static_cast<std::size_t>(returns.size()));
  for (Eigen::Index t = 0; t < returns.size(); ++t) {
    hits[static_cast<std::size_t>(t)] = returns(t) < -var_level;
  }
  return hits;
}

void validate(const Eigen::VectorXd& returns, double confidence) {
  if (returns.size() < 2) {
    throw std::invalid_argument("backtest: need at least 2 observations");
  }
  if (!(confidence > 0.0 && confidence < 1.0)) {
    throw std::invalid_argument("backtest: confidence must be in (0, 1)");
  }
}

}  // namespace

double chi_square_sf(double x, int df) {
  if (df < 1) throw std::invalid_argument("chi_square_sf: df must be >= 1");
  if (x <= 0.0) return 1.0;
  return gamma_q(0.5 * static_cast<double>(df), 0.5 * x);
}

double binomial_cdf(int k, int n, double p) {
  if (n < 0) throw std::invalid_argument("binomial_cdf: n must be >= 0");
  if (!(p >= 0.0 && p <= 1.0)) {
    throw std::invalid_argument("binomial_cdf: p must be in [0, 1]");
  }
  if (k < 0) return 0.0;
  if (k >= n) return 1.0;
  // Summed in log space: at n = 250 the individual terms are fine, but the
  // same routine is used for multi-year windows where they are not.
  double total = 0.0;
  const double log_p =
      (p > 0.0) ? std::log(p) : -std::numeric_limits<double>::infinity();
  const double log_q =
      (p < 1.0) ? std::log1p(-p) : -std::numeric_limits<double>::infinity();
  for (int i = 0; i <= k; ++i) {
    const double log_choose =
        std::lgamma(n + 1.0) - std::lgamma(i + 1.0) - std::lgamma(n - i + 1.0);
    const double term = log_choose + static_cast<double>(i) * log_p +
                        static_cast<double>(n - i) * log_q;
    total += std::exp(term);
  }
  return std::min(1.0, total);
}

KupiecResult kupiec_pof(const Eigen::VectorXd& returns, double var_level,
                        double confidence) {
  validate(returns, confidence);
  const auto hits = exception_flags(returns, var_level);
  const long T = static_cast<long>(hits.size());
  const long x = std::count(hits.begin(), hits.end(), true);

  KupiecResult r;
  r.observations = T;
  r.exceptions = x;
  r.expected_rate = 1.0 - confidence;
  r.observed_rate = static_cast<double>(x) / static_cast<double>(T);

  const double p = r.expected_rate;
  const double pi_hat = r.observed_rate;

  // LR_uc = -2 [ log L(p) - log L(pi_hat) ], each a binomial log-likelihood.
  const double ll_null = n_log_p(T - x, 1.0 - p) + n_log_p(x, p);
  const double ll_alt = n_log_p(T - x, 1.0 - pi_hat) + n_log_p(x, pi_hat);
  r.lr_statistic = -2.0 * (ll_null - ll_alt);
  if (r.lr_statistic < 0.0) r.lr_statistic = 0.0;  // round-off at the optimum

  r.p_value = chi_square_sf(r.lr_statistic, 1);
  r.reject_at_95 = r.lr_statistic > r.critical_value_95;
  return r;
}

ChristoffersenResult christoffersen(const Eigen::VectorXd& returns,
                                    double var_level, double confidence) {
  validate(returns, confidence);
  const auto hits = exception_flags(returns, var_level);

  ChristoffersenResult r;
  for (std::size_t t = 1; t < hits.size(); ++t) {
    const bool prev = hits[t - 1];
    const bool cur = hits[t];
    if (!prev && !cur)
      ++r.n00;
    else if (!prev && cur)
      ++r.n01;
    else if (prev && !cur)
      ++r.n10;
    else
      ++r.n11;
  }

  const long row0 = r.n00 + r.n01;
  const long row1 = r.n10 + r.n11;
  const long total = row0 + row1;
  r.pi01 =
      (row0 > 0) ? static_cast<double>(r.n01) / static_cast<double>(row0) : 0.0;
  r.pi11 =
      (row1 > 0) ? static_cast<double>(r.n11) / static_cast<double>(row1) : 0.0;
  const double pi = (total > 0) ? static_cast<double>(r.n01 + r.n11) /
                                      static_cast<double>(total)
                                : 0.0;

  // Null: exceptions are i.i.d. with probability pi.
  // Alternative: first-order Markov with separate pi01 and pi11.
  const double ll_null =
      n_log_p(r.n00 + r.n10, 1.0 - pi) + n_log_p(r.n01 + r.n11, pi);
  const double ll_alt = n_log_p(r.n00, 1.0 - r.pi01) + n_log_p(r.n01, r.pi01) +
                        n_log_p(r.n10, 1.0 - r.pi11) + n_log_p(r.n11, r.pi11);
  r.lr_independence = -2.0 * (ll_null - ll_alt);
  if (r.lr_independence < 0.0) r.lr_independence = 0.0;

  r.p_value_independence = chi_square_sf(r.lr_independence, 1);
  r.reject_independence_at_95 = r.lr_independence > 3.841458820694124;

  const KupiecResult k = kupiec_pof(returns, var_level, confidence);
  r.lr_conditional_coverage = k.lr_statistic + r.lr_independence;
  r.p_value_conditional_coverage = chi_square_sf(r.lr_conditional_coverage, 2);
  // chi2(2) at 95%.
  r.reject_conditional_coverage_at_95 =
      r.lr_conditional_coverage > 5.991464547107979;
  return r;
}

const char* to_string(BaselZone zone) {
  switch (zone) {
    case BaselZone::Green:
      return "green";
    case BaselZone::Yellow:
      return "yellow";
    case BaselZone::Red:
      return "red";
  }
  return "unknown";
}

BaselResult basel_traffic_light(const Eigen::VectorXd& returns,
                                double var_level, double confidence,
                                long window) {
  validate(returns, confidence);
  if (window < 1) {
    throw std::invalid_argument("basel_traffic_light: window must be >= 1");
  }

  BaselResult r;
  r.window = window;
  const long T = static_cast<long>(returns.size());
  const long n = std::min(T, window);
  r.window_complete = (T >= window);

  // The supervisory test looks at the most recent `window` observations.
  long exceptions = 0;
  for (Eigen::Index t = returns.size() - n; t < returns.size(); ++t) {
    if (returns(t) < -var_level) ++exceptions;
  }
  r.exceptions = exceptions;

  const double p = 1.0 - confidence;
  r.cumulative_probability =
      binomial_cdf(static_cast<int>(exceptions), static_cast<int>(n), p);

  if (r.cumulative_probability < 0.95) {
    r.zone = BaselZone::Green;
  } else if (r.cumulative_probability < 0.9999) {
    r.zone = BaselZone::Yellow;
  } else {
    r.zone = BaselZone::Red;
  }

  // The supervisory plus factors are published as a table indexed by the
  // exception count over 250 days at 99%, and they are graduated inside the
  // yellow zone rather than flat. They are only defined for that window and
  // that confidence, so outside it the multiplier stays at the 3.0 base and
  // `plus_factor_applicable` says the increment does not apply -- inventing
  // an interpolation for an arbitrary window would be worse than declining.
  const bool supervisory = (window == 250) &&
                           (std::abs(confidence - 0.99) < 1e-12) &&
                           r.window_complete;
  r.plus_factor_applicable = supervisory;
  if (supervisory) {
    switch (exceptions) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
        r.plus_factor = 0.00;
        break;
      case 5:
        r.plus_factor = 0.40;
        break;
      case 6:
        r.plus_factor = 0.50;
        break;
      case 7:
        r.plus_factor = 0.65;
        break;
      case 8:
        r.plus_factor = 0.75;
        break;
      case 9:
        r.plus_factor = 0.85;
        break;
      default:
        r.plus_factor = 1.00;
        break;  // 10 or more: red zone
    }
  }
  r.capital_multiplier = 3.0 + r.plus_factor;
  return r;
}

AcerbiSzekelyResult acerbi_szekely(const Eigen::VectorXd& returns,
                                   double var_level, double es_level,
                                   double confidence, double mean, double stdev,
                                   int simulations, std::uint64_t seed) {
  validate(returns, confidence);
  if (simulations < 1) {
    throw std::invalid_argument("acerbi_szekely: simulations must be >= 1");
  }
  if (!(es_level > 0.0)) {
    throw std::invalid_argument("acerbi_szekely: es_level must be positive");
  }

  const double p = 1.0 - confidence;
  const long T = static_cast<long>(returns.size());

  // Both statistics share the same accumulator: the sum over exception days of
  // (return / ES). Returns are negative in the tail and ES is a positive loss,
  // so each term is negative and the sums below sit near -1 before the +1.
  //
  // The realised statistic uses the VaR and ES the model reported. The null
  // simulation uses the VaR and ES implied by the predictive distribution it
  // draws from, which are not the same thing and must not be conflated.
  // Feeding the reported levels into the null as well makes both sides shift
  // together whenever the reported ES is wrong, and the p-value then cannot
  // detect a mis-stated ES at all -- which is the one thing this test exists
  // to do.
  auto statistics = [&](const Eigen::VectorXd& x, double var_used,
                        double es_used) {
    double acc = 0.0;
    long hits = 0;
    for (Eigen::Index t = 0; t < x.size(); ++t) {
      if (x(t) < -var_used) {
        acc += x(t) / es_used;
        ++hits;
      }
    }
    const double z1 = (hits > 0) ? acc / static_cast<double>(hits) + 1.0
                                 : std::numeric_limits<double>::quiet_NaN();
    const double z2 = acc / (static_cast<double>(x.size()) * p) + 1.0;
    return std::tuple<double, double, long>{z1, z2, hits};
  };

  AcerbiSzekelyResult r;
  r.observations = T;
  r.simulations = simulations;
  const auto [z1, z2, hits] = statistics(returns, var_level, es_level);
  r.exceptions = hits;
  r.z2 = z2;
  r.z1_defined = hits > 0;
  r.z1 = r.z1_defined ? z1 : 0.0;

  // Neither statistic has a closed-form null distribution, so simulate it:
  // draw T returns from the Gaussian predictive distribution, recompute, and
  // count how often the simulated statistic falls at or below the realised
  // one. The tail of interest is the left one -- a NEGATIVE Z means realised
  // losses beyond VaR were worse than the model said.
  // The Gaussian predictive distribution's own VaR and ES. Under the null the
  // data comes from N(mean, stdev), so these are the levels a correct model
  // would have reported; comparing the realised statistic against a null
  // built on them is what makes a wrong ES visible.
  const double z_alpha = normal_ppf(confidence);
  const double null_var = z_alpha * stdev - mean;
  const double null_es = stdev * normal_pdf(z_alpha) / p - mean;

  std::mt19937_64 gen(seed);
  long le1 = 0, le2 = 0, defined1 = 0;
  Eigen::VectorXd sim(T);
  for (int s = 0; s < simulations; ++s) {
    for (Eigen::Index t = 0; t < T; ++t) {
      sim(t) = mean + stdev * standard_normal(gen);
    }
    const auto [s1, s2, shits] = statistics(sim, null_var, null_es);
    if (shits > 0) {
      ++defined1;
      if (r.z1_defined && s1 <= r.z1) ++le1;
    }
    if (s2 <= r.z2) ++le2;
  }
  r.p_value_z1 = (r.z1_defined && defined1 > 0)
                     ? static_cast<double>(le1) / static_cast<double>(defined1)
                     : 1.0;
  r.p_value_z2 = static_cast<double>(le2) / static_cast<double>(simulations);
  return r;
}

}  // namespace risk
