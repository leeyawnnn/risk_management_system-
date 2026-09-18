// How precisely is the 99% VaR actually known, and what do fat tails do to it?
//
// A previous version of this project's README claimed that historical VaR
// exceeded parametric VaR at 99% "because the real return distribution has
// fatter tails than the Gaussian assumes". On the data it shipped -- Gaussian
// by construction -- that was false: excess kurtosis was zero and the gap was
// sampling noise in roughly the thirteenth-worst observation of 1,260.
//
// This executable replaces that claim with two measurements.
//
// 1. Bootstrap the historical VaR and CVaR. At 99% on ~1,238 observations only
//    about twelve returns sit beyond the threshold, so the estimator has a
//    standard error worth several basis points. Resampling the return series
//    with replacement gives a confidence interval, and whether the parametric
//    estimate falls inside it settles whether any gap is real. The sample
//    moments -- skewness and excess kurtosis -- are reported alongside so the
//    fat-tail question is answered by measurement rather than assertion.
//
// 2. Run the same three VaR methods on a Student-t sample calibrated to the
//    same variance. There the fat tails are known to be present and known by
//    how much, so the divergence the Gaussian methods miss can be quantified
//    instead of asserted.
//
// Output: reports/tail_study.json and reports/tail_bootstrap.csv.

#include <nlohmann/json.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "risk/config.hpp"
#include "risk/cvar.hpp"
#include "risk/portfolio.hpp"
#include "risk/var.hpp"

namespace fs = std::filesystem;
using namespace risk;

namespace {

struct Args {
  std::string portfolio = "config/portfolio.json";
  std::string data = "data/returns/";
  std::string output = "reports/";
  int resamples = 10000;
  std::uint64_t seed = 20260918;
  double student_t_df = 4.0;
};

void usage() {
  std::cout
      << "Usage: tail_study [options]\n"
      << "  --portfolio FILE  portfolio config (default "
         "config/portfolio.json)\n"
      << "  --data DIR        return CSV directory (default data/returns/)\n"
      << "  --output DIR      output directory (default reports/)\n"
      << "  --resamples N     bootstrap resamples (default 10000)\n"
      << "  --df X            Student-t degrees of freedom (default 4)\n"
      << "  --seed N          RNG seed (default 20260918)\n";
}

struct Moments {
  double mean = 0.0;
  double stdev = 0.0;
  double skewness = 0.0;
  double excess_kurtosis = 0.0;
};

Moments moments_of(const Eigen::VectorXd& x) {
  const auto n = static_cast<double>(x.size());
  Moments m;
  m.mean = x.mean();
  const Eigen::ArrayXd d = x.array() - m.mean;
  const double m2 = d.square().sum() / n;
  m.stdev = std::sqrt(d.square().sum() / (n - 1.0));
  m.skewness = (d.cube().sum() / n) / std::pow(m2, 1.5);
  m.excess_kurtosis = (d.pow(4).sum() / n) / (m2 * m2) - 3.0;
  return m;
}

// Percentile of an already-sorted vector, by linear interpolation.
double percentile_sorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) throw std::invalid_argument("percentile: empty");
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = static_cast<std::size_t>(std::ceil(pos));
  if (lo == hi) return sorted[lo];
  const double frac = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

struct BootstrapInterval {
  double point = 0.0;
  double lower = 0.0;  // 2.5th percentile
  double upper = 0.0;  // 97.5th percentile
  double standard_error = 0.0;
};

// Resample the return series with replacement and recompute the statistic.
// The i.i.d. bootstrap ignores volatility clustering, which means these
// intervals are, if anything, too narrow on real data -- stated in the README
// rather than glossed over.
template <typename Stat>
BootstrapInterval bootstrap(const Eigen::VectorXd& returns, int resamples,
                            std::uint64_t seed, Stat stat) {
  const auto n = static_cast<std::size_t>(returns.size());
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);

  std::vector<double> draws;
  draws.reserve(static_cast<std::size_t>(resamples));
  Eigen::VectorXd sample(returns.size());
  for (int b = 0; b < resamples; ++b) {
    for (std::size_t i = 0; i < n; ++i) {
      sample(static_cast<Eigen::Index>(i)) =
          returns(static_cast<Eigen::Index>(pick(gen)));
    }
    draws.push_back(stat(sample));
  }
  std::ranges::sort(draws);

  BootstrapInterval out;
  out.point = stat(returns);
  out.lower = percentile_sorted(draws, 0.025);
  out.upper = percentile_sorted(draws, 0.975);

  const double mean = std::accumulate(draws.begin(), draws.end(), 0.0) /
                      static_cast<double>(draws.size());
  double ss = 0.0;
  for (double d : draws) ss += (d - mean) * (d - mean);
  out.standard_error = std::sqrt(ss / static_cast<double>(draws.size() - 1));
  return out;
}

}  // namespace

int main(int argc, char** argv) try {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + flag);
      }
      return argv[++i];
    };
    if (s == "--portfolio")
      args.portfolio = next("--portfolio");
    else if (s == "--data")
      args.data = next("--data");
    else if (s == "--output")
      args.output = next("--output");
    else if (s == "--resamples")
      args.resamples = std::stoi(next("--resamples"));
    else if (s == "--df")
      args.student_t_df = std::stod(next("--df"));
    else if (s == "--seed")
      args.seed = std::stoull(next("--seed"));
    else if (s == "--help" || s == "-h") {
      usage();
      return 0;
    } else
      throw std::invalid_argument("unknown argument: " + s);
  }
  if (args.resamples < 100) {
    throw std::invalid_argument("--resamples must be at least 100");
  }
  if (!(args.student_t_df > 4.0 - 1e-12)) {
    // Below 4 degrees of freedom the kurtosis of a Student-t is undefined, so
    // the comparison table would print a meaningless number.
    throw std::invalid_argument("--df must be at least 4 for finite kurtosis");
  }

  const EngineConfig cfg = load_engine_config(args.portfolio);
  const Portfolio portfolio = make_portfolio(cfg);
  const Eigen::MatrixXd X = load_return_matrix(cfg, args.data);
  const Eigen::MatrixXd cov =
      estimate_covariance(cfg.covariance_method, X, cfg.ewma_lambda);
  const Eigen::VectorXd r = portfolio.return_series(X);
  const Eigen::VectorXd asset_mean = X.colwise().mean().transpose();

  const Moments m = moments_of(r);
  const double sigma = portfolio.volatility(cov);

  nlohmann::json out;
  out["sample"] = {
      {"observations", r.size()},
      {"mean_daily", m.mean},
      {"stdev_daily", m.stdev},
      {"skewness", m.skewness},
      {"excess_kurtosis", m.excess_kurtosis},
      {"model_vol_daily", sigma},
      {"covariance_method", cfg.covariance_method},
  };

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "portfolio return sample\n"
            << "  observations       " << r.size() << "\n"
            << "  daily vol          " << m.stdev * 100.0 << "%\n"
            << "  skewness           " << m.skewness << "\n"
            << "  excess kurtosis    " << m.excess_kurtosis << "\n\n";

  // ---- bootstrap the historical estimators --------------------------------
  std::ofstream csv(fs::path(args.output) / "tail_bootstrap.csv");
  fs::create_directories(args.output);
  csv << std::setprecision(10);
  csv << "measure,confidence,point,ci_lower,ci_upper,standard_error,"
         "parametric,parametric_inside_ci\n";

  nlohmann::json boots = nlohmann::json::array();
  std::cout << "bootstrap of the historical estimator (" << args.resamples
            << " resamples)\n";
  std::cout << "  measure   conf    point      95% CI              "
               "parametric   inside?\n";

  for (double conf : cfg.confidences) {
    struct Case {
      const char* label;
      double parametric;
      double (*stat)(const Eigen::VectorXd&, double, int);
    };
    const std::array<Case, 2> cases = {
        Case{.label = "VaR",
             .parametric = parametric_var(m.mean, sigma, conf, 1),
             .stat = &historical_var},
        Case{.label = "CVaR",
             .parametric = parametric_cvar(m.mean, sigma, conf, 1),
             .stat = &historical_cvar},
    };

    for (const auto& c : cases) {
      const auto bi = bootstrap(
          r, args.resamples, args.seed,
          [&](const Eigen::VectorXd& s) { return c.stat(s, conf, 1); });
      const bool inside = c.parametric >= bi.lower && c.parametric <= bi.upper;

      csv << c.label << ',' << conf << ',' << bi.point << ',' << bi.lower << ','
          << bi.upper << ',' << bi.standard_error << ',' << c.parametric << ','
          << (inside ? "true" : "false") << '\n';

      boots.push_back({{"measure", c.label},
                       {"confidence", conf},
                       {"historical", bi.point},
                       {"ci_lower", bi.lower},
                       {"ci_upper", bi.upper},
                       {"standard_error", bi.standard_error},
                       {"parametric", c.parametric},
                       {"parametric_inside_ci", inside}});

      std::cout << "  " << std::setw(8) << std::left << c.label << std::right
                << std::setw(5) << static_cast<int>(conf * 100) << "%  "
                << std::setw(7) << bi.point * 100.0 << "%  [" << std::setw(6)
                << bi.lower * 100.0 << "%, " << std::setw(6) << bi.upper * 100.0
                << "%]  " << std::setw(9) << c.parametric * 100.0 << "%   "
                << (inside ? "yes" : "NO") << "\n";
    }
  }
  out["bootstrap"] = boots;
  out["bootstrap_resamples"] = args.resamples;

  // ---- the controlled fat-tail comparison ---------------------------------
  // Draw Student-t returns with the SAME variance as the real book. The
  // variance of a t with v degrees of freedom is v/(v-2), so the scale that
  // matches sigma is sigma / sqrt(v/(v-2)). Everything the methods disagree
  // about is then attributable to tail shape alone.
  //
  // The reference tail is evaluated on a very large simulated sample rather
  // than on a 1,238-day one. That distinction matters: estimating the
  // parametric number from the same short fat-tailed draw inflates it, because
  // a sample standard deviation computed over fat tails is itself dominated by
  // the few extreme observations. Comparing against that would have made the
  // Gaussian model look conservative when it is the opposite.
  const double df = args.student_t_df;
  const double t_scale = m.stdev / std::sqrt(df / (df - 2.0));
  const int reference_draws = 5'000'000;

  std::mt19937_64 gen(args.seed + 1);
  std::student_t_distribution<double> td(df);

  Eigen::VectorXd reference(reference_draws);
  for (Eigen::Index i = 0; i < reference.size(); ++i) {
    reference(i) = m.mean + td(gen) * t_scale;
  }
  // A short draw of the same process, to show what estimation noise alone
  // does at the sample size the real book actually has.
  Eigen::VectorXd short_draw(r.size());
  for (Eigen::Index i = 0; i < short_draw.size(); ++i) {
    short_draw(i) = m.mean + td(gen) * t_scale;
  }

  const Moments rm = moments_of(reference);
  nlohmann::json comparison = nlohmann::json::array();

  std::cout << "\ncontrolled fat tails: Student-t(" << df
            << ") at matched variance\n"
            << "  excess kurtosis    FRED sample " << m.excess_kurtosis
            << "   t reference " << rm.excess_kurtosis << "\n"
            << "  reference draws    " << reference_draws << "\n\n"
            << "  conf  measure   Gaussian model   true t tail   understated by"
               "   t est. on "
            << r.size() << "d\n";

  for (double conf : cfg.confidences) {
    // The Gaussian model's answer, using the matched (correct) sigma.
    const double g_var = parametric_var(m.mean, m.stdev, conf, 1);
    const double g_cvar = parametric_cvar(m.mean, m.stdev, conf, 1);
    // The truth for this process, to simulation precision.
    const double t_var = historical_var(reference, conf, 1);
    const double t_cvar = historical_cvar(reference, conf, 1);
    // What a sample the size of the real one would have told you.
    const double s_var = historical_var(short_draw, conf, 1);
    const double s_cvar = historical_cvar(short_draw, conf, 1);

    comparison.push_back({{"confidence", conf},
                          {"gaussian_var", g_var},
                          {"gaussian_cvar", g_cvar},
                          {"student_t_var", t_var},
                          {"student_t_cvar", t_cvar},
                          {"var_understatement", t_var / g_var - 1.0},
                          {"cvar_understatement", t_cvar / g_cvar - 1.0},
                          {"short_sample_var", s_var},
                          {"short_sample_cvar", s_cvar}});

    auto row = [&](const char* label, double g, double t, double sd) {
      std::cout << std::setw(6) << static_cast<int>(conf * 100) << "%  "
                << std::setw(8) << std::left << label << std::right
                << std::setw(13) << g * 100.0 << "%" << std::setw(13)
                << t * 100.0 << "%" << std::setw(15) << (t / g - 1.0) * 100.0
                << "%" << std::setw(14) << sd * 100.0 << "%\n";
    };
    row("VaR", g_var, t_var, s_var);
    row("CVaR", g_cvar, t_cvar, s_cvar);
  }
  out["student_t"] = {
      {"degrees_of_freedom", df},
      {"scale", t_scale},
      {"reference_draws", reference_draws},
      {"reference_excess_kurtosis", rm.excess_kurtosis},
      {"short_sample_size", r.size()},
      {"comparison", comparison},
  };

  // Monte Carlo is included for completeness: on a linear book with Gaussian
  // draws it must agree with the parametric number up to simulation error,
  // and showing that agreement is the point.
  out["monte_carlo_var99"] = monte_carlo_var(
      asset_mean, cov, portfolio.weights(), 0.99, 1, cfg.mc_draws, cfg.seed);
  out["seed"] = args.seed;

  std::ofstream js(fs::path(args.output) / "tail_study.json");
  js << out.dump(2) << "\n";
  std::cout << "\nwrote " << args.output << "tail_study.json and "
            << "tail_bootstrap.csv\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << "\n";
  return 1;
}
