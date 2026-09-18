// Compute the full risk report for a portfolio and write it to disk.
//
// Every output is stamped with the data's as-of date, the git commit of the
// code that produced it and the RNG seed, so a number in the README can
// always be traced back to a specific run.

#include <nlohmann/json.hpp>

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "risk/backtest.hpp"
#include "risk/config.hpp"
#include "risk/covariance.hpp"
#include "risk/decomposition.hpp"
#include "risk/portfolio.hpp"
#include "risk/reporter.hpp"
#include "risk/stress_test.hpp"
#include "risk/version.hpp"

namespace fs = std::filesystem;
using namespace risk;

namespace {

struct Args {
  std::string portfolio = "config/portfolio.json";
  std::string data = "data/returns/";
  std::string factors = "data/factors.csv";
  std::string output = "output/";
  std::string estimator_csv;
  std::string as_of;
  std::string git_commit;
  std::uint64_t seed = 0;
  bool seed_set = false;
};

void usage() {
  std::cout
      << "compute_risk " << kVersionString << "\n\n"
      << "Usage: compute_risk [options]\n"
      << "  --portfolio FILE   portfolio config (default "
         "config/portfolio.json)\n"
      << "  --data DIR         return CSV directory (default data/returns/)\n"
      << "  --factors FILE     factor series (default data/factors.csv)\n"
      << "  --output DIR       output directory (default output/)\n"
      << "  --estimator-csv F  estimator study output, for that one figure\n"
      << "  --seed N           RNG seed; overrides the config\n"
      << "  --as-of DATE       data as-of date stamped into every artifact.\n"
      << "                     Defaults to the as_of in data/manifest.json.\n"
      << "  --commit SHA       code commit stamped into every artifact.\n"
      << "                     Defaults to `git rev-parse HEAD`.\n"
      << "  --help             this message\n";
}

// Read a value out of data/manifest.json, so the as-of date comes from the
// data rather than from the clock. A report that stamps itself with the
// wall-clock time claims to be more current than its slowest input.
std::string manifest_as_of(const std::string& data_dir) {
  // `data_dir` usually arrives with a trailing separator, and parent_path()
  // on "data/returns/" returns "data/returns", not "data". Going up through
  // ".." and normalising handles both spellings.
  const fs::path path =
      (fs::path(data_dir) / ".." / "manifest.json").lexically_normal();
  std::ifstream in(path);
  if (!in) return {};
  try {
    nlohmann::json j;
    in >> j;
    return j.value("as_of", std::string{});
  } catch (const std::exception&) {
    return {};
  }
}

// Short git SHA of the working tree, or empty when git is unavailable or this
// is not a checkout. Never fabricated.
std::string git_head() {
  std::array<char, 128> buffer{};
  // NOLINTNEXTLINE(cert-env33-c): fixed command, no user input reaches it.
  FILE* pipe = popen("git rev-parse --short=10 HEAD 2>/dev/null", "r");
  if (pipe == nullptr) return {};
  std::string out;
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) !=
         nullptr) {
    out += buffer.data();
  }
  pclose(pipe);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
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
    if (s == "--portfolio") {
      args.portfolio = next("--portfolio");
    } else if (s == "--data") {
      args.data = next("--data");
    } else if (s == "--factors") {
      args.factors = next("--factors");
    } else if (s == "--output") {
      args.output = next("--output");
    } else if (s == "--estimator-csv") {
      args.estimator_csv = next("--estimator-csv");
    } else if (s == "--seed") {
      args.seed = std::stoull(next("--seed"));
      args.seed_set = true;
    } else if (s == "--as-of") {
      args.as_of = next("--as-of");
    } else if (s == "--commit") {
      args.git_commit = next("--commit");
    } else if (s == "--help" || s == "-h") {
      usage();
      return 0;
    } else {
      throw std::invalid_argument("unknown argument: " + s);
    }
  }

  const EngineConfig cfg = load_engine_config(args.portfolio);
  const Portfolio portfolio = make_portfolio(cfg);

  const Eigen::MatrixXd X = load_return_matrix(cfg, args.data);
  std::cout << "loaded " << cfg.assets.size() << " instruments x " << X.rows()
            << " returns from " << args.data << "\n";

  const Eigen::MatrixXd cov =
      estimate_covariance(cfg.covariance_method, X, cfg.ewma_lambda);
  const auto diag = diagnose_matrix(cov);
  if (!diag.psd) {
    std::cerr << "error: estimated covariance is not positive semi-definite "
                 "(min eigenvalue "
              << diag.min_eigenvalue << ")\n";
    return 1;
  }
  if (diag.ill_conditioned) {
    // Not fatal, but it must be said out loud rather than discovered later in
    // whatever the first ldlt() does with it.
    std::cerr << "warning: covariance condition number is "
              << diag.condition_number
              << "; results that involve inverting Sigma should be treated "
                 "with caution\n";
  }

  // ---- stress scenarios ----------------------------------------------------
  std::vector<StressResult> stress;
  FactorBetas betas;
  bool have_betas = false;
  if (!cfg.factor_betas_path.empty()) {
    betas = load_factor_betas(cfg.factor_betas_path);
    have_betas = !betas.empty();
  }
  if (!cfg.stress_scenarios_path.empty()) {
    const auto scenarios = load_scenarios(cfg.stress_scenarios_path);
    stress = apply_scenarios(portfolio, scenarios, betas);
  }

  // ---- factor series -------------------------------------------------------
  FactorSeries factors;
  bool have_factors = false;
  if (fs::exists(args.factors)) {
    factors = load_factor_series(args.factors);
    have_factors = factors.values.rows() == X.rows();
    if (!have_factors) {
      std::cerr << "warning: factor series has " << factors.values.rows()
                << " rows against " << X.rows()
                << " returns; skipping the factor sections rather than "
                   "aligning them silently\n";
    }
  }

  // ---- report --------------------------------------------------------------
  ReportInputs in;
  in.asset_returns = &X;
  in.cov = &cov;
  in.cov_method = cfg.covariance_method;
  in.annualization_factor = cfg.annualization_factor;
  in.confidences = cfg.confidences;
  in.horizons = cfg.horizons;
  in.stress = stress;
  in.sectors = cfg.sectors;
  in.mc_draws = cfg.mc_draws;
  in.seed = args.seed_set ? args.seed : cfg.seed;
  in.ewma_lambda = cfg.ewma_lambda;
  in.as_of = args.as_of.empty() ? manifest_as_of(args.data) : args.as_of;
  in.git_commit = args.git_commit.empty() ? git_head() : args.git_commit;
  if (have_factors && have_betas) {
    in.factors = &factors;
    in.betas = &betas;
    in.reverse_stress_targets = {0.05, 0.10, 0.20};
  }
  // A representative trade on the largest and smallest positions, priced by
  // full revaluation. These are illustrative sizes, stated as such.
  if (!cfg.assets.empty()) {
    in.trades.emplace_back(cfg.assets.front(), 1'000'000.0);
    in.trades.emplace_back(cfg.assets.back(), 1'000'000.0);
    in.trades.emplace_back(cfg.assets.front(), -1'000'000.0);
  }

  const RiskReport rep = build_report(cfg.name, portfolio, in);

  fs::create_directories(args.output);
  fs::create_directories(fs::path(args.output) / "figures");
  write_reports(rep, args.output);
  write_all_figures(rep, (fs::path(args.output) / "figures").string(),
                    args.estimator_csv);

  // ---- console summary -----------------------------------------------------
  std::cout << "\n=== " << cfg.name << " ===\n"
            << "as-of " << (rep.as_of.empty() ? "unstamped" : rep.as_of)
            << " | commit "
            << (rep.git_commit.empty() ? "unstamped" : rep.git_commit)
            << " | seed " << rep.seed << "\n"
            << "covariance " << cfg.covariance_method << " | notional "
            << cfg.notional << "\n"
            << "portfolio vol " << rep.daily_vol * 100 << "% daily / "
            << rep.annual_vol * 100 << "% annual\n"
            << "excess kurtosis " << rep.portfolio_excess_kurtosis
            << " | cond(corr) " << rep.correlation_diagnostics.condition_number
            << "\n\n";

  std::cout << "VaR/CVaR, 1 day:\n";
  for (const auto& r : rep.var_cvar) {
    if (r.horizon_days != 1) continue;
    std::cout << "  " << r.method << " " << static_cast<int>(r.confidence * 100)
              << "%  VaR " << r.var * 100 << "%  CVaR " << r.cvar * 100
              << "%\n";
  }

  std::cout << "\nbacktests:\n";
  for (const auto& b : rep.backtests) {
    std::cout << "  " << b.method << " " << static_cast<int>(b.confidence * 100)
              << "%  " << b.kupiec.exceptions << " exceptions"
              << "  Kupiec p=" << b.kupiec.p_value
              << (b.kupiec.reject_at_95 ? " REJECT" : " pass")
              << "  independence p=" << b.christoffersen.p_value_independence
              << (b.christoffersen.reject_independence_at_95 ? " REJECT"
                                                             : " pass")
              << "  Basel " << to_string(b.basel.zone) << "\n";
  }

  std::cout << "\neffective bets "
            << rep.attribution.concentration.effective_num_bets << " | top "
            << rep.attribution.concentration.max_contributor << " "
            << rep.attribution.concentration.max_contribution * 100 << "%\n";
  std::cout << "reports written to " << args.output << "\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << "\n";
  return 1;
}
