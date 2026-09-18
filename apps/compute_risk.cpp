#include <nlohmann/json.hpp>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "risk/covariance.hpp"
#include "risk/cvar.hpp"
#include "risk/figures.hpp"
#include "risk/portfolio.hpp"
#include "risk/reporter.hpp"
#include "risk/return_series.hpp"
#include "risk/stress_test.hpp"
#include "risk/var.hpp"

namespace fs = std::filesystem;
using namespace risk;

namespace {

struct Args {
  std::string portfolio = "config/portfolio.json";
  std::string data = "data/returns/";
  std::string output = "output/";
};

void usage() {
  std::cout
      << "Usage: compute_risk [--portfolio FILE] [--data DIR] [--output DIR]\n"
      << "  --portfolio  portfolio JSON (default config/portfolio.json)\n"
      << "  --data       directory of <ASSET>.csv return files (default "
         "data/returns/)\n"
      << "  --output     output directory for reports + figures (default "
         "output/)\n";
}

bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (s == "--portfolio")
      a.portfolio = next("--portfolio");
    else if (s == "--data")
      a.data = next("--data");
    else if (s == "--output")
      a.output = next("--output");
    else if (s == "--help" || s == "-h") {
      usage();
      return false;
    } else
      throw std::invalid_argument("unknown argument: " + s);
  }
  return true;
}

ReturnType parse_return_type(const std::string& s) {
  if (s == "log") return ReturnType::Log;
  if (s == "simple") return ReturnType::Simple;
  throw std::invalid_argument("return_type must be \"log\" or \"simple\"");
}

MissingPolicy parse_missing(const std::string& s) {
  if (s == "skip") return MissingPolicy::Skip;
  if (s == "fill_forward" || s == "ffill") return MissingPolicy::FillForward;
  throw std::invalid_argument(
      "missing_policy must be \"skip\" or \"fill_forward\"");
}

}  // namespace

int main(int argc, char** argv) try {
  Args args;
  if (!parse_args(argc, argv, args)) return 0;

  // ---- load portfolio config ------------------------------------------------
  std::ifstream cfg_in(args.portfolio);
  if (!cfg_in) {
    std::cerr << "error: cannot open portfolio file " << args.portfolio << "\n";
    return 1;
  }
  nlohmann::json cfg;
  cfg_in >> cfg;

  const std::string name = cfg.value("name", "Portfolio");
  const double notional = cfg.value("notional", 1.0);
  const ReturnType rtype = parse_return_type(cfg.value("return_type", "log"));
  const MissingPolicy missing =
      parse_missing(cfg.value("missing_policy", "skip"));
  const double annual = cfg.value("annualization_factor", 252.0);
  const std::string cov_method = cfg.value("covariance", "sample");
  const double lambda = cfg.value("ewma_lambda", 0.94);
  const std::vector<double> confidences =
      cfg.value("confidences", std::vector<double>{0.95, 0.99});
  const std::vector<int> horizons =
      cfg.value("horizons", std::vector<int>{1, 10});
  int mc_draws = 100000;
  std::uint64_t mc_seed = 42;
  if (cfg.contains("monte_carlo")) {
    mc_draws = cfg["monte_carlo"].value("draws", 100000);
    mc_seed = cfg["monte_carlo"].value("seed", 42);
  }

  std::vector<std::string> assets;
  std::vector<double> weights_v;
  std::map<std::string, std::string> sectors;
  for (const auto& pos : cfg.at("positions")) {
    const std::string a = pos.at("asset");
    assets.push_back(a);
    weights_v.push_back(pos.at("weight").get<double>());
    if (pos.contains("sector")) sectors[a] = pos["sector"];
  }
  Eigen::VectorXd weights = Eigen::Map<Eigen::VectorXd>(
      weights_v.data(), static_cast<Eigen::Index>(weights_v.size()));
  Portfolio portfolio(assets, weights, notional);

  // ---- load return series for each asset -----------------------------------
  std::vector<ReturnSeries> series;
  for (const auto& a : assets) {
    const std::string path = (fs::path(args.data) / (a + ".csv")).string();
    series.push_back(ReturnSeries::from_csv(path, rtype, missing, annual));
    std::cout << "loaded " << path << " (" << series.back().size()
              << " returns)\n";
  }
  Eigen::MatrixXd X = to_return_matrix(series);  // T x N

  // ---- covariance ----------------------------------------------------------
  Eigen::MatrixXd cov;
  if (cov_method == "sample")
    cov = sample_covariance(X);
  else if (cov_method == "ewma")
    cov = ewma_covariance(X, lambda);
  else if (cov_method == "ledoit_wolf")
    cov = ledoit_wolf_covariance(X).cov;
  else
    throw std::invalid_argument("unknown covariance method: " + cov_method);

  if (!is_psd(cov)) {
    std::cerr << "error: estimated covariance is not positive semi-definite\n";
    return 1;
  }

  // ---- stress scenarios ----------------------------------------------------
  std::vector<StressResult> stress;
  if (cfg.contains("stress_scenarios")) {
    auto scenarios = load_scenarios(cfg["stress_scenarios"].get<std::string>());
    FactorBetas betas;
    if (cfg.contains("factor_betas"))
      betas = load_factor_betas(cfg["factor_betas"].get<std::string>());
    stress = apply_scenarios(portfolio, scenarios, betas);
  }

  // ---- build + write report ------------------------------------------------
  RiskReport rep =
      build_report(name, portfolio, X, cov, cov_method, annual, confidences,
                   horizons, stress, sectors, mc_draws, mc_seed);

  fs::create_directories(args.output);
  fs::create_directories(fs::path(args.output) / "figures");
  write_reports(rep, args.output);
  write_all_figures(rep, (fs::path(args.output) / "figures").string());

  // ---- console summary -----------------------------------------------------
  std::cout << "\n=== " << name << " ===\n";
  std::cout << "Covariance: " << cov_method << " | notional: " << notional
            << "\n";
  std::cout << "Portfolio vol: " << rep.daily_vol * 100 << "% daily / "
            << rep.annual_vol * 100 << "% annual\n";
  for (const auto& r : rep.var_cvar) {
    if (r.horizon_days != 1) continue;
    std::cout << "  " << r.method << " " << static_cast<int>(r.confidence * 100)
              << "% 1d VaR=" << r.var * 100 << "%  CVaR=" << r.cvar * 100
              << "%\n";
  }
  std::cout << "Effective # bets: "
            << rep.attribution.concentration.effective_num_bets
            << " | top: " << rep.attribution.concentration.max_contributor
            << " (" << rep.attribution.concentration.max_contribution * 100
            << "%)\n";
  std::cout << "Reports written to " << args.output
            << "report.json, report.md, "
            << "figures/*.svg\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << "\n";
  return 1;
}
