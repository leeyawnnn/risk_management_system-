#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "risk/attribution.hpp"
#include "risk/backtest.hpp"
#include "risk/covariance.hpp"
#include "risk/decomposition.hpp"
#include "risk/portfolio.hpp"
#include "risk/stress_test.hpp"

namespace risk {

// One row of the VaR/CVaR table: method x confidence x horizon.
struct VarCvarRow {
  std::string method;  // "Historical" | "Parametric" | "MonteCarlo"
  double confidence = 0.95;
  int horizon_days = 1;
  double var = 0.0;   // positive loss fraction
  double cvar = 0.0;  // positive loss fraction
};

// Per-asset descriptive statistics, with bootstrap uncertainty on the
// volatility estimate. A vol computed on ~1,240 days is an estimate, and the
// report says so rather than printing it as a constant.
struct AssetStat {
  std::string name;
  std::string sector;
  double weight = 0.0;
  double annual_vol = 0.0;
  double annual_vol_ci_lower = 0.0;
  double annual_vol_ci_upper = 0.0;
  double annual_return = 0.0;
  double sharpe = 0.0;  // annual_return / annual_vol, rf = 0
  double skewness = 0.0;
  double excess_kurtosis = 0.0;
  double corr_to_portfolio = 0.0;
  double pct_risk = 0.0;
};

// Portfolio risk under each covariance estimator, with conditioning.
struct EstimatorStat {
  std::string method;  // "Sample" | "EWMA" | "Ledoit-Wolf"
  double daily_vol = 0.0;
  double annual_vol = 0.0;
  double var95 = 0.0;
  double var99 = 0.0;
  double condition_number = 0.0;
  double correlation_condition_number = 0.0;
  double shrinkage = 0.0;  // non-zero only for Ledoit-Wolf
};

// Everything needed to render the machine- and human-readable reports.
struct RiskReport {
  std::string portfolio_name;
  std::string as_of;        // data as-of date, stamped into every artifact
  std::string git_commit;   // the code that produced this
  std::uint64_t seed = 42;  // so a reader can reproduce the simulations
  double notional = 0.0;
  std::vector<std::string> assets;
  Eigen::VectorXd weights;

  std::string covariance_method;
  double annualization_factor = 252.0;
  Eigen::MatrixXd covariance;  // daily
  Eigen::MatrixXd correlation;
  MatrixDiagnostics covariance_diagnostics;
  MatrixDiagnostics correlation_diagnostics;
  double daily_vol = 0.0;
  double annual_vol = 0.0;

  // Sample moments of the portfolio return series. Excess kurtosis is here
  // because whether the tail is fat is a measurement, not an assumption.
  double portfolio_skewness = 0.0;
  double portfolio_excess_kurtosis = 0.0;

  Eigen::VectorXd portfolio_returns;
  std::vector<VarCvarRow> var_cvar;
  RiskAttribution attribution;
  std::vector<VarDecomposition> var_decomposition;  // one per confidence
  std::vector<IncrementalVar> incremental;
  FactorDecomposition factor_decomposition;
  bool has_factor_decomposition = false;

  std::vector<StressResult> stress;
  std::vector<ReverseStressResult> reverse_stress;

  std::vector<AssetStat> asset_stats;
  std::vector<EstimatorStat> estimator_comparison;
  std::vector<BacktestRow> backtests;
};

// Inputs that are optional: pass empty/absent to skip the corresponding
// section rather than silently emitting a section built from defaults.
struct ReportInputs {
  const Eigen::MatrixXd* asset_returns = nullptr;
  const Eigen::MatrixXd* cov = nullptr;
  std::string cov_method;
  double annualization_factor = 252.0;
  std::vector<double> confidences{0.95, 0.99};
  std::vector<int> horizons{1};
  std::vector<StressResult> stress;
  std::map<std::string, std::string> sectors;
  int mc_draws = 100000;
  std::uint64_t seed = 42;
  double ewma_lambda = 0.94;
  std::string as_of;
  std::string git_commit;
  int bootstrap_resamples = 2000;

  // Factor inputs. All three must be present for the factor sections.
  const FactorSeries* factors = nullptr;
  const FactorBetas* betas = nullptr;
  std::vector<double> reverse_stress_targets;

  // Hypothetical trades to price with incremental VaR, as (asset, notional).
  std::vector<std::pair<std::string, double>> trades;
};

RiskReport build_report(const std::string& name, const Portfolio& portfolio,
                        const ReportInputs& in);

std::string to_json(const RiskReport& report, int indent = 2);
std::string to_markdown(const RiskReport& report);

// Write report.json and report.md into `dir`, which must exist.
void write_reports(const RiskReport& report, const std::string& dir);

// Write every SVG figure into `dir`, which must exist. `estimator_csv` is the
// path to the output of the estimator study; when it is missing or
// unreadable that one figure is skipped and the rest are still written.
void write_all_figures(const RiskReport& report, const std::string& dir,
                       const std::string& estimator_csv = "");

}  // namespace risk
