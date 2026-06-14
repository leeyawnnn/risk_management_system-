#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "risk/attribution.hpp"
#include "risk/portfolio.hpp"
#include "risk/stress_test.hpp"

namespace risk {

// One row of the VaR/CVaR table: method x confidence x horizon.
struct VarCvarRow {
  std::string method;   // "Historical" | "Parametric" | "MonteCarlo"
  double confidence = 0.95;
  int horizon_days = 1;
  double var = 0.0;     // positive loss fraction
  double cvar = 0.0;    // positive loss fraction
};

// Per-asset descriptive statistics (for the comparison table / figures).
struct AssetStat {
  std::string name;
  std::string sector;
  double weight = 0.0;
  double annual_vol = 0.0;       // stdev * sqrt(annualization)
  double annual_return = 0.0;    // mean * annualization
  double sharpe = 0.0;           // annual_return / annual_vol (rf = 0)
  double corr_to_portfolio = 0.0;
  double pct_risk = 0.0;         // share of portfolio risk
};

// Portfolio risk under each covariance estimator, for side-by-side comparison.
struct EstimatorStat {
  std::string method;       // "Sample" | "EWMA" | "Ledoit-Wolf"
  double daily_vol = 0.0;
  double annual_vol = 0.0;
  double var95 = 0.0;       // parametric 1-day 95% VaR under this estimator
  double var99 = 0.0;
};

// In-sample VaR backtest: how often realized returns breached the VaR level.
struct BacktestStat {
  double confidence = 0.95;
  long observations = 0;
  double expected_rate = 0.0;   // 1 - confidence
  long exceedances = 0;         // days with return < -VaR
  double actual_rate = 0.0;     // exceedances / observations
  double var_level = 0.0;       // the 1-day VaR used
};

// Everything needed to render the machine- and human-readable reports.
struct RiskReport {
  std::string portfolio_name;
  double notional = 0.0;
  std::vector<std::string> assets;
  Eigen::VectorXd weights;

  std::string covariance_method;     // e.g. "sample" | "ewma" | "ledoit_wolf"
  double annualization_factor = 252.0;
  Eigen::MatrixXd covariance;        // daily
  Eigen::MatrixXd correlation;
  double daily_vol = 0.0;
  double annual_vol = 0.0;

  Eigen::VectorXd portfolio_returns;  // historical, for the distribution figure
  std::vector<VarCvarRow> var_cvar;
  RiskAttribution attribution;
  std::vector<StressResult> stress;

  // Richer analytics for the detailed report.
  std::vector<AssetStat> asset_stats;
  std::vector<EstimatorStat> estimator_comparison;
  std::vector<BacktestStat> backtests;
};

// Assemble a full report: computes the VaR/CVaR table (all three methods over
// the requested confidences and horizons), volatility, correlation, and risk
// attribution. `asset_returns` is the (T x N) historical matrix; `cov` the
// chosen daily covariance estimate; `sectors` optional asset->sector metadata.
RiskReport build_report(const std::string& name, const Portfolio& portfolio,
                        const Eigen::MatrixXd& asset_returns,
                        const Eigen::MatrixXd& cov,
                        const std::string& cov_method,
                        double annualization_factor,
                        const std::vector<double>& confidences,
                        const std::vector<int>& horizons,
                        const std::vector<StressResult>& stress,
                        const std::map<std::string, std::string>& sectors = {},
                        int mc_draws = 100000, std::uint64_t seed = 42,
                        double ewma_lambda = 0.94);

// Serialize the report.
std::string to_json(const RiskReport& report, int indent = 2);
std::string to_markdown(const RiskReport& report);

// Write report.json and report.md into `dir` (which must exist).
void write_reports(const RiskReport& report, const std::string& dir);

// Write all seven SVG figures into `dir` (which must exist), deriving the
// distribution figure's VaR/CVaR markers from the first requested confidence.
void write_all_figures(const RiskReport& report, const std::string& dir);

}  // namespace risk
