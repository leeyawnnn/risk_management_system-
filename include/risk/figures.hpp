#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "risk/attribution.hpp"

namespace risk {

// Self-contained SVG figure generators (no external plotting dependency).
// Each returns a complete <svg>...</svg> document as a string.

// Correlation heatmap: blue (negative) -> white (0) -> red (positive), with the
// numeric value printed in each cell and asset labels on the axes.
std::string svg_correlation_heatmap(const Eigen::MatrixXd& correlation,
                                    const std::vector<std::string>& names);

// Horizontal bar chart of percentage risk contributions per asset. Bars to the
// right (positive risk) are red; left (risk-reducing / hedge) are green.
std::string svg_risk_contribution_bars(const RiskAttribution& attribution);

// Histogram of portfolio returns with vertical markers at the return levels
// implied by VaR and CVaR (i.e. at -var and -cvar). confidence labels the lines.
std::string svg_return_histogram(const Eigen::VectorXd& portfolio_returns,
                                 double var, double cvar, double confidence,
                                 int bins = 40);

// Horizontal bar chart of annualized volatility per instrument, coloured by
// sector (a simple legend is drawn). Bars are sorted from most to least volatile.
std::string svg_asset_volatility(const std::vector<std::string>& names,
                                 const std::vector<double>& annual_vol,
                                 const std::vector<std::string>& sectors);

// Per-asset comparison of capital weight (grey) vs. risk share (red). Where the
// red bar exceeds the grey one, the position contributes more risk than weight.
std::string svg_weight_vs_risk(const std::vector<std::string>& names,
                               const std::vector<double>& weights,
                               const std::vector<double>& pct_risk);

// Grouped bar chart of 95% and 99% 1-day VaR under each covariance estimator.
std::string svg_estimator_var_comparison(
    const std::vector<std::string>& methods,
    const std::vector<double>& var95, const std::vector<double>& var99);

// Time series of daily portfolio returns against the -VaR line, with breaches
// (returns below -var_level) highlighted in red.
std::string svg_var_backtest(const Eigen::VectorXd& portfolio_returns,
                             double var_level, double confidence);

// Write the original three figures into `dir` as correlation.svg,
// risk_contributions.svg, and return_distribution.svg. `dir` must exist.
void write_figures(const Eigen::MatrixXd& correlation,
                   const std::vector<std::string>& names,
                   const RiskAttribution& attribution,
                   const Eigen::VectorXd& portfolio_returns, double var,
                   double cvar, double confidence, const std::string& dir);

}  // namespace risk
