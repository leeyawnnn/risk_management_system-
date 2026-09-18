#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "risk/attribution.hpp"
#include "risk/backtest.hpp"
#include "risk/decomposition.hpp"

namespace risk {

// SVG figure generators. Every one routes through risk::plot (plot_style.hpp)
// for palette, type scale, axes and the provenance footer, and every one takes
// a `source` string that is stamped into the figure. A figure that cannot say
// where its numbers came from does not get written.

// Correlation heatmap with rows and columns reordered by hierarchical
// clustering, so asset-class blocks emerge from the data instead of being
// asserted in the caption. Upper triangle masked, diverging scale pinned at
// zero, cell values annotated.
std::string svg_correlation_heatmap(const Eigen::MatrixXd& correlation,
                                    const std::vector<std::string>& names,
                                    const std::string& source);

// Average-linkage hierarchical clustering on correlation distance
// d(i,j) = 1 - rho(i,j), returning the leaf order. Exposed for testing.
std::vector<std::size_t> correlation_cluster_order(
    const Eigen::MatrixXd& correlation);

// Dumbbell chart of capital weight against risk share, sorted by the gap so
// the positions punching above their weight rise to the top automatically.
std::string svg_weight_vs_risk(const std::vector<std::string>& names,
                               const std::vector<double>& weights,
                               const std::vector<double>& pct_risk,
                               const std::string& source);

// Risk contribution per position, with a reference line at 1/N and the
// effective number of bets annotated on the chart.
std::string svg_risk_contribution_bars(const RiskAttribution& attribution,
                                       const std::string& source);

// Distribution of daily portfolio returns with VaR and CVaR marked and the
// fitted Gaussian density overlaid, so a reader can see for themselves
// whether the empirical distribution departs from it.
std::string svg_return_histogram(const Eigen::VectorXd& portfolio_returns,
                                 double var, double cvar, double confidence,
                                 const std::string& source, int bins = 60);

// Annualized volatility per instrument, sorted, with bootstrap confidence
// intervals. A volatility estimated on ~1,240 days has real uncertainty and
// almost nobody draws it.
std::string svg_asset_volatility(const std::vector<std::string>& names,
                                 const std::vector<double>& annual_vol,
                                 const std::vector<double>& ci_lower,
                                 const std::vector<double>& ci_upper,
                                 const std::vector<std::string>& sectors,
                                 const std::string& source);

// One row of the estimator ground-truth study.
struct EstimatorErrorPoint {
  std::string estimator;
  int sample_size = 0;
  double error = 0.0;     // mean |estimate - truth|
  double error_se = 0.0;  // Monte Carlo standard error of that mean
};

// Estimator error against sample size on log-log axes, one line per
// estimator, with Monte Carlo confidence bands.
std::string svg_estimator_error(const std::vector<EstimatorErrorPoint>& points,
                                const std::string& y_label,
                                const std::string& title,
                                const std::string& source);

// One estimator's mean estimated eigenvalue against the truth, at one
// sample size.
struct SpectrumPoint {
  std::string estimator;
  int index = 0;  // 0 = smallest
  double estimated = 0.0;
  double truth = 0.0;
};

// Estimated eigenvalue spectrum against the true one, log y. This is the
// picture behind "shrinkage improves conditioning": the sample estimator
// pushes the smallest eigenvalues down and the largest up, and shrinkage
// pulls both back toward the truth.
std::string svg_eigenvalue_spectrum(const std::vector<SpectrumPoint>& points,
                                    int sample_size, const std::string& source);

// Daily returns against the VaR line with exceptions marked, the Basel zone
// as a background band over the trailing supervisory window, and the
// Kupiec/Christoffersen results in an annotation box.
std::string svg_var_backtest(const Eigen::VectorXd& portfolio_returns,
                             double var_level, double confidence,
                             const KupiecResult& kupiec,
                             const ChristoffersenResult& christoffersen,
                             const BaselResult& basel,
                             const std::string& source);

// Stacked decomposition of portfolio variance into each factor's
// contribution plus specific risk.
std::string svg_factor_decomposition(const FactorDecomposition& decomposition,
                                     const std::string& source);

}  // namespace risk
