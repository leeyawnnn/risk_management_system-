#pragma once

#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

#include "risk/portfolio.hpp"

namespace risk {

// ===========================================================================
// VaR decomposition
// ===========================================================================
// Volatility decomposition (in attribution.hpp) answers "where does my risk
// sit". These answer the questions a desk actually asks during the day:
// what does one more dollar of X cost me, and what would this trade do?
//
// Under the parametric Gaussian model
//
//     VaR(w) = z * sqrt(w' S w) - h * (w . mu)
//
// VaR is homogeneous of degree 1 in w, so Euler's theorem gives an exact
// additive split: sum_i w_i * dVaR/dw_i = VaR. The residual is reported
// rather than assumed, because it is the cheapest possible check that the
// gradient and the level agree.

struct VarDecomposition {
  std::vector<std::string> names;
  Eigen::VectorXd weights;

  double var = 0.0;  // total parametric VaR, positive loss fraction
  double confidence = 0.95;
  int horizon_days = 1;

  Eigen::VectorXd marginal;   // dVaR/dw_i
  Eigen::VectorXd component;  // w_i * marginal_i, sums to var
  Eigen::VectorXd percent;    // component_i / var, sums to 1

  // Euler check: sum(component) - var. Should be at machine precision.
  double euler_residual = 0.0;

  std::map<std::string, double> sector_component;
};

// `cov` and `mean` are 1-day asset moments; horizon scaling matches the VaR
// layer (mu linear in time, sigma in sqrt-time).
VarDecomposition decompose_var(
    const Portfolio& portfolio, const Eigen::MatrixXd& cov,
    const Eigen::VectorXd& mean, double confidence, int horizon_days = 1,
    const std::map<std::string, std::string>& sectors = {});

// ---------------------------------------------------------------------------
// Incremental VaR
// ---------------------------------------------------------------------------
// "What does adding $1mm of X do to portfolio VaR?" Computed by full
// revaluation -- rebuild the weight vector and recompute -- not by the
// marginal-VaR linearisation. For a trade of any size worth asking about, the
// two differ, and the linear answer always understates a risk-reducing trade
// and overstates a risk-adding one because VaR is convex in w.
struct IncrementalVar {
  std::string asset;
  double trade_notional = 0.0;   // signed dollars
  double var_before = 0.0;       // dollars
  double var_after = 0.0;        // dollars
  double incremental = 0.0;      // var_after - var_before, dollars
  double linear_estimate = 0.0;  // marginal VaR * trade, dollars
};

// `trade_notional` is signed and expressed in the same currency as the
// portfolio notional. The book's total notional grows by the trade, which is
// what actually happens when a desk adds a position.
IncrementalVar incremental_var(const Portfolio& portfolio,
                               const Eigen::MatrixXd& cov,
                               const Eigen::VectorXd& mean,
                               const std::string& asset, double trade_notional,
                               double confidence, int horizon_days = 1);

// ===========================================================================
// Factor risk decomposition
// ===========================================================================
// Splits portfolio variance into what the factor model explains and what it
// does not:
//
//     S ~= B Sigma_F B' + D      (D diagonal, the specific variances)
//     var_p = w'B Sigma_F B'w  +  w'Dw
//           = factor variance   +  specific variance
//
// The two must sum to the model's total. They will NOT sum to the variance
// implied by the sample covariance, because a six-factor model does not
// reproduce an 11x11 covariance matrix exactly. That gap is reported as
// `model_error` rather than hidden: it is the honest measure of how much of
// the book the factor model actually captures.
struct FactorDecomposition {
  std::vector<std::string> factors;

  Eigen::VectorXd exposures;  // b = B'w, portfolio beta to each factor
  // Contribution of each factor to portfolio VARIANCE. Sums to
  // factor_variance. Cross-factor covariance is allocated by the standard
  // b_k * (Sigma_F b)_k split, so a single entry can be negative when a
  // factor hedges the rest of the book.
  Eigen::VectorXd variance_contribution;
  Eigen::VectorXd percent_of_total;  // contribution / model_variance

  double factor_variance = 0.0;
  double specific_variance = 0.0;
  double model_variance = 0.0;  // factor + specific
  double model_vol = 0.0;       // sqrt(model_variance)

  double sample_variance = 0.0;     // w' S w under the covariance actually used
  double model_error = 0.0;         // model_variance - sample_variance
  double variance_explained = 0.0;  // factor_variance / model_variance
};

// `betas[asset][factor]`; assets missing from the map get zero loadings, and
// `specific_var[i]` is asset i's daily residual variance.
FactorDecomposition decompose_factor_risk(
    const Portfolio& portfolio,
    const std::map<std::string, std::map<std::string, double>>& betas,
    const std::vector<std::string>& factors, const Eigen::MatrixXd& factor_cov,
    const Eigen::VectorXd& specific_var, const Eigen::MatrixXd& cov);

// Load the daily factor realisations written by scripts/fetch_data.py.
// Returns the factor names and a (T x K) matrix in column order.
struct FactorSeries {
  std::vector<std::string> names;
  std::vector<std::string> dates;
  Eigen::MatrixXd values;  // T x K
};

FactorSeries load_factor_series(const std::string& path);

}  // namespace risk
