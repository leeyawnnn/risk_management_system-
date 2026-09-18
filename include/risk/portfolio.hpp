#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace risk {

// A portfolio: asset names, weights, and a total notional (dollar size).
//
// Weights are fractions of capital. They are NOT forced to sum to 1 (a
// long/short or leveraged book may not), but the risk decomposition below
// assumes w expresses the position sizes you want risk attributed to. All
// covariance inputs Sigma are taken to be *daily*; annualization/horizon
// scaling happens in the VaR layer, never silently here.
//
// Risk decomposition (Sigma = covariance, sigma_p = portfolio volatility):
//   variance              v   = w^T Sigma w
//   volatility            sigma_p = sqrt(v)
//   marginal contribution MCR_i = (Sigma w)_i / sigma_p     (= d sigma_p / d
//   w_i) component contribution CCR_i = w_i * MCR_i              (sums to
//   sigma_p) percent contribution  CCR_i / sigma_p                   (sums to
//   1)
class Portfolio {
 public:
  Portfolio(std::vector<std::string> names, Eigen::VectorXd weights,
            double total_notional = 1.0);

  // ---- accessors -----------------------------------------------------------
  [[nodiscard]] const std::vector<std::string>& names() const { return names_; }
  [[nodiscard]] const Eigen::VectorXd& weights() const { return weights_; }
  [[nodiscard]] double notional() const { return notional_; }
  [[nodiscard]] Eigen::Index size() const { return weights_.size(); }

  // ---- portfolio aggregation ----------------------------------------------
  // Weighted return series from a (T x N) asset return matrix: returns X * w,
  // a length-T vector of portfolio returns. Throws on column-count mismatch.
  [[nodiscard]] Eigen::VectorXd return_series(
      const Eigen::MatrixXd& asset_returns) const;

  // ---- risk -----------------------------------------------------------------
  [[nodiscard]] double variance(
      const Eigen::MatrixXd& cov) const;  // w^T Sigma w
  [[nodiscard]] double volatility(
      const Eigen::MatrixXd& cov) const;  // sqrt(variance)

  // Marginal contribution to risk per asset: (Sigma w) / sigma_p.
  [[nodiscard]] Eigen::VectorXd marginal_contributions(
      const Eigen::MatrixXd& cov) const;

  // Component contribution to risk per asset: w_i * MCR_i. Sums to
  // volatility().
  [[nodiscard]] Eigen::VectorXd component_contributions(
      const Eigen::MatrixXd& cov) const;

  // Percentage contribution per asset: CCR_i / sigma_p. Sums to 1.
  [[nodiscard]] Eigen::VectorXd percent_contributions(
      const Eigen::MatrixXd& cov) const;

 private:
  void validate_cov(const Eigen::MatrixXd& cov) const;

  std::vector<std::string> names_;
  Eigen::VectorXd weights_;
  double notional_;
};

}  // namespace risk
