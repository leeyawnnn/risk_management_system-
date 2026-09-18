#include "risk/portfolio.hpp"

#include <cmath>
#include <stdexcept>

namespace risk {

Portfolio::Portfolio(std::vector<std::string> names, Eigen::VectorXd weights,
                     double total_notional)
    : names_(std::move(names)),
      weights_(std::move(weights)),
      notional_(total_notional) {
  if (static_cast<Eigen::Index>(names_.size()) != weights_.size()) {
    throw std::invalid_argument("Portfolio: names and weights size mismatch");
  }
  if (weights_.size() == 0) {
    throw std::invalid_argument("Portfolio: empty portfolio");
  }
  if (!(notional_ > 0.0)) {
    throw std::invalid_argument("Portfolio: notional must be positive");
  }
}

void Portfolio::validate_cov(const Eigen::MatrixXd& cov) const {
  if (cov.rows() != weights_.size() || cov.cols() != weights_.size()) {
    throw std::invalid_argument(
        "Portfolio: covariance dimensions do not match weight vector");
  }
}

Eigen::VectorXd Portfolio::return_series(
    const Eigen::MatrixXd& asset_returns) const {
  if (asset_returns.cols() != weights_.size()) {
    throw std::invalid_argument(
        "Portfolio::return_series: asset_returns column count must equal the "
        "number of assets");
  }
  return asset_returns * weights_;  // (T x N)(N) -> T
}

double Portfolio::variance(const Eigen::MatrixXd& cov) const {
  validate_cov(cov);
  // w^T Sigma w. Clamp tiny negative round-off to zero.
  const double v = weights_.dot(cov * weights_);
  return v < 0.0 ? 0.0 : v;
}

double Portfolio::volatility(const Eigen::MatrixXd& cov) const {
  return std::sqrt(variance(cov));
}

Eigen::VectorXd Portfolio::marginal_contributions(
    const Eigen::MatrixXd& cov) const {
  validate_cov(cov);
  const double vol = volatility(cov);
  const Eigen::VectorXd sigma_w = cov * weights_;
  if (vol <= 0.0) {
    return Eigen::VectorXd::Zero(weights_.size());
  }
  return sigma_w / vol;
}

Eigen::VectorXd Portfolio::component_contributions(
    const Eigen::MatrixXd& cov) const {
  // CCR_i = w_i * MCR_i. Element-wise product with the weights.
  return weights_.cwiseProduct(marginal_contributions(cov));
}

Eigen::VectorXd Portfolio::percent_contributions(
    const Eigen::MatrixXd& cov) const {
  const double vol = volatility(cov);
  const Eigen::VectorXd ccr = component_contributions(cov);
  if (vol <= 0.0) {
    return Eigen::VectorXd::Zero(weights_.size());
  }
  return ccr / vol;
}

}  // namespace risk
