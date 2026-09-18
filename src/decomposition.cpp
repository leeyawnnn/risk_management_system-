#include "risk/decomposition.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include "risk/var.hpp"

namespace risk {

VarDecomposition decompose_var(
    const Portfolio& portfolio, const Eigen::MatrixXd& cov,
    const Eigen::VectorXd& mean, double confidence, int horizon_days,
    const std::map<std::string, std::string>& sectors) {
  if (!(confidence > 0.0 && confidence < 1.0)) {
    throw std::invalid_argument("decompose_var: confidence must be in (0, 1)");
  }
  if (horizon_days < 1) {
    throw std::invalid_argument("decompose_var: horizon_days must be >= 1");
  }
  if (mean.size() != portfolio.size()) {
    throw std::invalid_argument("decompose_var: mean/weights size mismatch");
  }

  const Eigen::VectorXd& w = portfolio.weights();
  const double z = normal_ppf(confidence);
  const auto h = static_cast<double>(horizon_days);
  const double sqrt_h = std::sqrt(h);

  const double sigma = portfolio.volatility(cov);  // 1-day
  const double mu = w.dot(mean);                   // 1-day

  VarDecomposition d;
  d.names = portfolio.names();
  d.weights = w;
  d.confidence = confidence;
  d.horizon_days = horizon_days;
  d.var = z * sigma * sqrt_h - mu * h;

  // dVaR/dw_i = z * sqrt(h) * (S w)_i / sigma  -  h * mu_i.
  if (sigma > 0.0) {
    d.marginal = (z * sqrt_h / sigma) * (cov * w) - h * mean;
  } else {
    d.marginal = -h * mean;
  }
  d.component = w.cwiseProduct(d.marginal);
  d.percent = (std::abs(d.var) > 0.0) ? Eigen::VectorXd(d.component / d.var)
                                      : Eigen::VectorXd::Zero(w.size());

  // Euler's theorem makes this exact for a degree-1 homogeneous function; any
  // departure beyond round-off means the gradient does not match the level.
  d.euler_residual = d.component.sum() - d.var;

  if (!sectors.empty()) {
    for (Eigen::Index i = 0; i < w.size(); ++i) {
      const std::string& name = d.names[static_cast<std::size_t>(i)];
      auto it = sectors.find(name);
      const std::string sector =
          (it != sectors.end()) ? it->second : std::string("Unclassified");
      d.sector_component[sector] += d.component(i);
    }
  }
  return d;
}

IncrementalVar incremental_var(const Portfolio& portfolio,
                               const Eigen::MatrixXd& cov,
                               const Eigen::VectorXd& mean,
                               const std::string& asset, double trade_notional,
                               double confidence, int horizon_days) {
  const auto& names = portfolio.names();
  const auto it = std::ranges::find(names, asset);
  if (it == names.end()) {
    throw std::invalid_argument("incremental_var: unknown asset " + asset);
  }
  const auto idx = static_cast<Eigen::Index>(std::distance(names.begin(), it));

  const double notional = portfolio.notional();
  const double new_notional = notional + trade_notional;
  if (!(new_notional > 0.0)) {
    throw std::invalid_argument(
        "incremental_var: trade would drive total notional non-positive");
  }

  const Eigen::VectorXd& w = portfolio.weights();
  const double z = normal_ppf(confidence);
  const auto h = static_cast<double>(horizon_days);
  const double sqrt_h = std::sqrt(h);

  auto var_dollars = [&](const Eigen::VectorXd& weights, double size) {
    const double sigma = std::sqrt(std::max(0.0, weights.dot(cov * weights)));
    const double mu = weights.dot(mean);
    return (z * sigma * sqrt_h - mu * h) * size;
  };

  // Rebuild the weight vector after the trade. Dollar positions are held
  // fixed and one is changed; weights are then re-derived against the new
  // total, which is what actually happens on the book.
  Eigen::VectorXd dollars = w * notional;
  dollars(idx) += trade_notional;
  const Eigen::VectorXd w_after = dollars / new_notional;

  IncrementalVar out;
  out.asset = asset;
  out.trade_notional = trade_notional;
  out.var_before = var_dollars(w, notional);
  out.var_after = var_dollars(w_after, new_notional);
  out.incremental = out.var_after - out.var_before;

  // The linearisation a desk would do in its head, for comparison.
  const VarDecomposition d =
      decompose_var(portfolio, cov, mean, confidence, horizon_days);
  out.linear_estimate = d.marginal(idx) * trade_notional;
  return out;
}

FactorDecomposition decompose_factor_risk(
    const Portfolio& portfolio,
    const std::map<std::string, std::map<std::string, double>>& betas,
    const std::vector<std::string>& factors, const Eigen::MatrixXd& factor_cov,
    const Eigen::VectorXd& specific_var, const Eigen::MatrixXd& cov) {
  const auto N = portfolio.size();
  const auto K = static_cast<Eigen::Index>(factors.size());
  if (factor_cov.rows() != K || factor_cov.cols() != K) {
    throw std::invalid_argument(
        "decompose_factor_risk: factor_cov must be K x K");
  }
  if (specific_var.size() != N) {
    throw std::invalid_argument(
        "decompose_factor_risk: specific_var must have one entry per asset");
  }

  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N, K);
  const auto& names = portfolio.names();
  for (Eigen::Index i = 0; i < N; ++i) {
    auto ait = betas.find(names[static_cast<std::size_t>(i)]);
    if (ait == betas.end()) continue;
    for (Eigen::Index k = 0; k < K; ++k) {
      auto fit = ait->second.find(factors[static_cast<std::size_t>(k)]);
      if (fit != ait->second.end()) B(i, k) = fit->second;
    }
  }

  const Eigen::VectorXd& w = portfolio.weights();
  FactorDecomposition d;
  d.factors = factors;
  d.exposures = B.transpose() * w;  // b = B'w

  const Eigen::VectorXd sigma_b = factor_cov * d.exposures;
  d.variance_contribution = d.exposures.cwiseProduct(sigma_b);
  d.factor_variance = d.variance_contribution.sum();
  d.specific_variance = w.cwiseProduct(w).dot(specific_var);
  d.model_variance = d.factor_variance + d.specific_variance;
  d.model_vol = std::sqrt(std::max(0.0, d.model_variance));

  d.percent_of_total =
      (d.model_variance > 0.0)
          ? Eigen::VectorXd(d.variance_contribution / d.model_variance)
          : Eigen::VectorXd::Zero(K);

  d.sample_variance = portfolio.variance(cov);
  d.model_error = d.model_variance - d.sample_variance;
  d.variance_explained =
      (d.model_variance > 0.0) ? d.factor_variance / d.model_variance : 0.0;
  return d;
}

FactorSeries load_factor_series(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::invalid_argument("load_factor_series: cannot open " + path);
  }

  std::string line;
  if (!std::getline(in, line)) {
    throw std::invalid_argument("load_factor_series: empty file " + path);
  }

  FactorSeries fs;
  {
    std::stringstream hs(line);
    std::string field;
    bool first = true;
    while (std::getline(hs, field, ',')) {
      if (first) {  // the date column
        first = false;
        continue;
      }
      fs.names.push_back(field);
    }
  }
  if (fs.names.empty()) {
    throw std::invalid_argument("load_factor_series: no factor columns in " +
                                path);
  }

  std::vector<std::vector<double>> rows;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::stringstream ls(line);
    std::string field;
    std::vector<double> row;
    bool first = true;
    while (std::getline(ls, field, ',')) {
      if (first) {
        fs.dates.push_back(field);
        first = false;
        continue;
      }
      row.push_back(std::stod(field));
    }
    if (row.size() != fs.names.size()) {
      throw std::invalid_argument("load_factor_series: ragged row in " + path);
    }
    rows.push_back(std::move(row));
  }
  if (rows.empty()) {
    throw std::invalid_argument("load_factor_series: no observations in " +
                                path);
  }

  fs.values.resize(static_cast<Eigen::Index>(rows.size()),
                   static_cast<Eigen::Index>(fs.names.size()));
  for (std::size_t t = 0; t < rows.size(); ++t) {
    for (std::size_t k = 0; k < fs.names.size(); ++k) {
      fs.values(static_cast<Eigen::Index>(t), static_cast<Eigen::Index>(k)) =
          rows[t][k];
    }
  }
  return fs;
}

}  // namespace risk
