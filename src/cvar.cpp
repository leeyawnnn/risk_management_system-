#include "risk/cvar.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "risk/var.hpp"

namespace risk {

double tail_mean(std::vector<double>& data, double tail_prob) {
  if (data.empty()) {
    throw std::invalid_argument("tail_mean: empty data");
  }
  tail_prob = std::clamp(tail_prob, 0.0, 1.0);
  const std::size_t n = data.size();
  // k = number of observations in the tail; always at least 1 (the worst).
  std::size_t k =
      static_cast<std::size_t>(std::floor(tail_prob * static_cast<double>(n)));
  if (k < 1) k = 1;
  if (k > n) k = n;

  // Partition so the k smallest values occupy positions [0, k).
  std::nth_element(data.begin(),
                   data.begin() + static_cast<std::ptrdiff_t>(k - 1),
                   data.end());
  double sum = 0.0;
  for (std::size_t i = 0; i < k; ++i) sum += data[i];
  return sum / static_cast<double>(k);
}

double historical_cvar(const Eigen::VectorXd& portfolio_returns,
                       double confidence, int horizon_days) {
  if (portfolio_returns.size() < 1) {
    throw std::invalid_argument("historical_cvar: empty return series");
  }
  if (!(confidence > 0.0 && confidence < 1.0)) {
    throw std::invalid_argument(
        "historical_cvar: confidence must be in (0, 1)");
  }
  if (horizon_days < 1) {
    throw std::invalid_argument("historical_cvar: horizon_days must be >= 1");
  }
  std::vector<double> r(portfolio_returns.data(),
                        portfolio_returns.data() + portfolio_returns.size());
  const double tm = tail_mean(r, 1.0 - confidence);  // negative (a loss)
  return -tm * std::sqrt(static_cast<double>(horizon_days));
}

double parametric_cvar(double mean, double stdev, double confidence,
                       int horizon_days) {
  if (stdev < 0.0) {
    throw std::invalid_argument("parametric_cvar: stdev must be non-negative");
  }
  if (!(confidence > 0.0 && confidence < 1.0)) {
    throw std::invalid_argument(
        "parametric_cvar: confidence must be in (0, 1)");
  }
  if (horizon_days < 1) {
    throw std::invalid_argument("parametric_cvar: horizon_days must be >= 1");
  }
  const double alpha_tail = 1.0 - confidence;
  const double z = normal_ppf(confidence);              // 1.645 at 95%
  const double es_factor = normal_pdf(z) / alpha_tail;  // phi(z)/(1-conf)
  const double h = static_cast<double>(horizon_days);
  return -mean * h + stdev * std::sqrt(h) * es_factor;
}

double monte_carlo_cvar(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov,
                        const Eigen::VectorXd& weights, double confidence,
                        int horizon_days, int draws, std::uint64_t seed) {
  if (!(confidence > 0.0 && confidence < 1.0)) {
    throw std::invalid_argument(
        "monte_carlo_cvar: confidence must be in (0, 1)");
  }
  Eigen::VectorXd sims =
      simulate_portfolio_returns(mean, cov, weights, horizon_days, draws, seed);
  std::vector<double> v(sims.data(), sims.data() + sims.size());
  const double tm = tail_mean(v, 1.0 - confidence);
  return -tm;  // path already covers the horizon
}

}  // namespace risk
