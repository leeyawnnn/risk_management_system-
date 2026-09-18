#include "risk/var.hpp"

#include "risk/random.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace risk {

double normal_pdf(double x) {
  // phi(x) = exp(-x^2/2) / sqrt(2 pi).
  return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

double normal_cdf(double x) {
  // Phi(x) = 0.5 * erfc(-x / sqrt(2)).
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double normal_ppf(double p) {
  if (!(p > 0.0 && p < 1.0)) {
    throw std::invalid_argument("normal_ppf: p must be in (0, 1)");
  }
  // Peter Acklam's rational approximation, then one Halley refinement step
  // using erfc for near machine-precision accuracy.
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                             -2.759285104469687e+02, 1.383577518672690e+02,
                             -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                             -1.556989798598866e+02, 6.680131188771972e+01,
                             -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                             -2.400758277161838e+00, -2.549732539343734e+00,
                             4.374664141464968e+00,  2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                             2.445134137142996e+00, 3.754408661907416e+00};
  const double plow = 0.02425;
  const double phigh = 1.0 - plow;

  double x;
  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  } else if (p <= phigh) {
    const double q = p - 0.5;
    const double r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  } else {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }

  // Halley step: refine using e = Phi(x) - p, u = e * sqrt(2 pi) * exp(x^2/2).
  const double e = normal_cdf(x) - p;
  const double u = e * std::sqrt(2.0 * M_PI) * std::exp(x * x / 2.0);
  x = x - u / (1.0 + x * u / 2.0);
  return x;
}

double empirical_quantile(std::vector<double>& data, double q) {
  if (data.empty()) {
    throw std::invalid_argument("empirical_quantile: empty data");
  }
  q = std::clamp(q, 0.0, 1.0);
  const std::size_t n = data.size();
  std::size_t k =
      static_cast<std::size_t>(std::floor(q * static_cast<double>(n)));
  if (k >= n) k = n - 1;
  // Partition so that data[k] holds the k-th smallest element. O(n) average.
  std::nth_element(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(k),
                   data.end());
  return data[k];
}

double historical_var(const Eigen::VectorXd& portfolio_returns,
                      double confidence, int horizon_days) {
  if (portfolio_returns.size() < 1) {
    throw std::invalid_argument("historical_var: empty return series");
  }
  if (!(confidence > 0.0 && confidence < 1.0)) {
    throw std::invalid_argument("historical_var: confidence must be in (0, 1)");
  }
  if (horizon_days < 1) {
    throw std::invalid_argument("historical_var: horizon_days must be >= 1");
  }
  std::vector<double> r(portfolio_returns.data(),
                        portfolio_returns.data() + portfolio_returns.size());
  // Lower-tail quantile at (1 - confidence); typically a negative return.
  const double q = empirical_quantile(r, 1.0 - confidence);
  const double var_1d = -q;  // loss is positive
  return var_1d * std::sqrt(static_cast<double>(horizon_days));
}

double parametric_var(double mean, double stdev, double confidence,
                      int horizon_days) {
  if (stdev < 0.0) {
    throw std::invalid_argument("parametric_var: stdev must be non-negative");
  }
  if (!(confidence > 0.0 && confidence < 1.0)) {
    throw std::invalid_argument("parametric_var: confidence must be in (0, 1)");
  }
  if (horizon_days < 1) {
    throw std::invalid_argument("parametric_var: horizon_days must be >= 1");
  }
  const double z = normal_ppf(confidence);
  const double h = static_cast<double>(horizon_days);
  // mu scales linearly with time, sigma with sqrt(time).
  const double mu_h = mean * h;
  const double sigma_h = stdev * std::sqrt(h);
  return z * sigma_h - mu_h;  // positive loss
}

Eigen::VectorXd simulate_portfolio_returns(const Eigen::VectorXd& mean,
                                           const Eigen::MatrixXd& cov,
                                           const Eigen::VectorXd& weights,
                                           int horizon_days, int draws,
                                           std::uint64_t seed) {
  const Eigen::Index N = weights.size();
  if (mean.size() != N || cov.rows() != N || cov.cols() != N) {
    throw std::invalid_argument(
        "simulate_portfolio_returns: dimension mismatch between mean, cov, "
        "weights");
  }
  if (horizon_days < 1 || draws < 1) {
    throw std::invalid_argument(
        "simulate_portfolio_returns: horizon_days and draws must be >= 1");
  }

  // Cholesky cov = L L^T. Fails loudly if cov is not positive (semi)definite.
  Eigen::LLT<Eigen::MatrixXd> llt(cov);
  if (llt.info() != Eigen::Success) {
    throw std::invalid_argument(
        "simulate_portfolio_returns: covariance is not positive definite "
        "(Cholesky failed)");
  }
  const Eigen::MatrixXd L = llt.matrixL();

  // For a linear book, w . (mean + L z) = (w . mean) + (L^T w) . z. Precompute
  // c = L^T w and the per-step drift; we still draw the full N-vector z so the
  // simulation is genuinely multivariate (and extensible to non-linear books).
  const Eigen::VectorXd c = L.transpose() * weights;
  const double step_drift = weights.dot(mean);

  // Draws come from risk::standard_normal rather than
  // std::normal_distribution: the latter's algorithm is implementation
  // defined, so the same seed gives different numbers under libstdc++ and
  // libc++ and the committed report would not reproduce across platforms.
  std::mt19937_64 gen(seed);

  Eigen::VectorXd sims(draws);
  Eigen::VectorXd z(N);
  for (int d = 0; d < draws; ++d) {
    double path = 0.0;
    for (int s = 0; s < horizon_days; ++s) {
      for (Eigen::Index i = 0; i < N; ++i) z(i) = standard_normal(gen);
      path += step_drift + c.dot(z);
    }
    sims(d) = path;
  }
  return sims;
}

double monte_carlo_var(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov,
                       const Eigen::VectorXd& weights, double confidence,
                       int horizon_days, int draws, std::uint64_t seed) {
  if (!(confidence > 0.0 && confidence < 1.0)) {
    throw std::invalid_argument(
        "monte_carlo_var: confidence must be in (0, 1)");
  }
  Eigen::VectorXd sims =
      simulate_portfolio_returns(mean, cov, weights, horizon_days, draws, seed);
  std::vector<double> v(sims.data(), sims.data() + sims.size());
  const double q = empirical_quantile(v, 1.0 - confidence);
  return -q;  // path already covers the horizon; positive loss
}

}  // namespace risk
