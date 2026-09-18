#include "risk/covariance.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace risk {

bool is_symmetric(const Eigen::MatrixXd& M, double tol) {
  if (M.rows() != M.cols()) return false;
  return (M - M.transpose()).cwiseAbs().maxCoeff() <= tol;
}

bool is_psd(const Eigen::MatrixXd& M, double tol) {
  if (!is_symmetric(M, std::max(tol, 1e-10))) return false;
  // Symmetric eigen-solver: eigenvalues are real and sorted ascending.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> const es(M);
  if (es.info() != Eigen::Success) return false;
  return es.eigenvalues().minCoeff() >= -tol;
}

MatrixDiagnostics diagnose_matrix(const Eigen::MatrixXd& M,
                                  double condition_threshold) {
  MatrixDiagnostics d;
  if (M.rows() != M.cols() || M.rows() == 0) {
    throw std::invalid_argument(
        "diagnose_matrix: need a non-empty square matrix");
  }
  d.symmetric = is_symmetric(M);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> const es(M);
  if (es.info() != Eigen::Success) {
    throw std::runtime_error("diagnose_matrix: eigen decomposition failed");
  }
  const Eigen::VectorXd& ev = es.eigenvalues();  // ascending
  d.min_eigenvalue = ev.minCoeff();
  d.max_eigenvalue = ev.maxCoeff();
  d.psd = d.symmetric && d.min_eigenvalue >= -1e-10;

  // A non-positive smallest eigenvalue means the matrix is singular to
  // working precision. Reporting infinity is more useful than a huge finite
  // number that invites someone to keep going.
  d.condition_number = (d.min_eigenvalue > 0.0)
                           ? d.max_eigenvalue / d.min_eigenvalue
                           : std::numeric_limits<double>::infinity();
  d.ill_conditioned = !(d.condition_number < condition_threshold);
  return d;
}

Eigen::MatrixXd to_return_matrix(const std::vector<ReturnSeries>& series) {
  if (series.empty()) {
    throw std::invalid_argument("to_return_matrix: no series provided");
  }
  const Eigen::Index T = series.front().size();
  const auto N = static_cast<Eigen::Index>(series.size());
  if (T < 1) {
    throw std::invalid_argument("to_return_matrix: empty series");
  }
  Eigen::MatrixXd X(T, N);
  for (Eigen::Index j = 0; j < N; ++j) {
    if (series[static_cast<std::size_t>(j)].size() != T) {
      throw std::invalid_argument(
          "to_return_matrix: series have mismatched lengths");
    }
    X.col(j) = series[static_cast<std::size_t>(j)].returns();
  }
  return X;
}

Eigen::MatrixXd sample_covariance(const Eigen::MatrixXd& X) {
  const Eigen::Index T = X.rows();
  if (T < 2) {
    throw std::invalid_argument(
        "sample_covariance: need at least 2 observations");
  }
  // Demean each column, then S = Xc^T Xc / (T-1).
  const Eigen::RowVectorXd mean = X.colwise().mean();
  const Eigen::MatrixXd Xc = X.rowwise() - mean;
  Eigen::MatrixXd S = (Xc.transpose() * Xc) / static_cast<double>(T - 1);
  // Force exact symmetry (kill asymmetric round-off) before the check.
  S = 0.5 * (S + S.transpose());
  assert(is_psd(S) && "sample_covariance produced a non-PSD matrix");
  return S;
}

Eigen::VectorXd ewma_weights(Eigen::Index T, double lambda) {
  if (T < 1) throw std::invalid_argument("ewma_weights: T must be >= 1");
  if (!(lambda > 0.0 && lambda < 1.0)) {
    throw std::invalid_argument("ewma_weights: lambda must be in (0, 1)");
  }
  Eigen::VectorXd w(T);
  // age = T-1-t : the most recent row (t = T-1) has age 0 and the largest
  // weight lambda^0 = 1; the oldest row has the smallest.
  for (Eigen::Index t = 0; t < T; ++t) {
    const auto age = static_cast<double>(T - 1 - t);
    w(t) = std::pow(lambda, age);
  }
  w /= w.sum();  // normalize to a probability vector
  return w;
}

Eigen::MatrixXd ewma_covariance(const Eigen::MatrixXd& X, double lambda) {
  const Eigen::Index T = X.rows();
  if (T < 2) {
    throw std::invalid_argument(
        "ewma_covariance: need at least 2 observations");
  }
  const Eigen::VectorXd w = ewma_weights(T, lambda);

  // Weighted mean (row vector over assets).
  Eigen::RowVectorXd const wmean = w.transpose() * X;  // (1xT)(TxN) = 1xN
  const Eigen::MatrixXd Xc = X.rowwise() - wmean;

  // Weighted covariance: Xc^T diag(w) Xc. PSD since w >= 0.
  Eigen::MatrixXd S = Xc.transpose() * w.asDiagonal() * Xc;
  S = 0.5 * (S + S.transpose());
  assert(is_psd(S) && "ewma_covariance produced a non-PSD matrix");
  return S;
}

LedoitWolf ledoit_wolf_covariance(const Eigen::MatrixXd& X) {
  const Eigen::Index T = X.rows();
  const Eigen::Index N = X.cols();
  if (T < 2) {
    throw std::invalid_argument(
        "ledoit_wolf_covariance: need at least 2 observations");
  }
  if (N < 2) {
    throw std::invalid_argument(
        "ledoit_wolf_covariance: need at least 2 assets");
  }
  const auto Td = static_cast<double>(T);

  // Demeaned data and the MLE (1/T) sample covariance used in the derivation.
  const Eigen::RowVectorXd mean = X.colwise().mean();
  const Eigen::MatrixXd Xc = X.rowwise() - mean;         // T x N
  const Eigen::MatrixXd S = (Xc.transpose() * Xc) / Td;  // N x N

  const Eigen::VectorXd var = S.diagonal();    // s_ii
  const Eigen::VectorXd sd = var.cwiseSqrt();  // sqrt(s_ii)

  // Average sample correlation r-bar over the off-diagonals (i < j).
  double sum_corr = 0.0;
  long pairs = 0;
  for (Eigen::Index i = 0; i < N; ++i) {
    for (Eigen::Index j = i + 1; j < N; ++j) {
      const double denom = sd(i) * sd(j);
      if (denom > 0.0) sum_corr += S(i, j) / denom;
      ++pairs;
    }
  }
  const double rbar = (pairs > 0) ? sum_corr / static_cast<double>(pairs) : 0.0;

  // Constant-correlation target F: f_ii = s_ii, f_ij = rbar * sqrt(s_ii s_jj).
  Eigen::MatrixXd F = rbar * (sd * sd.transpose());
  F.diagonal() = var;

  // --- Shrinkage intensity (Ledoit-Wolf 2004, constant-correlation case) ---

  // pi_hat: sum over i,j of asymptotic variances of sample covariance entries.
  // pi_ij = (1/T) sum_t (x_ti x_tj - s_ij)^2.
  Eigen::MatrixXd pi_mat = Eigen::MatrixXd::Zero(N, N);
  for (Eigen::Index t = 0; t < T; ++t) {
    const Eigen::RowVectorXd xt = Xc.row(t);
    // outer product of this observation, minus S, squared elementwise.
    Eigen::MatrixXd const d = (xt.transpose() * xt) - S;
    pi_mat += d.cwiseProduct(d);
  }
  pi_mat /= Td;
  const double pi_hat = pi_mat.sum();

  // rho_hat: diagonal part + off-diagonal cross terms with the target.
  double rho_hat = pi_mat.diagonal().sum();
  for (Eigen::Index i = 0; i < N; ++i) {
    for (Eigen::Index j = 0; j < N; ++j) {
      if (i == j) continue;
      if (sd(i) == 0.0 || sd(j) == 0.0) continue;
      // theta_ii,ij = (1/T) sum_t (x_ti^2 - s_ii)(x_ti x_tj - s_ij)
      // theta_jj,ij = (1/T) sum_t (x_tj^2 - s_jj)(x_ti x_tj - s_ij)
      double theta_ii = 0.0;
      double theta_jj = 0.0;
      for (Eigen::Index t = 0; t < T; ++t) {
        const double xi = Xc(t, i);
        const double xj = Xc(t, j);
        const double dij = xi * xj - S(i, j);
        theta_ii += (xi * xi - S(i, i)) * dij;
        theta_jj += (xj * xj - S(j, j)) * dij;
      }
      theta_ii /= Td;
      theta_jj /= Td;
      rho_hat +=
          rbar * 0.5 * (sd(j) / sd(i) * theta_ii + sd(i) / sd(j) * theta_jj);
    }
  }

  // gamma_hat: squared Frobenius distance between target and sample cov.
  const double gamma_hat = (F - S).squaredNorm();

  // Optimal shrinkage delta* = (pi - rho)/gamma / T, clipped to [0, 1].
  double delta = 0.0;
  if (gamma_hat > 0.0) {
    const double kappa = (pi_hat - rho_hat) / gamma_hat;
    delta = kappa / Td;
  }
  delta = std::max(0.0, std::min(1.0, delta));

  Eigen::MatrixXd shrunk = delta * F + (1.0 - delta) * S;
  shrunk = 0.5 * (shrunk + shrunk.transpose());
  assert(is_psd(shrunk) && "ledoit_wolf_covariance produced a non-PSD matrix");

  return LedoitWolf{shrunk, delta, rbar};
}

Eigen::MatrixXd cov_to_correlation(const Eigen::MatrixXd& cov) {
  if (!is_symmetric(cov)) {
    throw std::invalid_argument("cov_to_correlation: matrix not symmetric");
  }
  const Eigen::VectorXd sd = cov.diagonal().cwiseSqrt();
  const Eigen::Index N = cov.rows();
  Eigen::MatrixXd corr(N, N);
  for (Eigen::Index i = 0; i < N; ++i) {
    for (Eigen::Index j = 0; j < N; ++j) {
      const double denom = sd(i) * sd(j);
      corr(i, j) = (denom > 0.0) ? cov(i, j) / denom : 0.0;
    }
  }
  corr.diagonal().setOnes();
  return corr;
}

}  // namespace risk
