// Measure covariance estimator error against a known ground truth.
//
// On real data you can never see the estimation error: the true covariance is
// not observable, so "which estimator is best" degenerates into an argument.
// The way out is a simulation study. Fix a covariance matrix Sigma, draw
// samples from it, and measure how far each estimator lands from the answer
// you already know.
//
// The ground truth here is not invented. It is the factor model fitted to the
// committed FRED sample:
//
//     Sigma_true = B Sigma_F B' + D
//
// with B the OLS betas from config/factor_betas.json, Sigma_F the sample
// covariance of the observed factors, and D the diagonal of fitted residual
// variances. So Sigma_true has the eigenvalue structure of a real multi-asset
// book -- a dominant equity direction, a near-degenerate yield curve, three
// highly correlated equity indices -- rather than the well-conditioned
// matrices that make every estimator look fine.
//
// For each sample size n and each replication we draw n Gaussian returns from
// Sigma_true and record, per estimator:
//
//   * Frobenius norm error ||Sigma_hat - Sigma_true||_F
//   * condition number of Sigma_hat
//   * realised portfolio volatility against the true value, for the book's
//     actual weights
//   * the largest and smallest eigenvalues against the true spectrum
//   * 99% parametric VaR against the analytic truth, which is known exactly
//     because the data-generating process is Gaussian with known Sigma
//
// Output: reports/estimator_study.csv (one row per estimator and sample size)
// and reports/estimator_spectrum.csv (the eigenvalue comparison).

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "risk/config.hpp"
#include "risk/covariance.hpp"
#include "risk/decomposition.hpp"
#include "risk/portfolio.hpp"
#include "risk/stress_test.hpp"
#include "risk/var.hpp"

namespace fs = std::filesystem;
using namespace risk;

namespace {

struct Args {
  std::string portfolio = "config/portfolio.json";
  std::string data = "data/returns/";
  std::string factors = "data/factors.csv";
  std::string output = "reports/";
  int replications = 400;
  std::uint64_t seed = 20260918;
  std::vector<int> sample_sizes{60, 125, 250, 500, 1260};
};

void usage() {
  std::cout
      << "Usage: estimator_study [options]\n"
      << "  --portfolio FILE   portfolio config (default "
         "config/portfolio.json)\n"
      << "  --data DIR         return CSV directory (default data/returns/)\n"
      << "  --factors FILE     factor series (default data/factors.csv)\n"
      << "  --output DIR       output directory (default reports/)\n"
      << "  --replications N   Monte Carlo replications per size (default "
         "400)\n"
      << "  --seed N           RNG seed (default 20260918)\n";
}

// Running mean and standard error, so the study reports Monte Carlo
// uncertainty on its own estimates instead of a bare point number.
class Accumulator {
 public:
  void add(double x) {
    ++n_;
    const double delta = x - mean_;
    mean_ += delta / static_cast<double>(n_);
    m2_ += delta * (x - mean_);
  }
  [[nodiscard]] double mean() const { return mean_; }
  [[nodiscard]] double stdev() const {
    return (n_ > 1) ? std::sqrt(m2_ / static_cast<double>(n_ - 1)) : 0.0;
  }
  [[nodiscard]] double standard_error() const {
    return (n_ > 1) ? stdev() / std::sqrt(static_cast<double>(n_)) : 0.0;
  }
  [[nodiscard]] long count() const { return n_; }

 private:
  long n_ = 0;
  double mean_ = 0.0;
  double m2_ = 0.0;
};

struct Metrics {
  Accumulator frobenius;
  Accumulator condition;
  // Condition number of the implied CORRELATION matrix. The covariance
  // condition number conflates genuine collinearity with the fact that
  // 2y Treasuries run at 2% vol and crude at 41%; scaling to unit diagonal
  // isolates the part that actually threatens an inversion.
  Accumulator correlation_condition;
  Accumulator shrinkage;  // Ledoit-Wolf delta*, zero for the others
  Accumulator vol_error;  // signed: estimated - true
  Accumulator abs_vol_error;
  Accumulator var_error;  // signed: estimated - true, 99% 1-day
  Accumulator abs_var_error;
  Accumulator min_eigenvalue;
  Accumulator max_eigenvalue;
};

}  // namespace

int main(int argc, char** argv) try {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + flag);
      }
      return argv[++i];
    };
    if (s == "--portfolio")
      args.portfolio = next("--portfolio");
    else if (s == "--data")
      args.data = next("--data");
    else if (s == "--factors")
      args.factors = next("--factors");
    else if (s == "--output")
      args.output = next("--output");
    else if (s == "--replications")
      args.replications = std::stoi(next("--replications"));
    else if (s == "--seed")
      args.seed = std::stoull(next("--seed"));
    else if (s == "--help" || s == "-h") {
      usage();
      return 0;
    } else
      throw std::invalid_argument("unknown argument: " + s);
  }
  if (args.replications < 2) {
    throw std::invalid_argument("--replications must be at least 2");
  }

  // ---- ground truth --------------------------------------------------------
  const EngineConfig cfg = load_engine_config(args.portfolio);
  const Portfolio portfolio = make_portfolio(cfg);
  const Eigen::MatrixXd observed = load_return_matrix(cfg, args.data);
  const FactorSeries fs = load_factor_series(args.factors);
  const FactorBetas betas = load_factor_betas(cfg.factor_betas_path);

  const auto N = portfolio.size();
  const auto K = static_cast<Eigen::Index>(fs.names.size());
  const Eigen::MatrixXd factor_cov = sample_covariance(fs.values);

  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N, K);
  for (Eigen::Index i = 0; i < N; ++i) {
    auto it = betas.find(cfg.assets[static_cast<std::size_t>(i)]);
    if (it == betas.end()) continue;
    for (Eigen::Index k = 0; k < K; ++k) {
      auto f = it->second.find(fs.names[static_cast<std::size_t>(k)]);
      if (f != it->second.end()) B(i, k) = f->second;
    }
  }

  // Residual variances: what the factor model leaves unexplained per asset,
  // measured on the observed sample. Floored at a small positive number so
  // Sigma_true is strictly positive definite even where a constituent is
  // definitionally identical to a factor and its residual is numerically zero.
  const Eigen::MatrixXd fitted = fs.values * B.transpose();  // T x N
  const Eigen::MatrixXd resid = observed - fitted;
  Eigen::VectorXd specific(N);
  const double floor_var = 1e-14;
  for (Eigen::Index i = 0; i < N; ++i) {
    const double v =
        (resid.col(i).array() - resid.col(i).mean()).square().sum() /
        static_cast<double>(resid.rows() - K - 1);
    specific(i) = std::max(v, floor_var);
  }

  Eigen::MatrixXd sigma_true =
      B * factor_cov * B.transpose() + Eigen::MatrixXd(specific.asDiagonal());
  sigma_true = 0.5 * (sigma_true + sigma_true.transpose());

  const auto truth_diag = diagnose_matrix(sigma_true);
  if (!truth_diag.psd) {
    throw std::runtime_error("ground-truth covariance is not PSD");
  }

  const Eigen::VectorXd& w = portfolio.weights();
  const double true_vol = std::sqrt(w.dot(sigma_true * w));
  // The DGP is Gaussian with known Sigma and zero drift, so the 99% VaR is
  // analytic. No estimator can beat this number; the study measures how far
  // each one lands from it.
  const double true_var99 = normal_ppf(0.99) * true_vol;

  std::cout << "ground truth from the fitted factor model\n"
            << "  assets              " << N << "\n"
            << "  factors             " << K << "\n"
            << "  portfolio vol       " << true_vol * 100.0 << "% daily\n"
            << "  true 99% 1d VaR     " << true_var99 * 100.0 << "%\n"
            << "  cond(Sigma)         " << truth_diag.condition_number << "\n"
            << "  cond(correlation)   "
            << diagnose_matrix(cov_to_correlation(sigma_true)).condition_number
            << "\n"
            << "  min/max eigenvalue  " << truth_diag.min_eigenvalue << " / "
            << truth_diag.max_eigenvalue << "\n\n";

  Eigen::LLT<Eigen::MatrixXd> llt(sigma_true);
  if (llt.info() != Eigen::Success) {
    throw std::runtime_error("ground-truth covariance failed Cholesky");
  }
  const Eigen::MatrixXd L = llt.matrixL();

  // ---- the study -----------------------------------------------------------
  const std::vector<std::string> estimators = {"Sample", "EWMA", "Ledoit-Wolf"};
  std::map<std::string, std::map<int, Metrics>> results;
  std::map<int, std::map<std::string, std::vector<Accumulator>>> spectra;

  for (int n : args.sample_sizes) {
    // One seed per sample size, so a rerun with more sizes does not perturb
    // the results for the sizes already reported.
    std::mt19937_64 gen(args.seed + static_cast<std::uint64_t>(n));
    std::normal_distribution<double> nd(0.0, 1.0);

    for (const auto& e : estimators) {
      spectra[n][e].assign(static_cast<std::size_t>(N), Accumulator{});
    }

    for (int rep = 0; rep < args.replications; ++rep) {
      Eigen::MatrixXd X(n, N);
      for (int t = 0; t < n; ++t) {
        Eigen::VectorXd z(N);
        for (Eigen::Index i = 0; i < N; ++i) z(i) = nd(gen);
        X.row(t) = (L * z).transpose();
      }

      for (const auto& name : estimators) {
        Eigen::MatrixXd est;
        double delta = 0.0;
        if (name == "Sample") {
          est = sample_covariance(X);
        } else if (name == "EWMA") {
          est = ewma_covariance(X, cfg.ewma_lambda);
        } else {
          const auto lw = ledoit_wolf_covariance(X);
          est = lw.cov;
          delta = lw.shrinkage;
        }

        Metrics& m = results[name][n];
        m.shrinkage.add(delta);
        m.frobenius.add((est - sigma_true).norm());

        const auto d = diagnose_matrix(est);
        if (std::isfinite(d.condition_number))
          m.condition.add(d.condition_number);
        const auto dc = diagnose_matrix(cov_to_correlation(est));
        if (std::isfinite(dc.condition_number)) {
          m.correlation_condition.add(dc.condition_number);
        }
        m.min_eigenvalue.add(d.min_eigenvalue);
        m.max_eigenvalue.add(d.max_eigenvalue);

        const double vol = std::sqrt(std::max(0.0, w.dot(est * w)));
        m.vol_error.add(vol - true_vol);
        m.abs_vol_error.add(std::abs(vol - true_vol));

        const double var99 = normal_ppf(0.99) * vol;
        m.var_error.add(var99 - true_var99);
        m.abs_var_error.add(std::abs(var99 - true_var99));

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(est);
        const Eigen::VectorXd& ev = es.eigenvalues();  // ascending
        for (Eigen::Index i = 0; i < N; ++i) {
          spectra[n][name][static_cast<std::size_t>(i)].add(ev(i));
        }
      }
    }
  }

  // ---- output --------------------------------------------------------------
  fs::create_directories(args.output);

  {
    std::ofstream out(fs::path(args.output) / "estimator_study.csv");
    out << std::setprecision(10);
    out << "estimator,sample_size,replications,"
           "frobenius_mean,frobenius_se,"
           "condition_mean,condition_se,"
           "correlation_condition_mean,shrinkage_mean,"
           "vol_bias,vol_abs_error,vol_abs_error_se,true_vol,"
           "var99_bias,var99_abs_error,var99_abs_error_se,true_var99,"
           "min_eigenvalue_mean,max_eigenvalue_mean,"
           "true_min_eigenvalue,true_max_eigenvalue\n";
    for (const auto& name : estimators) {
      for (int n : args.sample_sizes) {
        const Metrics& m = results[name][n];
        out << name << ',' << n << ',' << args.replications << ','
            << m.frobenius.mean() << ',' << m.frobenius.standard_error() << ','
            << m.condition.mean() << ',' << m.condition.standard_error() << ','
            << m.correlation_condition.mean() << ',' << m.shrinkage.mean()
            << ',' << m.vol_error.mean() << ',' << m.abs_vol_error.mean() << ','
            << m.abs_vol_error.standard_error() << ',' << true_vol << ','
            << m.var_error.mean() << ',' << m.abs_var_error.mean() << ','
            << m.abs_var_error.standard_error() << ',' << true_var99 << ','
            << m.min_eigenvalue.mean() << ',' << m.max_eigenvalue.mean() << ','
            << truth_diag.min_eigenvalue << ',' << truth_diag.max_eigenvalue
            << '\n';
      }
    }
  }

  {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sigma_true);
    const Eigen::VectorXd& true_ev = es.eigenvalues();
    std::ofstream out(fs::path(args.output) / "estimator_spectrum.csv");
    out << std::setprecision(10);
    out << "estimator,sample_size,index,estimated_eigenvalue,true_eigenvalue\n";
    for (int n : args.sample_sizes) {
      for (const auto& name : estimators) {
        for (Eigen::Index i = 0; i < N; ++i) {
          out << name << ',' << n << ',' << i << ','
              << spectra[n][name][static_cast<std::size_t>(i)].mean() << ','
              << true_ev(i) << '\n';
        }
      }
    }
  }

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "n      estimator        ||S-Sigma||_F  cond(corr)   delta*"
               "   |vol err|   |VaR99 err|\n";
  for (int n : args.sample_sizes) {
    for (const auto& name : estimators) {
      const Metrics& m = results[name][n];
      std::cout << std::setw(6) << std::left << n << " " << std::setw(14)
                << name << std::right << std::setw(11)
                << m.frobenius.mean() * 1e4 << "e-4" << std::setw(12)
                << m.correlation_condition.mean() << std::setw(9)
                << m.shrinkage.mean() << std::setw(10)
                << m.abs_vol_error.mean() * 1e4 << "bp" << std::setw(11)
                << m.abs_var_error.mean() * 1e4 << "bp\n";
    }
  }
  std::cout << "\nwrote " << args.output << "estimator_study.csv and "
            << "estimator_spectrum.csv\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << "\n";
  return 1;
}
