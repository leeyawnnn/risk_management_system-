#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "risk/portfolio.hpp"
#include "risk/return_series.hpp"

namespace risk {

// Everything config/portfolio.json declares, parsed once. The CLI and both
// study executables share this so they cannot drift into reading the same
// file three slightly different ways.
struct EngineConfig {
  std::string name = "Portfolio";
  double notional = 1.0;
  ReturnType return_type = ReturnType::Log;
  MissingPolicy missing_policy = MissingPolicy::Skip;
  double annualization_factor = 252.0;
  std::string covariance_method = "sample";
  double ewma_lambda = 0.94;
  std::vector<double> confidences{0.95, 0.99};
  std::vector<int> horizons{1, 10};
  int mc_draws = 100000;
  std::uint64_t seed = 42;

  std::string stress_scenarios_path;
  std::string factor_betas_path;
  std::string factor_series_path;

  std::vector<std::string> assets;
  std::vector<double> weights;
  std::map<std::string, std::string> sectors;
};

EngineConfig load_engine_config(const std::string& path);

Portfolio make_portfolio(const EngineConfig& cfg);

// Load one CSV per asset from `data_dir` and stack them into a (T x N)
// matrix, in the order the config lists the positions.
Eigen::MatrixXd load_return_matrix(
    const EngineConfig& cfg, const std::string& data_dir,
    std::vector<ReturnSeries>* out_series = nullptr);

// Dispatch on cfg.covariance_method. Throws on an unknown name rather than
// silently falling back to the sample estimator.
Eigen::MatrixXd estimate_covariance(const std::string& method,
                                    const Eigen::MatrixXd& returns,
                                    double ewma_lambda);

}  // namespace risk
