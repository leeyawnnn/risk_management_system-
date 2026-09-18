#include "risk/config.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "risk/covariance.hpp"

namespace fs = std::filesystem;

namespace risk {

namespace {

ReturnType parse_return_type(const std::string& s) {
  if (s == "log") return ReturnType::Log;
  if (s == "simple") return ReturnType::Simple;
  throw std::invalid_argument(R"(return_type must be "log" or "simple")");
}

MissingPolicy parse_missing(const std::string& s) {
  if (s == "skip") return MissingPolicy::Skip;
  if (s == "fill_forward" || s == "ffill") return MissingPolicy::FillForward;
  throw std::invalid_argument(
      R"(missing_policy must be "skip" or "fill_forward")");
}

}  // namespace

EngineConfig load_engine_config(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::invalid_argument("load_engine_config: cannot open " + path);
  }
  nlohmann::json j;
  in >> j;

  EngineConfig cfg;
  cfg.name = j.value("name", cfg.name);
  cfg.notional = j.value("notional", cfg.notional);
  cfg.return_type = parse_return_type(j.value("return_type", "log"));
  cfg.missing_policy = parse_missing(j.value("missing_policy", "skip"));
  cfg.annualization_factor =
      j.value("annualization_factor", cfg.annualization_factor);
  cfg.covariance_method = j.value("covariance", cfg.covariance_method);
  cfg.ewma_lambda = j.value("ewma_lambda", cfg.ewma_lambda);
  cfg.confidences = j.value("confidences", cfg.confidences);
  cfg.horizons = j.value("horizons", cfg.horizons);
  if (j.contains("monte_carlo")) {
    cfg.mc_draws = j["monte_carlo"].value("draws", cfg.mc_draws);
    cfg.seed = j["monte_carlo"].value("seed", cfg.seed);
  }
  cfg.stress_scenarios_path = j.value("stress_scenarios", std::string{});
  cfg.factor_betas_path = j.value("factor_betas", std::string{});
  cfg.factor_series_path = j.value("factor_series", std::string{});

  if (!j.contains("positions") || !j["positions"].is_array() ||
      j["positions"].empty()) {
    throw std::invalid_argument(
        R"(load_engine_config: "positions" must be a non-empty array in )" +
        path);
  }
  for (const auto& pos : j.at("positions")) {
    const std::string asset = pos.at("asset").get<std::string>();
    cfg.assets.push_back(asset);
    cfg.weights.push_back(pos.at("weight").get<double>());
    if (pos.contains("sector")) cfg.sectors[asset] = pos["sector"];
  }
  return cfg;
}

Portfolio make_portfolio(const EngineConfig& cfg) {
  Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(
      cfg.weights.data(), static_cast<Eigen::Index>(cfg.weights.size()));
  return {cfg.assets, w, cfg.notional};
}

Eigen::MatrixXd load_return_matrix(const EngineConfig& cfg,
                                   const std::string& data_dir,
                                   std::vector<ReturnSeries>* out_series) {
  std::vector<ReturnSeries> series;
  series.reserve(cfg.assets.size());
  for (const auto& asset : cfg.assets) {
    const std::string path = (fs::path(data_dir) / (asset + ".csv")).string();
    series.push_back(ReturnSeries::from_csv(
        path, cfg.return_type, cfg.missing_policy, cfg.annualization_factor));
  }
  Eigen::MatrixXd X = to_return_matrix(series);
  if (out_series != nullptr) *out_series = std::move(series);
  return X;
}

Eigen::MatrixXd estimate_covariance(const std::string& method,
                                    const Eigen::MatrixXd& returns,
                                    double ewma_lambda) {
  if (method == "sample") return sample_covariance(returns);
  if (method == "ewma") return ewma_covariance(returns, ewma_lambda);
  if (method == "ledoit_wolf") return ledoit_wolf_covariance(returns).cov;
  throw std::invalid_argument("unknown covariance method: " + method);
}

}  // namespace risk
