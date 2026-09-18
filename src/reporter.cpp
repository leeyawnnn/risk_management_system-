#include "risk/reporter.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "risk/covariance.hpp"
#include "risk/cvar.hpp"
#include "risk/figures.hpp"
#include "risk/var.hpp"

namespace risk {

namespace {

std::string pct(double v, int prec = 2) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << v * 100.0 << "%";
  return os.str();
}

std::string money(double v) {
  const bool negative = v < 0.0;
  std::ostringstream os;
  os << std::fixed << std::setprecision(0) << std::abs(v);
  const std::string digits = os.str();
  std::string grouped;
  int count = 0;
  for (char c : std::ranges::reverse_view(digits)) {
    if (count > 0 && count % 3 == 0) grouped.push_back(',');
    grouped.push_back(c);
    ++count;
  }
  std::ranges::reverse(grouped);
  return (negative ? "-$" : "$") + grouped;
}

std::string num(double v, int prec = 4) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << v;
  return os.str();
}

std::vector<std::vector<double>> mat_to_vec(const Eigen::MatrixXd& M) {
  std::vector<std::vector<double>> out(static_cast<std::size_t>(M.rows()));
  for (Eigen::Index i = 0; i < M.rows(); ++i) {
    auto& row = out[static_cast<std::size_t>(i)];
    row.resize(static_cast<std::size_t>(M.cols()));
    for (Eigen::Index j = 0; j < M.cols(); ++j) {
      row[static_cast<std::size_t>(j)] = M(i, j);
    }
  }
  return out;
}

struct SampleMoments {
  double mean = 0.0;
  double stdev = 0.0;
  double skewness = 0.0;
  double excess_kurtosis = 0.0;
};

SampleMoments moments(const Eigen::VectorXd& x) {
  const auto n = static_cast<double>(x.size());
  SampleMoments m;
  if (n < 2.0) return m;
  m.mean = x.mean();
  const Eigen::ArrayXd d = x.array() - m.mean;
  const double m2 = d.square().sum() / n;
  m.stdev = std::sqrt(d.square().sum() / (n - 1.0));
  if (m2 > 0.0) {
    m.skewness = (d.cube().sum() / n) / std::pow(m2, 1.5);
    m.excess_kurtosis = (d.pow(4).sum() / n) / (m2 * m2) - 3.0;
  }
  return m;
}

// Bootstrap interval for one instrument's annualized volatility. The point of
// carrying this through the report is that a volatility is an estimate: on
// ~1,240 days the 95% interval on an equity vol is more than a point wide,
// and printing a single number invites the reader to forget that.
std::pair<double, double> bootstrap_vol_ci(const Eigen::VectorXd& col,
                                           double annualization, int resamples,
                                           std::uint64_t seed) {
  const auto n = static_cast<std::size_t>(col.size());
  if (n < 8 || resamples < 50) return {0.0, 0.0};
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);

  std::vector<double> vols;
  vols.reserve(static_cast<std::size_t>(resamples));
  const auto nn = static_cast<double>(n);
  for (int b = 0; b < resamples; ++b) {
    double sum = 0.0;
    double sumsq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double v = col(static_cast<Eigen::Index>(pick(gen)));
      sum += v;
      sumsq += v * v;
    }
    const double var = (sumsq - sum * sum / nn) / (nn - 1.0);
    vols.push_back(std::sqrt(std::max(0.0, var) * annualization));
  }
  std::ranges::sort(vols);
  const auto lo =
      static_cast<std::size_t>(0.025 * static_cast<double>(vols.size() - 1));
  const auto hi =
      static_cast<std::size_t>(0.975 * static_cast<double>(vols.size() - 1));
  return {vols[lo], vols[hi]};
}

const char* verdict(bool reject) {
  return reject ? "REJECT" : "pass";
}

}  // namespace

RiskReport build_report(const std::string& name, const Portfolio& portfolio,
                        const ReportInputs& in) {
  if (in.asset_returns == nullptr || in.cov == nullptr) {
    throw std::invalid_argument(
        "build_report: asset_returns and cov are required");
  }
  const Eigen::MatrixXd& X = *in.asset_returns;
  const Eigen::MatrixXd& cov = *in.cov;

  RiskReport rep;
  rep.portfolio_name = name;
  rep.as_of = in.as_of;
  rep.git_commit = in.git_commit;
  rep.seed = in.seed;
  rep.notional = portfolio.notional();
  rep.assets = portfolio.names();
  rep.weights = portfolio.weights();
  rep.covariance_method = in.cov_method;
  rep.annualization_factor = in.annualization_factor;
  rep.covariance = cov;
  rep.correlation = cov_to_correlation(cov);
  rep.covariance_diagnostics = diagnose_matrix(cov);
  rep.correlation_diagnostics = diagnose_matrix(rep.correlation);
  rep.daily_vol = portfolio.volatility(cov);
  rep.annual_vol = rep.daily_vol * std::sqrt(in.annualization_factor);
  rep.portfolio_returns = portfolio.return_series(X);
  rep.stress = in.stress;

  const SampleMoments port = moments(rep.portfolio_returns);
  rep.portfolio_skewness = port.skewness;
  rep.portfolio_excess_kurtosis = port.excess_kurtosis;

  const double port_mean = port.mean;
  const double port_sigma = rep.daily_vol;
  const Eigen::VectorXd asset_mean = X.colwise().mean().transpose();
  const Eigen::VectorXd& w = portfolio.weights();

  for (double conf : in.confidences) {
    for (int h : in.horizons) {
      rep.var_cvar.push_back({"Historical", conf, h,
                              historical_var(rep.portfolio_returns, conf, h),
                              historical_cvar(rep.portfolio_returns, conf, h)});
      rep.var_cvar.push_back({"Parametric", conf, h,
                              parametric_var(port_mean, port_sigma, conf, h),
                              parametric_cvar(port_mean, port_sigma, conf, h)});
      rep.var_cvar.push_back(
          {"MonteCarlo", conf, h,
           monte_carlo_var(asset_mean, cov, w, conf, h, in.mc_draws, in.seed),
           monte_carlo_cvar(asset_mean, cov, w, conf, h, in.mc_draws,
                            in.seed)});
    }
  }

  rep.attribution = compute_attribution(portfolio, cov, in.sectors);
  for (double conf : in.confidences) {
    rep.var_decomposition.push_back(
        decompose_var(portfolio, cov, asset_mean, conf, 1, in.sectors));
  }
  for (const auto& [asset, trade] : in.trades) {
    rep.incremental.push_back(
        incremental_var(portfolio, cov, asset_mean, asset, trade, 0.99, 1));
  }

  // ---- per-asset statistics ------------------------------------------------
  const Eigen::Index T = X.rows();
  const Eigen::Index N = X.cols();
  for (Eigen::Index a = 0; a < N; ++a) {
    const Eigen::VectorXd col = X.col(a);
    const SampleMoments am = moments(col);

    AssetStat st;
    st.name = rep.assets[static_cast<std::size_t>(a)];
    auto sit = in.sectors.find(st.name);
    st.sector = (sit != in.sectors.end()) ? sit->second : "";
    st.weight = w(a);
    st.annual_vol = am.stdev * std::sqrt(in.annualization_factor);
    st.annual_return = am.mean * in.annualization_factor;
    st.sharpe = (st.annual_vol > 0.0) ? st.annual_return / st.annual_vol : 0.0;
    st.skewness = am.skewness;
    st.excess_kurtosis = am.excess_kurtosis;
    st.pct_risk = rep.attribution.percent(a);

    // A distinct seed per instrument, derived from the run seed, so the
    // intervals are reproducible without every asset resampling identically.
    const auto [ci_lo, ci_hi] =
        bootstrap_vol_ci(col, in.annualization_factor, in.bootstrap_resamples,
                         in.seed + 1000 + static_cast<std::uint64_t>(a));
    st.annual_vol_ci_lower = ci_lo;
    st.annual_vol_ci_upper = ci_hi;

    const double cov_ap =
        ((col.array() - am.mean) * (rep.portfolio_returns.array() - port.mean))
            .sum() /
        static_cast<double>(T - 1);
    st.corr_to_portfolio = (am.stdev > 0.0 && port.stdev > 0.0)
                               ? cov_ap / (am.stdev * port.stdev)
                               : 0.0;
    rep.asset_stats.push_back(st);
  }

  // ---- estimator comparison ------------------------------------------------
  auto add_estimator = [&](const std::string& label, const Eigen::MatrixXd& C,
                           double shrinkage) {
    EstimatorStat es;
    es.method = label;
    es.daily_vol = portfolio.volatility(C);
    es.annual_vol = es.daily_vol * std::sqrt(in.annualization_factor);
    es.var95 = parametric_var(port_mean, es.daily_vol, 0.95, 1);
    es.var99 = parametric_var(port_mean, es.daily_vol, 0.99, 1);
    es.condition_number = diagnose_matrix(C).condition_number;
    es.correlation_condition_number =
        diagnose_matrix(cov_to_correlation(C)).condition_number;
    es.shrinkage = shrinkage;
    rep.estimator_comparison.push_back(es);
  };
  add_estimator("Sample", sample_covariance(X), 0.0);
  add_estimator("EWMA", ewma_covariance(X, in.ewma_lambda), 0.0);
  const auto lw = ledoit_wolf_covariance(X);
  add_estimator("Ledoit-Wolf", lw.cov, lw.shrinkage);

  // ---- backtests -----------------------------------------------------------
  // Every method at every confidence level, so the table says which of them
  // are calibrated rather than reporting one number and a picture.
  for (double conf : in.confidences) {
    struct MethodVar {
      const char* label;
      double var;
      double es;
    };
    const std::vector<MethodVar> methods = {
        {"Historical", historical_var(rep.portfolio_returns, conf, 1),
         historical_cvar(rep.portfolio_returns, conf, 1)},
        {"Parametric", parametric_var(port_mean, port_sigma, conf, 1),
         parametric_cvar(port_mean, port_sigma, conf, 1)},
        {"MonteCarlo",
         monte_carlo_var(asset_mean, cov, w, conf, 1, in.mc_draws, in.seed),
         monte_carlo_cvar(asset_mean, cov, w, conf, 1, in.mc_draws, in.seed)},
    };
    for (const auto& m : methods) {
      BacktestRow row;
      row.method = m.label;
      row.confidence = conf;
      row.var_level = m.var;
      row.es_level = m.es;
      row.kupiec = kupiec_pof(rep.portfolio_returns, m.var, conf);
      row.christoffersen = christoffersen(rep.portfolio_returns, m.var, conf);
      row.basel = basel_traffic_light(rep.portfolio_returns, m.var, conf, 250);
      row.es_test = acerbi_szekely(rep.portfolio_returns, m.var, m.es, conf,
                                   port_mean, port_sigma, 5000, in.seed);
      rep.backtests.push_back(row);
    }
  }

  // ---- factor decomposition and reverse stress -----------------------------
  if (in.factors != nullptr && in.betas != nullptr) {
    const Eigen::MatrixXd factor_cov = sample_covariance(in.factors->values);
    const auto K = static_cast<Eigen::Index>(in.factors->names.size());

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N, K);
    for (Eigen::Index i = 0; i < N; ++i) {
      auto ait = in.betas->find(rep.assets[static_cast<std::size_t>(i)]);
      if (ait == in.betas->end()) continue;
      for (Eigen::Index k = 0; k < K; ++k) {
        auto fit =
            ait->second.find(in.factors->names[static_cast<std::size_t>(k)]);
        if (fit != ait->second.end()) B(i, k) = fit->second;
      }
    }

    Eigen::VectorXd specific = Eigen::VectorXd::Zero(N);
    if (in.factors->values.rows() == X.rows()) {
      const Eigen::MatrixXd resid = X - in.factors->values * B.transpose();
      for (Eigen::Index i = 0; i < N; ++i) {
        specific(i) =
            (resid.col(i).array() - resid.col(i).mean()).square().sum() /
            static_cast<double>(resid.rows() - K - 1);
      }
    }

    rep.factor_decomposition = decompose_factor_risk(
        portfolio, *in.betas, in.factors->names, factor_cov, specific, cov);
    rep.has_factor_decomposition = true;

    for (double target : in.reverse_stress_targets) {
      rep.reverse_stress.push_back(reverse_stress(
          portfolio, *in.betas, in.factors->names, factor_cov, target));
    }
  }

  return rep;
}

std::string to_json(const RiskReport& rep, int indent) {
  using nlohmann::json;
  json j;
  j["provenance"] = {{"portfolio_name", rep.portfolio_name},
                     {"as_of", rep.as_of},
                     {"git_commit", rep.git_commit},
                     {"seed", rep.seed},
                     {"covariance_method", rep.covariance_method},
                     {"annualization_factor", rep.annualization_factor}};
  j["notional"] = rep.notional;

  json positions = json::array();
  for (Eigen::Index i = 0; i < rep.weights.size(); ++i) {
    positions.push_back({{"asset", rep.assets[static_cast<std::size_t>(i)]},
                         {"weight", rep.weights(i)}});
  }
  j["positions"] = positions;

  j["volatility"] = {{"daily", rep.daily_vol}, {"annual", rep.annual_vol}};
  j["portfolio_moments"] = {{"skewness", rep.portfolio_skewness},
                            {"excess_kurtosis", rep.portfolio_excess_kurtosis}};
  j["conditioning"] = {
      {"covariance",
       {{"min_eigenvalue", rep.covariance_diagnostics.min_eigenvalue},
        {"max_eigenvalue", rep.covariance_diagnostics.max_eigenvalue},
        {"condition_number", rep.covariance_diagnostics.condition_number},
        {"psd", rep.covariance_diagnostics.psd},
        {"ill_conditioned", rep.covariance_diagnostics.ill_conditioned}}},
      {"correlation",
       {{"condition_number", rep.correlation_diagnostics.condition_number},
        {"min_eigenvalue", rep.correlation_diagnostics.min_eigenvalue}}}};
  j["covariance"] = mat_to_vec(rep.covariance);
  j["correlation"] = mat_to_vec(rep.correlation);

  json vc = json::array();
  for (const auto& r : rep.var_cvar) {
    vc.push_back({{"method", r.method},
                  {"confidence", r.confidence},
                  {"horizon_days", r.horizon_days},
                  {"var", r.var},
                  {"var_dollar", r.var * rep.notional},
                  {"cvar", r.cvar},
                  {"cvar_dollar", r.cvar * rep.notional}});
  }
  j["var_cvar"] = vc;

  json bt = json::array();
  for (const auto& b : rep.backtests) {
    bt.push_back(
        {{"method", b.method},
         {"confidence", b.confidence},
         {"var_level", b.var_level},
         {"es_level", b.es_level},
         {"kupiec",
          {{"observations", b.kupiec.observations},
           {"exceptions", b.kupiec.exceptions},
           {"expected_rate", b.kupiec.expected_rate},
           {"observed_rate", b.kupiec.observed_rate},
           {"lr_statistic", b.kupiec.lr_statistic},
           {"p_value", b.kupiec.p_value},
           {"reject_at_95", b.kupiec.reject_at_95}}},
         {"christoffersen",
          {{"n00", b.christoffersen.n00},
           {"n01", b.christoffersen.n01},
           {"n10", b.christoffersen.n10},
           {"n11", b.christoffersen.n11},
           {"lr_independence", b.christoffersen.lr_independence},
           {"p_value_independence", b.christoffersen.p_value_independence},
           {"reject_independence_at_95",
            b.christoffersen.reject_independence_at_95},
           {"lr_conditional_coverage",
            b.christoffersen.lr_conditional_coverage},
           {"p_value_conditional_coverage",
            b.christoffersen.p_value_conditional_coverage},
           {"reject_conditional_coverage_at_95",
            b.christoffersen.reject_conditional_coverage_at_95}}},
         {"basel",
          {{"window", b.basel.window},
           {"exceptions", b.basel.exceptions},
           {"cumulative_probability", b.basel.cumulative_probability},
           {"zone", to_string(b.basel.zone)},
           {"window_complete", b.basel.window_complete}}},
         {"acerbi_szekely",
          {{"z1", b.es_test.z1},
           {"z1_defined", b.es_test.z1_defined},
           {"p_value_z1", b.es_test.p_value_z1},
           {"z2", b.es_test.z2},
           {"p_value_z2", b.es_test.p_value_z2},
           {"simulations", b.es_test.simulations}}}});
  }
  j["backtests"] = bt;

  const auto& a = rep.attribution;
  json attr;
  attr["portfolio_vol"] = a.portfolio_vol;
  json contribs = json::array();
  for (Eigen::Index i = 0; i < a.weights.size(); ++i) {
    contribs.push_back({{"asset", a.names[static_cast<std::size_t>(i)]},
                        {"weight", a.weights(i)},
                        {"marginal", a.marginal(i)},
                        {"component", a.component(i)},
                        {"percent", a.percent(i)}});
  }
  attr["contributions"] = contribs;
  attr["concentration"] = {
      {"herfindahl", a.concentration.herfindahl},
      {"effective_num_bets", a.concentration.effective_num_bets},
      {"max_contribution", a.concentration.max_contribution},
      {"max_contributor", a.concentration.max_contributor}};
  if (!a.sector_percent.empty()) {
    json sectors = json::object();
    for (const auto& [s, p] : a.sector_percent) {
      sectors[s] = {{"percent", p}, {"component", a.sector_component.at(s)}};
    }
    attr["sectors"] = sectors;
  }
  j["volatility_attribution"] = attr;

  json vd = json::array();
  for (const auto& d : rep.var_decomposition) {
    json rows = json::array();
    for (Eigen::Index i = 0; i < d.weights.size(); ++i) {
      rows.push_back({{"asset", d.names[static_cast<std::size_t>(i)]},
                      {"marginal_var", d.marginal(i)},
                      {"component_var", d.component(i)},
                      {"component_var_dollar", d.component(i) * rep.notional},
                      {"percent", d.percent(i)}});
    }
    vd.push_back({{"confidence", d.confidence},
                  {"horizon_days", d.horizon_days},
                  {"var", d.var},
                  {"euler_residual", d.euler_residual},
                  {"positions", rows}});
  }
  j["var_decomposition"] = vd;

  json inc = json::array();
  for (const auto& t : rep.incremental) {
    inc.push_back({{"asset", t.asset},
                   {"trade_notional", t.trade_notional},
                   {"var_before", t.var_before},
                   {"var_after", t.var_after},
                   {"incremental", t.incremental},
                   {"linear_estimate", t.linear_estimate}});
  }
  j["incremental_var"] = inc;

  if (rep.has_factor_decomposition) {
    const auto& d = rep.factor_decomposition;
    json factors = json::array();
    for (std::size_t k = 0; k < d.factors.size(); ++k) {
      const auto i = static_cast<Eigen::Index>(k);
      factors.push_back({{"factor", d.factors[k]},
                         {"exposure", d.exposures(i)},
                         {"variance_contribution", d.variance_contribution(i)},
                         {"percent_of_model_variance", d.percent_of_total(i)}});
    }
    j["factor_decomposition"] = {{"factors", factors},
                                 {"factor_variance", d.factor_variance},
                                 {"specific_variance", d.specific_variance},
                                 {"model_variance", d.model_variance},
                                 {"model_vol", d.model_vol},
                                 {"sample_variance", d.sample_variance},
                                 {"model_error", d.model_error},
                                 {"variance_explained", d.variance_explained}};
  }

  json st = json::array();
  for (const auto& s : rep.stress) {
    st.push_back({{"name", s.name},
                  {"pct_pnl", s.pct_pnl},
                  {"dollar_pnl", s.dollar_pnl}});
  }
  j["stress_tests"] = st;

  json rs = json::array();
  for (const auto& s : rep.reverse_stress) {
    json moves = json::object();
    for (std::size_t k = 0; k < s.factors.size(); ++k) {
      moves[s.factors[k]] = s.factor_move(static_cast<Eigen::Index>(k));
    }
    rs.push_back({{"target_loss_pct", s.target_loss_pct},
                  {"target_loss_dollar", s.target_loss_dollar},
                  {"factor_move", moves},
                  {"mahalanobis_distance", s.mahalanobis_distance},
                  {"factor_model_vol", s.factor_model_vol},
                  {"gaussian_probability", s.gaussian_probability}});
  }
  j["reverse_stress"] = rs;

  json as = json::array();
  for (const auto& s : rep.asset_stats) {
    as.push_back(
        {{"asset", s.name},
         {"sector", s.sector},
         {"weight", s.weight},
         {"annual_vol", s.annual_vol},
         {"annual_vol_ci", {s.annual_vol_ci_lower, s.annual_vol_ci_upper}},
         {"annual_return", s.annual_return},
         {"sharpe", s.sharpe},
         {"skewness", s.skewness},
         {"excess_kurtosis", s.excess_kurtosis},
         {"corr_to_portfolio", s.corr_to_portfolio},
         {"pct_risk", s.pct_risk}});
  }
  j["asset_stats"] = as;

  json ec = json::array();
  for (const auto& e : rep.estimator_comparison) {
    ec.push_back(
        {{"method", e.method},
         {"daily_vol", e.daily_vol},
         {"annual_vol", e.annual_vol},
         {"var95", e.var95},
         {"var99", e.var99},
         {"condition_number", e.condition_number},
         {"correlation_condition_number", e.correlation_condition_number},
         {"shrinkage", e.shrinkage}});
  }
  j["estimator_comparison"] = ec;

  return j.dump(indent);
}

std::string to_markdown(const RiskReport& rep) {
  std::ostringstream m;
  m << "# Risk report: " << rep.portfolio_name << "\n\n";
  m << "| | |\n|---|---|\n";
  m << "| Data as-of | " << (rep.as_of.empty() ? "unstamped" : rep.as_of)
    << " |\n";
  m << "| Code commit | "
    << (rep.git_commit.empty() ? "unstamped" : rep.git_commit) << " |\n";
  m << "| Seed | " << rep.seed << " |\n";
  m << "| Notional | " << money(rep.notional) << " |\n";
  m << "| Covariance estimator | " << rep.covariance_method << " |\n";
  m << "| Observations | " << rep.portfolio_returns.size() << " |\n";
  m << "| Portfolio volatility | " << pct(rep.daily_vol) << " daily / "
    << pct(rep.annual_vol) << " annual |\n";
  m << "| Return skewness | " << num(rep.portfolio_skewness, 2) << " |\n";
  m << "| Return excess kurtosis | " << num(rep.portfolio_excess_kurtosis, 2)
    << " |\n";
  m << "| cond(Sigma) | " << num(rep.covariance_diagnostics.condition_number, 1)
    << " |\n";
  m << "| cond(correlation) | "
    << num(rep.correlation_diagnostics.condition_number, 1) << " |\n\n";

  m << "## Value at Risk and Expected Shortfall\n\n";
  m << "| Method | Conf | Horizon | VaR | VaR ($) | CVaR | CVaR ($) |\n";
  m << "|---|---:|---:|---:|---:|---:|---:|\n";
  for (const auto& r : rep.var_cvar) {
    m << "| " << r.method << " | " << pct(r.confidence, 0) << " | "
      << r.horizon_days << "d | " << pct(r.var) << " | "
      << money(r.var * rep.notional) << " | " << pct(r.cvar) << " | "
      << money(r.cvar * rep.notional) << " |\n";
  }
  m << "\n";

  if (!rep.backtests.empty()) {
    m << "## Backtests\n\n";
    m << "Kupiec tests whether exceptions arrived at the advertised rate. "
         "Christoffersen tests whether they clustered, which is the failure "
         "mode that actually costs money. Basel reads the exception count over "
         "the trailing 250 sessions against the supervisory table. "
         "Acerbi-Szekely Z2 tests Expected Shortfall, which cannot be "
         "backtested by counting hits because it is not elicitable; a negative "
         "Z means realised tail losses were worse than the model said.\n\n";
    m << "| Method | Conf | VaR | Exc. | Exp. | Kupiec p | Christ. p | "
         "Cond. cov. p | Basel | AS Z2 | Z2 p |\n";
    m << "|---|---:|---:|---:|---:|---|---|---|---|---:|---:|\n";
    for (const auto& b : rep.backtests) {
      const double expected =
          b.kupiec.expected_rate * static_cast<double>(b.kupiec.observations);
      m << "| " << b.method << " | " << pct(b.confidence, 0) << " | "
        << pct(b.var_level) << " | " << b.kupiec.exceptions << " | "
        << num(expected, 1) << " | " << num(b.kupiec.p_value, 3) << " "
        << verdict(b.kupiec.reject_at_95) << " | "
        << num(b.christoffersen.p_value_independence, 3) << " "
        << verdict(b.christoffersen.reject_independence_at_95) << " | "
        << num(b.christoffersen.p_value_conditional_coverage, 3) << " "
        << verdict(b.christoffersen.reject_conditional_coverage_at_95) << " | "
        << to_string(b.basel.zone) << " (" << b.basel.exceptions << "/"
        << b.basel.window << ") | " << num(b.es_test.z2, 3) << " | "
        << num(b.es_test.p_value_z2, 3) << " |\n";
    }
    m << "\n";
  }

  if (!rep.asset_stats.empty()) {
    m << "## Instruments\n\n";
    m << "| Asset | Class | Weight | Ann. vol | 95% CI | Ann. return | Sharpe "
         "| "
         "Skew | Ex. kurt | Corr to book | % of risk |\n";
    m << "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& s : rep.asset_stats) {
      m << "| " << s.name << " | " << s.sector << " | " << pct(s.weight, 1)
        << " | " << pct(s.annual_vol) << " | [" << pct(s.annual_vol_ci_lower, 1)
        << ", " << pct(s.annual_vol_ci_upper, 1) << "] | "
        << pct(s.annual_return) << " | " << num(s.sharpe, 2) << " | "
        << num(s.skewness, 2) << " | " << num(s.excess_kurtosis, 2) << " | "
        << num(s.corr_to_portfolio, 2) << " | " << pct(s.pct_risk, 1) << " |\n";
    }
    m << "\n";
  }

  if (!rep.estimator_comparison.empty()) {
    m << "## Covariance estimators\n\n";
    m << "The same book under three estimators. The spread in the VaR columns "
         "is model risk; the condition numbers are why shrinkage exists.\n\n";
    m << "| Estimator | Daily vol | Annual vol | 95% VaR | 99% VaR | "
         "cond(Sigma) | cond(corr) | delta* |\n";
    m << "|---|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& e : rep.estimator_comparison) {
      m << "| " << e.method << " | " << pct(e.daily_vol) << " | "
        << pct(e.annual_vol) << " | " << pct(e.var95) << " | " << pct(e.var99)
        << " | " << num(e.condition_number, 1) << " | "
        << num(e.correlation_condition_number, 1) << " | "
        << (e.shrinkage > 0.0 ? num(e.shrinkage, 4) : std::string("-"))
        << " |\n";
    }
    m << "\n";
  }

  const auto& a = rep.attribution;
  m << "## Volatility attribution\n\n";
  m << "| Asset | Weight | Marginal | Component | % of risk |\n";
  m << "|---|---:|---:|---:|---:|\n";
  for (Eigen::Index i = 0; i < a.weights.size(); ++i) {
    m << "| " << a.names[static_cast<std::size_t>(i)] << " | "
      << pct(a.weights(i), 1) << " | " << num(a.marginal(i)) << " | "
      << num(a.component(i)) << " | " << pct(a.percent(i), 1) << " |\n";
  }
  m << "| **Total** |  |  | **" << num(a.component.sum()) << "** | **"
    << pct(a.percent.sum(), 1) << "** |\n\n";

  if (!a.sector_percent.empty()) {
    m << "| Asset class | % of risk |\n|---|---:|\n";
    for (const auto& [s, p] : a.sector_percent) {
      m << "| " << s << " | " << pct(p, 1) << " |\n";
    }
    m << "\n";
  }
  m << "Effective number of bets " << num(a.concentration.effective_num_bets, 2)
    << " across " << rep.assets.size() << " holdings; largest contributor "
    << a.concentration.max_contributor << " at "
    << pct(a.concentration.max_contribution, 1) << ".\n\n";

  for (const auto& d : rep.var_decomposition) {
    m << "## VaR decomposition at " << pct(d.confidence, 0) << "\n\n";
    m << "Component VaR sums to total VaR exactly, by Euler's theorem. The "
         "residual is printed so the identity is checked rather than "
         "assumed.\n\n";
    m << "| Asset | Marginal VaR | Component VaR | Component ($) | % of VaR "
         "|\n";
    m << "|---|---:|---:|---:|---:|\n";
    for (Eigen::Index i = 0; i < d.weights.size(); ++i) {
      m << "| " << d.names[static_cast<std::size_t>(i)] << " | "
        << num(d.marginal(i)) << " | " << num(d.component(i)) << " | "
        << money(d.component(i) * rep.notional) << " | " << pct(d.percent(i), 1)
        << " |\n";
    }
    m << "| **Total** |  | **" << num(d.var) << "** | **"
      << money(d.var * rep.notional) << "** | **100.0%** |\n\n";
    m << "Euler residual: " << num(d.euler_residual, 18) << "\n\n";
  }

  if (!rep.incremental.empty()) {
    m << "## Incremental VaR on a hypothetical trade\n\n";
    m << "Full revaluation at 99% over one day, against the marginal-VaR "
         "linearisation a desk would do in its head.\n\n";
    m << "| Trade | VaR before | VaR after | Incremental | Linear estimate | "
         "Linearisation error |\n";
    m << "|---|---:|---:|---:|---:|---:|\n";
    for (const auto& t : rep.incremental) {
      const double err = t.linear_estimate - t.incremental;
      m << "| " << (t.trade_notional >= 0 ? "+" : "") << money(t.trade_notional)
        << " " << t.asset << " | " << money(t.var_before) << " | "
        << money(t.var_after) << " | " << money(t.incremental) << " | "
        << money(t.linear_estimate) << " | " << money(err) << " |\n";
    }
    m << "\n";
  }

  if (rep.has_factor_decomposition) {
    const auto& d = rep.factor_decomposition;
    m << "## Factor risk decomposition\n\n";
    m << "| Factor | Exposure | Variance contribution | % of model variance "
         "|\n";
    m << "|---|---:|---:|---:|\n";
    for (std::size_t k = 0; k < d.factors.size(); ++k) {
      const auto i = static_cast<Eigen::Index>(k);
      m << "| " << d.factors[k] << " | " << num(d.exposures(i), 4) << " | "
        << num(d.variance_contribution(i), 10) << " | "
        << pct(d.percent_of_total(i), 1) << " |\n";
    }
    m << "| specific |  | " << num(d.specific_variance, 10) << " | "
      << pct(d.specific_variance / d.model_variance, 1) << " |\n";
    m << "| **total** |  | **" << num(d.model_variance, 10)
      << "** | **100.0%** |\n\n";
    m << "Factor model volatility " << pct(d.model_vol, 3)
      << " daily against a sample volatility of "
      << pct(std::sqrt(std::max(0.0, d.sample_variance)), 3) << ". The gap, "
      << pct(d.model_error / d.sample_variance, 1)
      << " of sample variance, is what the factor model does not capture.\n\n";
  }

  if (!rep.stress.empty()) {
    m << "## Stress tests\n\n";
    m << "| Scenario | P&L | P&L ($) |\n|---|---:|---:|\n";
    for (const auto& s : rep.stress) {
      m << "| " << s.name << " | " << pct(s.pct_pnl) << " | "
        << money(s.dollar_pnl) << " |\n";
    }
    m << "\n";
  }

  if (!rep.reverse_stress.empty()) {
    m << "## Reverse stress\n\n";
    m << "The most likely factor move reaching each loss, found by minimising "
         "Mahalanobis distance subject to the loss constraint. The distance "
         "column is the answer: how many standard deviations of the book's own "
         "factor distribution the scenario sits at.\n\n";
    m << "| Target loss | Target ($) | Distance (sigma) | Gaussian prob. | "
         "Implied factor move |\n";
    m << "|---:|---:|---:|---:|---|\n";
    for (const auto& s : rep.reverse_stress) {
      std::ostringstream moves;
      bool first = true;
      for (std::size_t k = 0; k < s.factors.size(); ++k) {
        const double v = s.factor_move(static_cast<Eigen::Index>(k));
        if (std::abs(v) < 1e-9) continue;
        if (!first) moves << ", ";
        const bool is_yield = s.factors[k].starts_with("rates") ||
                              s.factors[k].starts_with("credit");
        moves << s.factors[k] << " " << (v >= 0 ? "+" : "")
              << num(v * (is_yield ? 1e4 : 100.0), is_yield ? 0 : 2)
              << (is_yield ? "bp" : "%");
        first = false;
      }
      m << "| " << pct(s.target_loss_pct, 1) << " | "
        << money(s.target_loss_dollar) << " | "
        << num(s.mahalanobis_distance, 2) << " | "
        << num(s.gaussian_probability, 8) << " | " << moves.str() << " |\n";
    }
    m << "\n";
  }

  m << "## Figures\n\n";
  struct Fig {
    const char* file;
    const char* caption;
  };
  const std::vector<Fig> figs = {
      {"correlation.svg",
       "Correlation matrix, reordered by hierarchical clustering."},
      {"weight_vs_risk.svg",
       "Capital share against risk share, sorted by the gap."},
      {"risk_contributions.svg",
       "Component contribution to portfolio volatility."},
      {"return_distribution.svg",
       "Return distribution against the fitted normal, with VaR and CVaR."},
      {"asset_volatility.svg",
       "Annualized volatility per instrument with bootstrap intervals."},
      {"var_backtest.svg",
       "VaR exceptions with Kupiec, Christoffersen and Basel results."},
      {"factor_risk.svg", "Variance split across factors and specific risk."},
      {"estimator_error.svg",
       "Estimator error against a known covariance, by sample size."},
  };
  for (const auto& f : figs) {
    m << "**" << f.caption << "**\n\n![" << f.caption << "](figures/" << f.file
      << ")\n\n";
  }

  return m.str();
}

void write_reports(const RiskReport& rep, const std::string& dir) {
  std::ofstream js(dir + "/report.json");
  if (!js) {
    throw std::invalid_argument("write_reports: cannot write report.json");
  }
  js << to_json(rep) << "\n";

  std::ofstream md(dir + "/report.md");
  if (!md) {
    throw std::invalid_argument("write_reports: cannot write report.md");
  }
  md << to_markdown(rep);
}

void write_all_figures(const RiskReport& rep, const std::string& dir,
                       const std::string& estimator_csv) {
  auto dump = [&](const std::string& file, const std::string& content) {
    std::ofstream out(dir + "/" + file);
    if (!out) {
      throw std::invalid_argument("write_all_figures: cannot write " + file);
    }
    out << content;
  };

  std::ostringstream src;
  src << "Source: FRED daily series, as of "
      << (rep.as_of.empty() ? "unstamped" : rep.as_of) << ". "
      << rep.portfolio_returns.size()
      << " daily observations. Generated by compute_risk at commit "
      << (rep.git_commit.empty() ? "unstamped" : rep.git_commit) << ".";
  const std::string source = src.str();

  const double conf =
      rep.var_cvar.empty() ? 0.95 : rep.var_cvar.front().confidence;
  const double fig_var = historical_var(rep.portfolio_returns, conf, 1);
  const double fig_cvar = historical_cvar(rep.portfolio_returns, conf, 1);

  std::vector<std::string> names;
  std::vector<std::string> sectors;
  std::vector<double> vols;
  std::vector<double> ci_lo;
  std::vector<double> ci_hi;
  std::vector<double> weights;
  std::vector<double> pct_risk;
  for (const auto& s : rep.asset_stats) {
    names.push_back(s.name);
    sectors.push_back(s.sector);
    vols.push_back(s.annual_vol);
    ci_lo.push_back(s.annual_vol_ci_lower);
    ci_hi.push_back(s.annual_vol_ci_upper);
    weights.push_back(s.weight);
    pct_risk.push_back(s.pct_risk);
  }

  dump("correlation.svg",
       svg_correlation_heatmap(rep.correlation, rep.assets, source));
  dump("risk_contributions.svg",
       svg_risk_contribution_bars(rep.attribution, source));
  dump("return_distribution.svg",
       svg_return_histogram(rep.portfolio_returns, fig_var, fig_cvar, conf,
                            source));
  if (!rep.asset_stats.empty()) {
    dump("asset_volatility.svg",
         svg_asset_volatility(names, vols, ci_lo, ci_hi, sectors, source));
    dump("weight_vs_risk.svg",
         svg_weight_vs_risk(names, weights, pct_risk, source));
  }

  // The backtest figure uses the historical VaR at the first confidence
  // level, which is the row the rest of the report leads with.
  for (const auto& b : rep.backtests) {
    if (b.method == "Historical" && std::abs(b.confidence - conf) < 1e-12) {
      dump("var_backtest.svg",
           svg_var_backtest(rep.portfolio_returns, b.var_level, b.confidence,
                            b.kupiec, b.christoffersen, b.basel, source));
      break;
    }
  }

  if (rep.has_factor_decomposition) {
    dump("factor_risk.svg",
         svg_factor_decomposition(rep.factor_decomposition, source));
  }

  // The estimator figure is driven by the separate study executable. When its
  // CSV is absent this one figure is skipped and the rest still render, so a
  // fresh clone that has not run the study still gets a report.
  if (estimator_csv.empty()) return;
  std::ifstream in(estimator_csv);
  if (!in) return;

  std::string line;
  if (!std::getline(in, line)) return;
  std::vector<std::string> header;
  {
    std::stringstream hs(line);
    std::string f;
    while (std::getline(hs, f, ',')) header.push_back(f);
  }
  auto column_of = [&](const std::string& want) -> std::size_t {
    for (std::size_t i = 0; i < header.size(); ++i) {
      if (header[i] == want) return i;
    }
    throw std::invalid_argument("estimator CSV missing column " + want);
  };
  const std::size_t c_est = column_of("estimator");
  const std::size_t c_n = column_of("sample_size");
  const std::size_t c_err = column_of("var99_abs_error");
  const std::size_t c_se = column_of("var99_abs_error_se");
  const std::size_t widest = std::max({c_est, c_n, c_err, c_se});

  std::vector<EstimatorErrorPoint> points;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> f;
    std::stringstream ls(line);
    std::string cell;
    while (std::getline(ls, cell, ',')) f.push_back(cell);
    if (f.size() <= widest) continue;
    points.push_back(
        {f[c_est], std::stoi(f[c_n]), std::stod(f[c_err]), std::stod(f[c_se])});
  }
  if (points.empty()) return;

  dump("estimator_error.svg",
       svg_estimator_error(
           points, "mean absolute 99% VaR error",
           "More history helps the sample estimator and does nothing for EWMA",
           "Source: apps/estimator_study.cpp, simulated from a factor model "
           "fitted to the FRED sample. 400 replications per point."));
}

}  // namespace risk
