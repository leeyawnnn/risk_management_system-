#include "risk/reporter.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

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
  std::ostringstream os;
  os << std::fixed << std::setprecision(0) << v;
  return os.str();
}
std::string num(double v, int prec = 4) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << v;
  return os.str();
}
std::vector<std::vector<double>> mat_to_vec(const Eigen::MatrixXd& M) {
  std::vector<std::vector<double>> out(M.rows());
  for (Eigen::Index i = 0; i < M.rows(); ++i) {
    out[static_cast<std::size_t>(i)].resize(M.cols());
    for (Eigen::Index j = 0; j < M.cols(); ++j)
      out[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = M(i, j);
  }
  return out;
}
}  // namespace

RiskReport build_report(const std::string& name, const Portfolio& portfolio,
                        const Eigen::MatrixXd& asset_returns,
                        const Eigen::MatrixXd& cov,
                        const std::string& cov_method,
                        double annualization_factor,
                        const std::vector<double>& confidences,
                        const std::vector<int>& horizons,
                        const std::vector<StressResult>& stress,
                        const std::map<std::string, std::string>& sectors,
                        int mc_draws, std::uint64_t seed, double ewma_lambda) {
  RiskReport rep;
  rep.portfolio_name = name;
  rep.notional = portfolio.notional();
  rep.assets = portfolio.names();
  rep.weights = portfolio.weights();
  rep.covariance_method = cov_method;
  rep.annualization_factor = annualization_factor;
  rep.covariance = cov;
  rep.correlation = cov_to_correlation(cov);
  rep.daily_vol = portfolio.volatility(cov);
  rep.annual_vol = rep.daily_vol * std::sqrt(annualization_factor);
  rep.portfolio_returns = portfolio.return_series(asset_returns);
  rep.stress = stress;

  const double port_mean = rep.portfolio_returns.mean();
  const double port_sigma = rep.daily_vol;  // from the chosen covariance
  const Eigen::VectorXd asset_mean = asset_returns.colwise().mean().transpose();
  const Eigen::VectorXd w = portfolio.weights();

  for (double conf : confidences) {
    for (int h : horizons) {
      rep.var_cvar.push_back(
          {"Historical", conf, h, historical_var(rep.portfolio_returns, conf, h),
           historical_cvar(rep.portfolio_returns, conf, h)});
      rep.var_cvar.push_back(
          {"Parametric", conf, h, parametric_var(port_mean, port_sigma, conf, h),
           parametric_cvar(port_mean, port_sigma, conf, h)});
      rep.var_cvar.push_back(
          {"MonteCarlo", conf, h,
           monte_carlo_var(asset_mean, cov, w, conf, h, mc_draws, seed),
           monte_carlo_cvar(asset_mean, cov, w, conf, h, mc_draws, seed)});
    }
  }

  rep.attribution = compute_attribution(portfolio, cov, sectors);

  // ---- per-asset statistics (for comparison) -------------------------------
  const Eigen::Index T = asset_returns.rows();
  const Eigen::Index N = asset_returns.cols();
  const double port_sd = std::sqrt(
      (rep.portfolio_returns.array() - rep.portfolio_returns.mean()).square().sum() /
      static_cast<double>(T - 1));
  for (Eigen::Index a = 0; a < N; ++a) {
    const Eigen::VectorXd col = asset_returns.col(a);
    const double mu = col.mean();
    const double sd = std::sqrt((col.array() - mu).square().sum() /
                                static_cast<double>(T - 1));
    AssetStat st;
    st.name = rep.assets[static_cast<std::size_t>(a)];
    auto sit = sectors.find(st.name);
    st.sector = (sit != sectors.end()) ? sit->second : "";
    st.weight = w(a);
    st.annual_vol = sd * std::sqrt(annualization_factor);
    st.annual_return = mu * annualization_factor;
    st.sharpe = (st.annual_vol > 0.0) ? st.annual_return / st.annual_vol : 0.0;
    st.pct_risk = rep.attribution.percent(a);
    // Correlation of this asset with the portfolio return series.
    const double cov_ap =
        ((col.array() - mu) *
         (rep.portfolio_returns.array() - rep.portfolio_returns.mean()))
            .sum() /
        static_cast<double>(T - 1);
    st.corr_to_portfolio = (sd > 0.0 && port_sd > 0.0)
                               ? cov_ap / (sd * port_sd)
                               : 0.0;
    rep.asset_stats.push_back(st);
  }

  // ---- covariance-estimator comparison -------------------------------------
  auto add_estimator = [&](const std::string& label, const Eigen::MatrixXd& C) {
    EstimatorStat es;
    es.method = label;
    es.daily_vol = portfolio.volatility(C);
    es.annual_vol = es.daily_vol * std::sqrt(annualization_factor);
    es.var95 = parametric_var(port_mean, es.daily_vol, 0.95, 1);
    es.var99 = parametric_var(port_mean, es.daily_vol, 0.99, 1);
    rep.estimator_comparison.push_back(es);
  };
  add_estimator("Sample", sample_covariance(asset_returns));
  add_estimator("EWMA", ewma_covariance(asset_returns, ewma_lambda));
  add_estimator("Ledoit-Wolf", ledoit_wolf_covariance(asset_returns).cov);

  // ---- in-sample VaR backtest ----------------------------------------------
  for (double conf : confidences) {
    BacktestStat bt;
    bt.confidence = conf;
    bt.observations = T;
    bt.expected_rate = 1.0 - conf;
    bt.var_level = historical_var(rep.portfolio_returns, conf, 1);
    long exc = 0;
    for (Eigen::Index t = 0; t < T; ++t)
      if (rep.portfolio_returns(t) < -bt.var_level) ++exc;
    bt.exceedances = exc;
    bt.actual_rate = static_cast<double>(exc) / static_cast<double>(T);
    rep.backtests.push_back(bt);
  }

  return rep;
}

std::string to_json(const RiskReport& rep, int indent) {
  using nlohmann::json;
  json j;
  j["portfolio_name"] = rep.portfolio_name;
  j["notional"] = rep.notional;
  j["covariance_method"] = rep.covariance_method;
  j["annualization_factor"] = rep.annualization_factor;

  json positions = json::array();
  for (Eigen::Index i = 0; i < rep.weights.size(); ++i) {
    positions.push_back({{"asset", rep.assets[static_cast<std::size_t>(i)]},
                         {"weight", rep.weights(i)}});
  }
  j["positions"] = positions;

  j["volatility"] = {{"daily", rep.daily_vol}, {"annual", rep.annual_vol}};
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
    for (const auto& [s, p] : a.sector_percent)
      sectors[s] = {{"percent", p}, {"component", a.sector_component.at(s)}};
    attr["sectors"] = sectors;
  }
  j["attribution"] = attr;

  json st = json::array();
  for (const auto& s : rep.stress) {
    st.push_back({{"name", s.name},
                  {"pct_pnl", s.pct_pnl},
                  {"dollar_pnl", s.dollar_pnl}});
  }
  j["stress_tests"] = st;

  json as = json::array();
  for (const auto& a : rep.asset_stats) {
    as.push_back({{"asset", a.name},
                  {"sector", a.sector},
                  {"weight", a.weight},
                  {"annual_vol", a.annual_vol},
                  {"annual_return", a.annual_return},
                  {"sharpe", a.sharpe},
                  {"corr_to_portfolio", a.corr_to_portfolio},
                  {"pct_risk", a.pct_risk}});
  }
  j["asset_stats"] = as;

  json ec = json::array();
  for (const auto& e : rep.estimator_comparison) {
    ec.push_back({{"method", e.method},
                  {"daily_vol", e.daily_vol},
                  {"annual_vol", e.annual_vol},
                  {"var95", e.var95},
                  {"var99", e.var99}});
  }
  j["estimator_comparison"] = ec;

  json bt = json::array();
  for (const auto& b : rep.backtests) {
    bt.push_back({{"confidence", b.confidence},
                  {"observations", b.observations},
                  {"expected_rate", b.expected_rate},
                  {"exceedances", b.exceedances},
                  {"actual_rate", b.actual_rate},
                  {"var_level", b.var_level}});
  }
  j["var_backtest"] = bt;

  return j.dump(indent);
}

std::string to_markdown(const RiskReport& rep) {
  std::ostringstream m;
  m << "# Risk Report: " << rep.portfolio_name << "\n\n";
  m << "- **Notional:** " << money(rep.notional) << "\n";
  m << "- **Covariance estimator:** " << rep.covariance_method << "\n";
  m << "- **Annualization factor:** " << rep.annualization_factor << "\n";
  m << "- **Portfolio volatility:** " << pct(rep.daily_vol) << " daily / "
    << pct(rep.annual_vol) << " annual\n\n";

  // Positions.
  m << "## Positions\n\n";
  m << "| Asset | Weight |\n|---|---:|\n";
  for (Eigen::Index i = 0; i < rep.weights.size(); ++i)
    m << "| " << rep.assets[static_cast<std::size_t>(i)] << " | "
      << pct(rep.weights(i)) << " |\n";
  m << "\n";

  // Per-asset comparison table.
  if (!rep.asset_stats.empty()) {
    m << "## Asset comparison\n\n";
    m << "Annualized statistics per instrument, and how much portfolio risk "
         "each one actually contributes. Note where **% of risk diverges from "
         "weight** — that gap is the whole point of risk attribution.\n\n";
    m << "| Asset | Sector | Weight | Ann. vol | Ann. return | Sharpe | "
         "Corr. to ptf | % of risk |\n";
    m << "|---|---|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& a : rep.asset_stats) {
      m << "| " << a.name << " | " << a.sector << " | " << pct(a.weight) << " | "
        << pct(a.annual_vol) << " | " << pct(a.annual_return) << " | "
        << num(a.sharpe, 2) << " | " << num(a.corr_to_portfolio, 2) << " | "
        << pct(a.pct_risk) << " |\n";
    }
    m << "\n";
  }

  // VaR / CVaR.
  m << "## Value at Risk & Expected Shortfall\n\n";
  m << "| Method | Confidence | Horizon | VaR | VaR ($) | CVaR | CVaR ($) |\n";
  m << "|---|---:|---:|---:|---:|---:|---:|\n";
  for (const auto& r : rep.var_cvar) {
    m << "| " << r.method << " | " << pct(r.confidence, 0) << " | "
      << r.horizon_days << "d | " << pct(r.var) << " | "
      << money(r.var * rep.notional) << " | " << pct(r.cvar) << " | "
      << money(r.cvar * rep.notional) << " |\n";
  }
  m << "\n";

  // Estimator comparison.
  if (!rep.estimator_comparison.empty()) {
    m << "## Covariance estimator comparison\n\n";
    m << "The same portfolio, priced with three different covariance estimators."
         " Divergence here is *model risk*: the VaR number you report depends on"
         " how you estimated `Sigma`.\n\n";
    m << "| Estimator | Daily vol | Annual vol | 95% 1d VaR | 99% 1d VaR |\n";
    m << "|---|---:|---:|---:|---:|\n";
    for (const auto& e : rep.estimator_comparison)
      m << "| " << e.method << " | " << pct(e.daily_vol) << " | "
        << pct(e.annual_vol) << " | " << pct(e.var95) << " | " << pct(e.var99)
        << " |\n";
    m << "\n";
  }

  // VaR backtest.
  if (!rep.backtests.empty()) {
    m << "## VaR backtest (in-sample)\n\n";
    m << "How often realized daily returns actually breached the historical "
         "VaR. A well-calibrated model breaches at roughly `(1 - confidence)` "
         "of days; far more means VaR is too optimistic.\n\n";
    m << "| Confidence | Obs | Expected breaches | Actual breaches | "
         "Actual rate | Verdict |\n";
    m << "|---:|---:|---:|---:|---:|---|\n";
    for (const auto& b : rep.backtests) {
      const double expected = b.expected_rate * static_cast<double>(b.observations);
      const char* verdict =
          (b.actual_rate <= b.expected_rate * 1.5) ? "OK" : "too optimistic";
      m << "| " << pct(b.confidence, 0) << " | " << b.observations << " | "
        << num(expected, 1) << " | " << b.exceedances << " | "
        << pct(b.actual_rate) << " | " << verdict << " |\n";
    }
    m << "\n";
  }

  // Risk attribution.
  const auto& a = rep.attribution;
  m << "## Risk attribution\n\n";
  m << "| Asset | Weight | Marginal | Component | % of risk |\n";
  m << "|---|---:|---:|---:|---:|\n";
  for (Eigen::Index i = 0; i < a.weights.size(); ++i)
    m << "| " << a.names[static_cast<std::size_t>(i)] << " | "
      << pct(a.weights(i)) << " | " << num(a.marginal(i)) << " | "
      << num(a.component(i)) << " | " << pct(a.percent(i)) << " |\n";
  m << "| **Total** |  |  | **" << num(a.component.sum()) << "** | **"
    << pct(a.percent.sum()) << "** |\n\n";

  if (!a.sector_percent.empty()) {
    m << "### By sector\n\n| Sector | % of risk |\n|---|---:|\n";
    for (const auto& [s, p] : a.sector_percent)
      m << "| " << s << " | " << pct(p) << " |\n";
    m << "\n";
  }

  m << "**Concentration:** effective number of bets = "
    << num(a.concentration.effective_num_bets, 2)
    << "; largest contributor = " << a.concentration.max_contributor << " ("
    << pct(a.concentration.max_contribution) << ").\n\n";

  // Stress tests.
  if (!rep.stress.empty()) {
    m << "## Stress tests\n\n| Scenario | P&L | P&L ($) |\n|---|---:|---:|\n";
    for (const auto& s : rep.stress)
      m << "| " << s.name << " | " << pct(s.pct_pnl) << " | "
        << money(s.dollar_pnl) << " |\n";
    m << "\n";
  }

  // Figures (written alongside by write_all_figures).
  m << "## Figures\n\n";
  m << "**Asset volatility comparison** — annualized vol of every instrument, "
       "coloured by sector.\n\n";
  m << "![Asset volatility](figures/asset_volatility.svg)\n\n";
  m << "**Weight vs. risk share** — capital allocation against actual risk "
       "contribution. Where the red (risk) bar dwarfs the grey (weight) bar, a "
       "small position is punching above its weight.\n\n";
  m << "![Weight vs risk](figures/weight_vs_risk.svg)\n\n";
  m << "**Correlation matrix** — red positive, blue negative; the off-diagonal "
       "structure is what makes the book more (or less) than the sum of its "
       "parts.\n\n";
  m << "![Correlation matrix](figures/correlation.svg)\n\n";
  m << "**Risk contribution by position** — each bar is a share of total "
       "portfolio volatility (bars sum to 100%).\n\n";
  m << "![Risk contributions](figures/risk_contributions.svg)\n\n";
  m << "**Return distribution** — daily portfolio returns with the VaR and CVaR "
       "thresholds marked.\n\n";
  m << "![Return distribution](figures/return_distribution.svg)\n\n";
  m << "**Estimator comparison** — 95% / 99% VaR under each covariance "
       "estimator (model risk at a glance).\n\n";
  m << "![Estimator comparison](figures/estimator_comparison.svg)\n\n";
  m << "**VaR backtest** — daily returns over time against the 95% VaR line; "
       "red dots are breaches.\n\n";
  m << "![VaR backtest](figures/var_backtest.svg)\n\n";

  // Detailed commentary.
  m << "## Commentary — what to look at\n\n";
  if (!rep.stress.empty()) {
    const StressResult* worst = &rep.stress.front();
    for (const auto& s : rep.stress)
      if (s.dollar_pnl < worst->dollar_pnl) worst = &s;
    m << "- **Tail vs. routine risk.** Daily 99% VaR is on the order of a single"
         " bad session, but the worst stress scenario, **" << worst->name
      << "**, costs " << pct(worst->pct_pnl) << " (" << money(worst->dollar_pnl)
      << ") — an order of magnitude larger. VaR and stress answer different "
         "questions; report both.\n";
  }
  m << "- **Concentration.** The largest single risk contributor is **"
    << a.concentration.max_contributor << "** at "
    << pct(a.concentration.max_contribution)
    << " of portfolio risk; the effective number of bets is "
    << num(a.concentration.effective_num_bets, 1) << " across "
    << rep.assets.size() << " holdings.\n";
  // Highlight the biggest weight-vs-risk divergence.
  if (!rep.asset_stats.empty()) {
    const AssetStat* div = &rep.asset_stats.front();
    double best = -1.0;
    for (const auto& s : rep.asset_stats) {
      const double gap = std::abs(s.pct_risk - s.weight);
      if (gap > best) { best = gap; div = &s; }
    }
    m << "- **Weight != risk.** **" << div->name << "** is " << pct(div->weight)
      << " of capital but " << pct(div->pct_risk)
      << " of risk — the clearest case where position size misleads.\n";
  }
  // Diversifiers (negative correlation to the book).
  for (const auto& s : rep.asset_stats) {
    if (s.corr_to_portfolio < 0.0) {
      m << "- **Diversifier.** **" << s.name << "** is negatively correlated to "
           "the book (" << num(s.corr_to_portfolio, 2)
        << "), so it *reduces* total risk despite its own "
        << pct(s.annual_vol) << " volatility.\n";
      break;
    }
  }
  // Estimator spread = model risk.
  if (rep.estimator_comparison.size() >= 2) {
    double lo = rep.estimator_comparison.front().var95;
    double hi = lo;
    for (const auto& e : rep.estimator_comparison) {
      lo = std::min(lo, e.var95);
      hi = std::max(hi, e.var95);
    }
    m << "- **Model risk.** The 95% VaR ranges from " << pct(lo) << " to "
      << pct(hi) << " across covariance estimators — the same book, a "
      << num((hi / lo - 1.0) * 100.0, 0) << "% spread purely from estimation "
         "choice.\n";
  }
  return m.str();
}

void write_reports(const RiskReport& rep, const std::string& dir) {
  std::ofstream js(dir + "/report.json");
  if (!js) throw std::invalid_argument("write_reports: cannot write report.json");
  js << to_json(rep);

  std::ofstream md(dir + "/report.md");
  if (!md) throw std::invalid_argument("write_reports: cannot write report.md");
  md << to_markdown(rep);
}

void write_all_figures(const RiskReport& rep, const std::string& dir) {
  auto dump = [&](const std::string& file, const std::string& content) {
    std::ofstream out(dir + "/" + file);
    if (!out) throw std::invalid_argument("write_all_figures: cannot write " + file);
    out << content;
  };

  // Distribution markers: historical VaR/CVaR at the first requested confidence.
  const double conf = rep.var_cvar.empty() ? 0.95 : rep.var_cvar.front().confidence;
  const double fig_var = historical_var(rep.portfolio_returns, conf, 1);
  const double fig_cvar = historical_cvar(rep.portfolio_returns, conf, 1);

  // Pull per-asset vectors out of the report.
  std::vector<std::string> names, sectors, methods;
  std::vector<double> vols, weights, pct_risk, var95, var99;
  for (const auto& a : rep.asset_stats) {
    names.push_back(a.name);
    sectors.push_back(a.sector);
    vols.push_back(a.annual_vol);
    weights.push_back(a.weight);
    pct_risk.push_back(a.pct_risk);
  }
  for (const auto& e : rep.estimator_comparison) {
    methods.push_back(e.method);
    var95.push_back(e.var95);
    var99.push_back(e.var99);
  }

  dump("correlation.svg", svg_correlation_heatmap(rep.correlation, rep.assets));
  dump("risk_contributions.svg", svg_risk_contribution_bars(rep.attribution));
  dump("return_distribution.svg",
       svg_return_histogram(rep.portfolio_returns, fig_var, fig_cvar, conf));
  if (!rep.asset_stats.empty()) {
    dump("asset_volatility.svg", svg_asset_volatility(names, vols, sectors));
    dump("weight_vs_risk.svg", svg_weight_vs_risk(names, weights, pct_risk));
  }
  if (!rep.estimator_comparison.empty())
    dump("estimator_comparison.svg",
         svg_estimator_var_comparison(methods, var95, var99));
  dump("var_backtest.svg", svg_var_backtest(rep.portfolio_returns, fig_var, conf));
}

}  // namespace risk
