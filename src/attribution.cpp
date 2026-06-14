#include "risk/attribution.hpp"

namespace risk {

RiskAttribution compute_attribution(
    const Portfolio& portfolio, const Eigen::MatrixXd& cov,
    const std::map<std::string, std::string>& sectors) {
  RiskAttribution a;
  a.names = portfolio.names();
  a.weights = portfolio.weights();
  a.marginal = portfolio.marginal_contributions(cov);
  a.component = portfolio.component_contributions(cov);
  a.percent = portfolio.percent_contributions(cov);
  a.portfolio_vol = portfolio.volatility(cov);

  // Concentration: Herfindahl of risk shares, and the effective number of bets
  // (1/HHI). For n equal, independent positions each pct = 1/n, HHI = 1/n and
  // ENB = n. A single dominant risk source pushes ENB toward 1.
  const double hhi = a.percent.squaredNorm();  // sum pct_i^2
  a.concentration.herfindahl = hhi;
  a.concentration.effective_num_bets = (hhi > 0.0) ? 1.0 / hhi : 0.0;

  Eigen::Index argmax = 0;
  if (a.percent.size() > 0) {
    a.percent.maxCoeff(&argmax);
    a.concentration.max_contribution = a.percent(argmax);
    a.concentration.max_contributor = a.names[static_cast<std::size_t>(argmax)];
  }

  // Sector roll-up.
  if (!sectors.empty()) {
    for (Eigen::Index i = 0; i < a.weights.size(); ++i) {
      const std::string& name = a.names[static_cast<std::size_t>(i)];
      auto it = sectors.find(name);
      const std::string sector =
          (it != sectors.end()) ? it->second : std::string("Unclassified");
      a.sector_component[sector] += a.component(i);
      a.sector_percent[sector] += a.percent(i);
    }
  }
  return a;
}

}  // namespace risk
