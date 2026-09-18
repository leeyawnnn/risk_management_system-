#include "risk/figures.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "risk/plot_style.hpp"
#include "risk/var.hpp"

namespace risk {

using plot::Figure;
using plot::Scale;

namespace {

std::string pct1(double v) {
  return plot::percent(v, 1);
}
std::string pct2(double v) {
  return plot::percent(v, 2);
}

// Colour per sector, assigned on first sight so an instrument keeps the same
// colour in every figure of a report.
std::string sector_color(const std::string& sector,
                         std::map<std::string, std::size_t>& assigned) {
  auto it = assigned.find(sector);
  if (it == assigned.end()) {
    const std::size_t next = assigned.size();
    assigned[sector] = next;
    return plot::categorical_color(next);
  }
  return plot::categorical_color(it->second);
}

constexpr const char* kPassColor = "#009e73";
constexpr const char* kFailColor = "#d55e00";

}  // namespace

// ---------------------------------------------------------------------------
// Hierarchical clustering of the correlation matrix
// ---------------------------------------------------------------------------

std::vector<std::size_t> correlation_cluster_order(
    const Eigen::MatrixXd& correlation) {
  const auto n = static_cast<std::size_t>(correlation.rows());
  if (n == 0) return {};
  if (n == 1) return {0};

  // Average-linkage agglomerative clustering on d(i,j) = 1 - rho(i,j), which
  // runs from 0 for a perfectly correlated pair to 2 for a perfectly
  // anticorrelated one. Ordering the axes this way lets the asset-class
  // blocks appear on their own instead of being pointed out in a caption.
  std::vector<std::vector<std::size_t>> clusters;
  clusters.reserve(n);
  for (std::size_t i = 0; i < n; ++i) clusters.push_back({i});

  auto average_distance = [&](const std::vector<std::size_t>& a,
                              const std::vector<std::size_t>& b) {
    double total = 0.0;
    for (std::size_t const i : a) {
      for (std::size_t const j : b) {
        total += 1.0 - correlation(static_cast<Eigen::Index>(i),
                                   static_cast<Eigen::Index>(j));
      }
    }
    return total / static_cast<double>(a.size() * b.size());
  };

  while (clusters.size() > 1) {
    double best = std::numeric_limits<double>::infinity();
    std::size_t bi = 0;
    std::size_t bj = 1;
    for (std::size_t i = 0; i < clusters.size(); ++i) {
      for (std::size_t j = i + 1; j < clusters.size(); ++j) {
        const double d = average_distance(clusters[i], clusters[j]);
        if (d < best) {
          best = d;
          bi = i;
          bj = j;
        }
      }
    }
    // Merge j into i, preserving within-cluster order, then drop j.
    clusters[bi].insert(clusters[bi].end(), clusters[bj].begin(),
                        clusters[bj].end());
    clusters.erase(clusters.begin() + static_cast<std::ptrdiff_t>(bj));
  }
  return clusters.front();
}

std::string svg_correlation_heatmap(const Eigen::MatrixXd& correlation,
                                    const std::vector<std::string>& names,
                                    const std::string& source) {
  const auto n = static_cast<int>(correlation.rows());
  const std::vector<std::size_t> order = correlation_cluster_order(correlation);

  const int cell = 44;
  const double grid_left = 138.0;
  const double grid_top = 100.0;
  const int width = static_cast<int>(grid_left) + n * cell + 190;
  const int height = static_cast<int>(grid_top) + n * cell + 96;

  Figure fig("Correlations cluster into rates, credit, equity and FX blocks",
             "Daily log-return correlations. Rows and columns ordered by "
             "average-linkage clustering on 1 - rho; lower triangle only.",
             width, height);

  for (int r = 0; r < n; ++r) {
    const auto i =
        static_cast<Eigen::Index>(order[static_cast<std::size_t>(r)]);
    const double y = grid_top + r * cell;

    fig.text(grid_left - 10.0, y + cell / 2.0 + 3.5,
             names[static_cast<std::size_t>(i)], plot::kTickSize, plot::kInk,
             "end");

    for (int c = 0; c <= r; ++c) {
      const auto j =
          static_cast<Eigen::Index>(order[static_cast<std::size_t>(c)]);
      const double x = grid_left + c * cell;
      const double rho = correlation(i, j);

      fig.rect(x, y, cell, cell, plot::diverging_color(rho, 0.0, 1.0),
               "#ffffff", 1.0);
      // White text on the saturated ends, dark ink in the pale middle, so the
      // annotation never disappears into its own cell.
      const char* ink = (std::abs(rho) > 0.62) ? "#ffffff" : plot::kInk;
      fig.text(x + cell / 2.0, y + cell / 2.0 + 3.0, plot::fixed(rho, 2),
               plot::kAnnotationSize, ink, "middle");
    }

    // Column label below the diagonal cell, rotated to stay readable.
    const double dx = grid_left + r * cell + cell / 2.0;
    std::ostringstream os;
    os << R"SVG(<text transform="translate()SVG" << plot::fixed(dx, 2) << ","
       << plot::fixed(grid_top + n * cell + 12.0, 2)
       << R"SVG() rotate(-55)" font-size=")SVG"
       << plot::fixed(plot::kTickSize, 1) << R"SVG(" fill=")SVG" << plot::kInk
       << R"SVG(" text-anchor="end">)SVG"
       << plot::escape(names[static_cast<std::size_t>(i)]) << "</text>\n";
    fig.raw(os.str());
  }

  // Colour key.
  const double key_x = grid_left + n * cell + 46.0;
  const double key_top = grid_top;
  const double key_h = std::min(240.0, static_cast<double>(n * cell));
  const int steps = 60;
  for (int s = 0; s < steps; ++s) {
    const double t = 1.0 - 2.0 * static_cast<double>(s) / (steps - 1);
    fig.rect(key_x, key_top + key_h * s / steps, 16.0, key_h / steps + 0.6,
             plot::diverging_color(t, 0.0, 1.0));
  }
  fig.text(key_x + 22.0, key_top + 8.0, "+1.0", plot::kAnnotationSize,
           plot::kMutedInk);
  fig.text(key_x + 22.0, key_top + key_h / 2.0 + 3.0, "0.0",
           plot::kAnnotationSize, plot::kMutedInk);
  fig.text(key_x + 22.0, key_top + key_h, "-1.0", plot::kAnnotationSize,
           plot::kMutedInk);
  fig.text(key_x, key_top - 12.0, "correlation", plot::kAnnotationSize,
           plot::kInk);

  return fig.str(source);
}

// ---------------------------------------------------------------------------
// Weight against risk share
// ---------------------------------------------------------------------------

std::string svg_weight_vs_risk(const std::vector<std::string>& names,
                               const std::vector<double>& weights,
                               const std::vector<double>& pct_risk,
                               const std::string& source) {
  const std::size_t n = names.size();
  const int height = static_cast<int>(180 + n * 34);
  Figure fig("Risk share and capital share come apart across the book",
             "Each rule spans a position's share of capital and its share of "
             "portfolio volatility. Sorted by the gap, widest first.",
             plot::kDefaultWidth, height);
  fig.set_margins(150.0, 210.0, 92.0, 70.0);

  // Sorting by (risk - capital) puts the positions punching above their
  // weight at the top without anyone choosing an order by hand.
  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::ranges::sort(idx, [&](std::size_t a, std::size_t b) {
    return (pct_risk[a] - weights[a]) > (pct_risk[b] - weights[b]);
  });

  double lo = 0.0;
  double hi = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    lo = std::min({lo, weights[i], pct_risk[i]});
    hi = std::max({hi, weights[i], pct_risk[i]});
  }
  plot::pad_range(lo, hi, 0.08);

  const Scale x{.domain_lo = lo,
                .domain_hi = hi,
                .range_lo = fig.plot_left(),
                .range_hi = fig.plot_right()};

  for (double const t : plot::nice_ticks(lo, hi, 7)) {
    fig.line(x(t), fig.plot_top(), x(t), fig.plot_bottom(), plot::kGrid, 1.0);
    fig.text(x(t), fig.plot_bottom() + 17.0, pct1(t), plot::kTickSize,
             plot::kMutedInk, "middle");
  }
  if (lo < 0.0 && hi > 0.0) {
    fig.line(x(0.0), fig.plot_top(), x(0.0), fig.plot_bottom(), plot::kAxis,
             1.2);
  }
  fig.text((fig.plot_left() + fig.plot_right()) / 2.0, fig.plot_bottom() + 40.0,
           "share of the book (%)", plot::kAxisLabelSize, plot::kInk, "middle");

  const std::string capital_color = plot::categorical_color(5);
  const std::string risk_color = plot::categorical_color(1);
  const double row = fig.plot_height() / static_cast<double>(n);

  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t i = idx[k];
    const double y = fig.plot_top() + row * (static_cast<double>(k) + 0.5);
    const double xw = x(weights[i]);
    const double xr = x(pct_risk[i]);

    fig.text(fig.plot_left() - 12.0, y + 3.5, names[i], plot::kTickSize,
             plot::kInk, "end");
    fig.line(xw, y, xr, y, plot::kFaintInk, 1.6);
    fig.circle(xw, y, 5.0, capital_color);
    fig.circle(xr, y, 5.0, risk_color);

    const double gap = pct_risk[i] - weights[i];
    const std::string sign = gap >= 0.0 ? "+" : "";
    fig.text(fig.plot_right() + 14.0, y + 3.5,
             sign + plot::fixed(gap * 100.0, 1) + "pp risk over capital",
             plot::kAnnotationSize, gap >= 0.0 ? risk_color : plot::kMutedInk,
             "start");
  }

  // Direct labels on the top row rather than a legend.
  {
    const std::size_t i = idx.front();
    const double y = fig.plot_top() + row * 0.5;
    fig.text(x(weights[i]), y - 13.0, "capital", plot::kAnnotationSize,
             capital_color, "middle");
    fig.text(x(pct_risk[i]), y - 13.0, "risk", plot::kAnnotationSize,
             risk_color, "middle");
  }

  return fig.str(source);
}

// ---------------------------------------------------------------------------
// Risk contribution per position
// ---------------------------------------------------------------------------

std::string svg_risk_contribution_bars(const RiskAttribution& a,
                                       const std::string& source) {
  const auto n = static_cast<std::size_t>(a.percent.size());
  const int height = static_cast<int>(200 + n * 34);
  Figure fig("A handful of positions carry most of the volatility",
             "Component contribution to portfolio volatility, "
             "w_i (Sigma w)_i / sigma_p. Bars sum to 100%.",
             plot::kDefaultWidth, height);
  fig.set_margins(150.0, 210.0, 100.0, 74.0);

  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::ranges::sort(idx, [&](std::size_t p, std::size_t q) {
    return a.percent(static_cast<Eigen::Index>(p)) >
           a.percent(static_cast<Eigen::Index>(q));
  });

  double lo = 0.0;
  double hi = 0.0;
  for (Eigen::Index i = 0; i < a.percent.size(); ++i) {
    lo = std::min(lo, a.percent(i));
    hi = std::max(hi, a.percent(i));
  }
  const double equal = 1.0 / static_cast<double>(n);
  hi = std::max(hi, equal);
  plot::pad_range(lo, hi, 0.10);

  const Scale x{.domain_lo = lo,
                .domain_hi = hi,
                .range_lo = fig.plot_left(),
                .range_hi = fig.plot_right()};
  for (double const t : plot::nice_ticks(lo, hi, 7)) {
    fig.line(x(t), fig.plot_top(), x(t), fig.plot_bottom(), plot::kGrid, 1.0);
    fig.text(x(t), fig.plot_bottom() + 17.0, pct1(t), plot::kTickSize,
             plot::kMutedInk, "middle");
  }
  fig.text((fig.plot_left() + fig.plot_right()) / 2.0, fig.plot_bottom() + 40.0,
           "share of portfolio volatility (%)", plot::kAxisLabelSize,
           plot::kInk, "middle");

  const double zero_x = x(0.0);
  const double row = fig.plot_height() / static_cast<double>(n);
  const std::string add_color = plot::categorical_color(1);
  const std::string cut_color = plot::categorical_color(2);

  for (std::size_t k = 0; k < n; ++k) {
    const auto i = static_cast<Eigen::Index>(idx[k]);
    const double v = a.percent(i);
    const double y = fig.plot_top() + row * static_cast<double>(k);
    const bool reducing = v < 0.0;
    fig.rect(zero_x, y + row * 0.18, x(v) - zero_x, row * 0.64,
             reducing ? cut_color : add_color);
    fig.text(fig.plot_left() - 12.0, y + row * 0.5 + 3.5,
             a.names[static_cast<std::size_t>(i)], plot::kTickSize, plot::kInk,
             "end");
    const double label_x = reducing ? x(v) - 8.0 : x(v) + 8.0;
    fig.text(label_x, y + row * 0.5 + 3.5, pct1(v), plot::kAnnotationSize,
             plot::kMutedInk, reducing ? "end" : "start");
  }

  // Reference line at 1/N: what equal contribution would look like.
  fig.line(x(equal), fig.plot_top() - 6.0, x(equal), fig.plot_bottom(),
           plot::kMutedInk, 1.2, "5 3");
  fig.text(x(equal), fig.plot_top() - 12.0,
           "equal contribution 1/N = " + pct1(equal), plot::kAnnotationSize,
           plot::kMutedInk, "middle");

  // Concentration summary on the chart, not in the caption.
  const double box_x = fig.plot_right() + 16.0;
  fig.text(box_x, fig.plot_top() + 14.0,
           "effective bets " +
               plot::fixed(a.concentration.effective_num_bets, 1) + " of " +
               std::to_string(n),
           plot::kAnnotationSize, plot::kInk);
  fig.text(box_x, fig.plot_top() + 29.0,
           "top: " + a.concentration.max_contributor + " " +
               pct1(a.concentration.max_contribution),
           plot::kAnnotationSize, plot::kMutedInk);
  fig.text(box_x, fig.plot_top() + 44.0,
           "risk-share HHI " + plot::fixed(a.concentration.herfindahl, 3),
           plot::kAnnotationSize, plot::kMutedInk);

  return fig.str(source);
}

// ---------------------------------------------------------------------------
// Return distribution
// ---------------------------------------------------------------------------

std::string svg_return_histogram(const Eigen::VectorXd& portfolio_returns,
                                 double var, double cvar, double confidence,
                                 const std::string& source, int bins) {
  const Eigen::VectorXd& r = portfolio_returns;
  const auto n = static_cast<double>(r.size());
  const double mean = r.mean();
  const double sd = std::sqrt((r.array() - mean).square().sum() / (n - 1.0));

  // The title states what the data says, computed here rather than asserted,
  // so refreshing the sample cannot leave a stale claim on the chart.
  const double m2 = (r.array() - mean).square().sum() / n;
  const double excess_kurtosis =
      (m2 > 0.0) ? (r.array() - mean).pow(4).sum() / n / (m2 * m2) - 3.0 : 0.0;
  const std::string title =
      excess_kurtosis > 0.5
          ? "The loss tail is heavier than a normal of the same variance"
          : "The return distribution is close to normal";
  std::ostringstream sub;
  sub << "Daily portfolio log returns against the fitted normal density. "
      << "Excess kurtosis " << plot::fixed(excess_kurtosis, 2) << " on "
      << static_cast<long>(n) << " observations.";

  Figure fig(title, sub.str(), plot::kDefaultWidth, plot::kDefaultHeight);
  fig.set_margins(92.0, 46.0, 92.0, 82.0);

  double lo = r.minCoeff();
  double hi = r.maxCoeff();
  plot::pad_range(lo, hi, 0.04);

  std::vector<int> counts(static_cast<std::size_t>(bins), 0);
  const double bin_w = (hi - lo) / bins;
  for (Eigen::Index i = 0; i < r.size(); ++i) {
    auto b = static_cast<int>((r(i) - lo) / bin_w);
    b = std::clamp(b, 0, bins - 1);
    ++counts[static_cast<std::size_t>(b)];
  }
  const int max_count = *std::ranges::max_element(counts);

  // A density y-axis, so the histogram and the fitted curve share a scale.
  const double max_density = static_cast<double>(max_count) / (n * bin_w);
  const double y_hi = max_density * 1.12;
  const Scale x{.domain_lo = lo,
                .domain_hi = hi,
                .range_lo = fig.plot_left(),
                .range_hi = fig.plot_right()};
  const Scale y{.domain_lo = 0.0,
                .domain_hi = y_hi,
                .range_lo = fig.plot_bottom(),
                .range_hi = fig.plot_top()};

  fig.y_axis(y, plot::nice_ticks(0.0, y_hi, 6), "density", nullptr);
  fig.x_axis(x, plot::nice_ticks(lo, hi, 9), "daily return (%)", &pct1);

  const std::string bar_color = plot::categorical_color(5);
  const std::string tail_color = plot::categorical_color(1);
  for (int b = 0; b < bins; ++b) {
    const double left = lo + b * bin_w;
    const double density =
        static_cast<double>(counts[static_cast<std::size_t>(b)]) / (n * bin_w);
    if (density <= 0.0) continue;
    // Bins entirely beyond -VaR are the exceptions; colouring them ties this
    // figure to the backtest without needing a second chart.
    const bool in_tail = (left + bin_w) <= -var;
    fig.rect(x(left) + 0.5, y(density), x(left + bin_w) - x(left) - 1.0,
             fig.plot_bottom() - y(density), in_tail ? tail_color : bar_color);
  }

  // Fitted normal density at the same mean and variance.
  std::vector<std::pair<double, double>> curve;
  const int samples = 260;
  curve.reserve(static_cast<std::size_t>(samples));
  for (int i = 0; i < samples; ++i) {
    const double v = lo + (hi - lo) * i / (samples - 1);
    const double d = normal_pdf((v - mean) / sd) / sd;
    curve.emplace_back(x(v), y(d));
  }
  fig.polyline(curve, plot::kInk, 1.8);

  const int conf_pct = static_cast<int>(std::lround(confidence * 100.0));
  // CVaR always sits deeper in the tail than VaR, so its line is to the left.
  // Labelling CVaR leftward and VaR rightward keeps each clear of the other's
  // vertical rule; anchoring both the same way put one label straight through
  // the other's line.
  auto marker = [&](double level, const std::string& label,
                    const std::string& color, double label_dy, bool to_right) {
    const double px = x(-level);
    fig.line(px, fig.plot_top(), px, fig.plot_bottom(), color, 1.6, "6 4");
    fig.text(px + (to_right ? 6.0 : -6.0), fig.plot_top() + label_dy,
             label + " " + pct2(level), plot::kAnnotationSize, color,
             to_right ? "start" : "end");
  };
  marker(var, std::to_string(conf_pct) + "% VaR", plot::categorical_color(0),
         16.0, true);
  marker(cvar, std::to_string(conf_pct) + "% CVaR", plot::categorical_color(3),
         16.0, false);

  fig.text(fig.plot_right() - 4.0, fig.plot_top() + 16.0, "fitted normal",
           plot::kAnnotationSize, plot::kInk, "end");
  fig.text(fig.plot_right() - 4.0, fig.plot_top() + 30.0,
           "shaded bins breach VaR", plot::kAnnotationSize, tail_color, "end");

  return fig.str(source);
}

// ---------------------------------------------------------------------------
// Asset volatility
// ---------------------------------------------------------------------------

std::string svg_asset_volatility(const std::vector<std::string>& names,
                                 const std::vector<double>& annual_vol,
                                 const std::vector<double>& ci_lower,
                                 const std::vector<double>& ci_upper,
                                 const std::vector<std::string>& sectors,
                                 const std::string& source) {
  const std::size_t n = names.size();
  const int height = static_cast<int>(200 + n * 34);

  // State the spread the data actually shows instead of hardcoding a
  // multiple that a data refresh could quietly falsify.
  double vmin = std::numeric_limits<double>::infinity();
  double vmax = 0.0;
  for (double const v : annual_vol) {
    if (v > 0.0) vmin = std::min(vmin, v);
    vmax = std::max(vmax, v);
  }
  const double ratio = (vmin > 0.0 && std::isfinite(vmin)) ? vmax / vmin : 0.0;
  std::ostringstream vol_title;
  vol_title << "Instrument volatility spans a " << plot::fixed(ratio, 0)
            << "-fold range";

  Figure fig(vol_title.str(),
             "Annualized volatility of daily log returns with 95% bootstrap "
             "intervals. Sorted, coloured by asset class.",
             plot::kDefaultWidth, height);
  fig.set_margins(150.0, 130.0, 100.0, 74.0);

  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::ranges::sort(idx, [&](std::size_t a, std::size_t b) {
    return annual_vol[a] > annual_vol[b];
  });

  double hi = 0.0;
  for (std::size_t i = 0; i < n; ++i) hi = std::max(hi, ci_upper[i]);
  double lo = 0.0;
  plot::pad_range(lo, hi, 0.06);
  lo = 0.0;  // a volatility axis starts at zero or it misleads

  const Scale x{.domain_lo = lo,
                .domain_hi = hi,
                .range_lo = fig.plot_left(),
                .range_hi = fig.plot_right()};
  for (double const t : plot::nice_ticks(lo, hi, 7)) {
    fig.line(x(t), fig.plot_top(), x(t), fig.plot_bottom(), plot::kGrid, 1.0);
    fig.text(x(t), fig.plot_bottom() + 17.0, pct1(t), plot::kTickSize,
             plot::kMutedInk, "middle");
  }
  fig.text((fig.plot_left() + fig.plot_right()) / 2.0, fig.plot_bottom() + 40.0,
           "annualized volatility (%)", plot::kAxisLabelSize, plot::kInk,
           "middle");

  std::map<std::string, std::size_t> assigned;
  const double row = fig.plot_height() / static_cast<double>(n);
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t i = idx[k];
    const double y = fig.plot_top() + row * static_cast<double>(k);
    const std::string color = sector_color(sectors[i], assigned);

    fig.rect(x(0.0), y + row * 0.22, x(annual_vol[i]) - x(0.0), row * 0.56,
             color);
    // Error bar with caps. A volatility estimated on ~1,240 days carries real
    // uncertainty, and drawing it is the difference between a number and an
    // estimate.
    const double ye = y + row * 0.5;
    fig.line(x(ci_lower[i]), ye, x(ci_upper[i]), ye, plot::kInk, 1.4);
    fig.line(x(ci_lower[i]), ye - 4.0, x(ci_lower[i]), ye + 4.0, plot::kInk,
             1.4);
    fig.line(x(ci_upper[i]), ye - 4.0, x(ci_upper[i]), ye + 4.0, plot::kInk,
             1.4);

    fig.text(fig.plot_left() - 12.0, ye + 3.5, names[i], plot::kTickSize,
             plot::kInk, "end");
    fig.text(x(ci_upper[i]) + 9.0, ye + 3.5, pct1(annual_vol[i]),
             plot::kAnnotationSize, plot::kMutedInk, "start");
  }

  // Legend: one swatch per asset class along the top.
  double lx = fig.plot_left();
  for (const auto& [sector, slot] : assigned) {
    fig.rect(lx, fig.plot_top() - 24.0, 10.0, 10.0,
             plot::categorical_color(slot));
    fig.text(lx + 15.0, fig.plot_top() - 15.0, sector, plot::kAnnotationSize,
             plot::kMutedInk);
    lx += 26.0 + 6.2 * static_cast<double>(sector.size());
  }

  return fig.str(source);
}

// ---------------------------------------------------------------------------
// Estimator error against sample size
// ---------------------------------------------------------------------------

std::string svg_estimator_error(const std::vector<EstimatorErrorPoint>& points,
                                const std::string& y_label,
                                const std::string& title,
                                const std::string& source) {
  Figure fig(title,
             "Mean error against a known covariance over Monte Carlo "
             "replications, log-log. Bands are +/- 2 standard errors.",
             plot::kDefaultWidth, plot::kDefaultHeight);
  fig.set_margins(104.0, 190.0, 92.0, 82.0);
  if (points.empty()) return fig.str(source);

  std::vector<std::string> estimators;
  std::set<int> sizes;
  for (const auto& p : points) {
    if (std::ranges::find(estimators, p.estimator) == estimators.end()) {
      estimators.push_back(p.estimator);
    }
    sizes.insert(p.sample_size);
  }

  double ylo = std::numeric_limits<double>::infinity();
  double yhi = 0.0;
  for (const auto& p : points) {
    ylo = std::min(ylo, std::max(1e-12, p.error - 2.0 * p.error_se));
    yhi = std::max(yhi, p.error + 2.0 * p.error_se);
  }
  // Log axes: both error and sample size span more than a decade here, and a
  // linear axis would flatten the whole left-hand side into the corner.
  const double log_ylo = std::log10(ylo) - 0.08;
  const double log_yhi = std::log10(yhi) + 0.08;
  const double log_xlo = std::log10(static_cast<double>(*sizes.begin())) - 0.06;
  const double log_xhi =
      std::log10(static_cast<double>(*sizes.rbegin())) + 0.06;

  const Scale x{.domain_lo = log_xlo,
                .domain_hi = log_xhi,
                .range_lo = fig.plot_left(),
                .range_hi = fig.plot_right()};
  const Scale y{.domain_lo = log_ylo,
                .domain_hi = log_yhi,
                .range_lo = fig.plot_bottom(),
                .range_hi = fig.plot_top()};

  // Decade gridlines with minor ticks between them, as a log axis requires.
  // Labels go on the 1, 2 and 5 mantissas rather than decades alone: this
  // data spans barely more than one decade, and labelling only powers of ten
  // leaves the axis with a single number on it.
  for (int e = static_cast<int>(std::floor(log_ylo));
       e <= static_cast<int>(std::ceil(log_yhi)); ++e) {
    for (int m = 1; m <= 9; ++m) {
      const double value = m * std::pow(10.0, e);
      const double v = std::log10(value);
      if (v < log_ylo || v > log_yhi) continue;
      const bool decade = (m == 1);
      const bool labelled = (m == 1 || m == 2 || m == 5);
      fig.line(fig.plot_left(), y(v), fig.plot_right(), y(v), plot::kGrid,
               decade ? 1.0 : 0.5);
      if (labelled) {
        // These are loss fractions, so basis points read far better than
        // either scientific notation or a string of leading zeros.
        fig.text(fig.plot_left() - 9.0, y(v) + 3.5,
                 plot::fixed(value * 1e4, value * 1e4 < 1.0 ? 2 : 1) + "bp",
                 plot::kTickSize, plot::kMutedInk, "end");
      }
    }
  }
  fig.line(fig.plot_left(), fig.plot_top(), fig.plot_left(), fig.plot_bottom(),
           plot::kAxis, 1.0);
  fig.line(fig.plot_left(), fig.plot_bottom(), fig.plot_right(),
           fig.plot_bottom(), plot::kAxis, 1.0);

  for (int const s : sizes) {
    const double px = x(std::log10(static_cast<double>(s)));
    fig.line(px, fig.plot_bottom(), px, fig.plot_bottom() + 4.0, plot::kAxis,
             1.0);
    fig.text(px, fig.plot_bottom() + 17.0, std::to_string(s), plot::kTickSize,
             plot::kMutedInk, "middle");
  }
  fig.text((fig.plot_left() + fig.plot_right()) / 2.0, fig.plot_bottom() + 40.0,
           "sample size, trading days (log scale)", plot::kAxisLabelSize,
           plot::kInk, "middle");
  {
    std::ostringstream os;
    os << R"SVG(<text transform="translate(26,)SVG"
       << plot::fixed((fig.plot_top() + fig.plot_bottom()) / 2.0, 2)
       << R"SVG() rotate(-90)" font-size=")SVG"
       << plot::fixed(plot::kAxisLabelSize, 1) << R"SVG(" fill=")SVG"
       << plot::kInk << R"SVG(" text-anchor="middle">)SVG"
       << plot::escape(y_label + " (log scale)") << "</text>\n";
    fig.raw(os.str());
  }

  std::vector<double> label_ys;
  for (std::size_t e = 0; e < estimators.size(); ++e) {
    const std::string& name = estimators[e];
    const std::string color = plot::categorical_color(e);

    std::vector<std::pair<double, double>> centre;
    std::vector<std::pair<double, double>> upper;
    std::vector<std::pair<double, double>> lower;
    for (int const s : sizes) {
      for (const auto& p : points) {
        if (p.estimator != name || p.sample_size != s) continue;
        const double px = x(std::log10(static_cast<double>(s)));
        centre.emplace_back(px, y(std::log10(std::max(1e-12, p.error))));
        upper.emplace_back(
            px, y(std::log10(std::max(1e-12, p.error + 2.0 * p.error_se))));
        lower.emplace_back(
            px, y(std::log10(std::max(1e-12, p.error - 2.0 * p.error_se))));
      }
    }
    if (centre.empty()) continue;

    // Confidence band as a closed polygon: up the upper edge, back along the
    // lower one.
    std::vector<std::pair<double, double>> band = upper;
    band.insert(band.end(), lower.rbegin(), lower.rend());
    std::ostringstream poly;
    poly << "<polygon points=\"";
    for (const auto& [px, py] : band) {
      poly << plot::fixed(px, 2) << "," << plot::fixed(py, 2) << " ";
    }
    poly << "\" fill=\"" << color
         << "\" fill-opacity=\"0.16\" stroke=\"none\"/>\n";
    fig.raw(poly.str());

    fig.polyline(centre, color, 2.2);
    for (const auto& [px, py] : centre) fig.circle(px, py, 3.4, color);

    // Direct label at the right end instead of a legend. Two estimators that
    // land on top of each other -- which is exactly what Sample and
    // Ledoit-Wolf do here -- would otherwise print one label over the other
    // and read as a single smudge, so nudge each new label clear of the ones
    // already placed.
    double label_y = centre.back().second + 3.5;
    bool moved = false;
    for (double const used : label_ys) {
      if (std::abs(label_y - used) < 12.0) {
        label_y = used + 12.0;
        moved = true;
      }
    }
    label_ys.push_back(label_y);
    if (moved) {
      // A short leader so a displaced label still points at its own line.
      fig.line(centre.back().first + 5.0, centre.back().second,
               centre.back().first + 10.0, label_y - 3.5, color, 0.9);
    }
    fig.text(centre.back().first + 12.0, label_y, name, plot::kAnnotationSize,
             color, "start");
  }

  return fig.str(source);
}

// ---------------------------------------------------------------------------
// Eigenvalue spectrum
// ---------------------------------------------------------------------------

std::string svg_eigenvalue_spectrum(const std::vector<SpectrumPoint>& points,
                                    int sample_size,
                                    const std::string& source) {
  // The headline is what each estimator does to the SMALLEST eigenvalue:
  // that direction sets the condition number and it is the one a short
  // sample gets worst. Computed from the data, not asserted, so the caption
  // cannot outlive the result.
  double worst_ratio = 1.0;
  std::string worst_name;
  for (const auto& p : points) {
    if (p.index != 0 || !(p.truth > 0.0) || !(p.estimated > 0.0)) continue;
    const double ratio = p.estimated / p.truth;
    if (std::abs(std::log(ratio)) > std::abs(std::log(worst_ratio))) {
      worst_ratio = ratio;
      worst_name = p.estimator;
    }
  }
  std::ostringstream title;
  if (!worst_name.empty() && worst_ratio > 1.05) {
    title << worst_name << " lifts the smallest eigenvalue "
          << plot::fixed(worst_ratio, 1) << "x above the truth";
  } else if (!worst_name.empty() && worst_ratio < 0.95) {
    title << worst_name << " pushes the smallest eigenvalue "
          << plot::fixed((1.0 - worst_ratio) * 100.0, 0) << "% below the truth";
  } else {
    title << "Every estimator recovers the spectrum at this sample size";
  }

  std::ostringstream sub;
  sub << "Mean estimated eigenvalue against the truth at n = " << sample_size
      << " observations, log scale. Ordered smallest to largest; the smallest "
         "is the one that sets the condition number.";
  Figure fig(title.str(), sub.str(), plot::kDefaultWidth, plot::kDefaultHeight);
  fig.set_margins(104.0, 190.0, 92.0, 82.0);
  if (points.empty()) return fig.str(source);

  std::vector<std::string> estimators;
  int max_index = 0;
  double lo = std::numeric_limits<double>::infinity();
  double hi = 0.0;
  for (const auto& p : points) {
    if (std::ranges::find(estimators, p.estimator) == estimators.end()) {
      estimators.push_back(p.estimator);
    }
    max_index = std::max(max_index, p.index);
    for (double const v : {p.estimated, p.truth}) {
      if (v > 0.0) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
    }
  }
  if (!(hi > 0.0) || !std::isfinite(lo)) return fig.str(source);

  const double log_lo = std::log10(lo) - 0.15;
  const double log_hi = std::log10(hi) + 0.15;
  const Scale x{.domain_lo = 0.0,
                .domain_hi = static_cast<double>(max_index),
                .range_lo = fig.plot_left(),
                .range_hi = fig.plot_right()};
  const Scale y{.domain_lo = log_lo,
                .domain_hi = log_hi,
                .range_lo = fig.plot_bottom(),
                .range_hi = fig.plot_top()};

  for (int e = static_cast<int>(std::floor(log_lo));
       e <= static_cast<int>(std::ceil(log_hi)); ++e) {
    for (int m = 1; m <= 9; ++m) {
      const double v = std::log10(m * std::pow(10.0, e));
      if (v < log_lo || v > log_hi) continue;
      fig.line(fig.plot_left(), y(v), fig.plot_right(), y(v), plot::kGrid,
               m == 1 ? 1.0 : 0.5);
      if (m == 1) {
        std::ostringstream lab;
        lab << "1e" << e;
        fig.text(fig.plot_left() - 9.0, y(v) + 3.5, lab.str(), plot::kTickSize,
                 plot::kMutedInk, "end");
      }
    }
  }
  fig.line(fig.plot_left(), fig.plot_top(), fig.plot_left(), fig.plot_bottom(),
           plot::kAxis, 1.0);
  fig.line(fig.plot_left(), fig.plot_bottom(), fig.plot_right(),
           fig.plot_bottom(), plot::kAxis, 1.0);
  for (int i = 0; i <= max_index; ++i) {
    fig.text(x(i), fig.plot_bottom() + 17.0, std::to_string(i + 1),
             plot::kTickSize, plot::kMutedInk, "middle");
  }
  fig.text((fig.plot_left() + fig.plot_right()) / 2.0, fig.plot_bottom() + 40.0,
           "eigenvalue rank, smallest to largest", plot::kAxisLabelSize,
           plot::kInk, "middle");
  {
    std::ostringstream os;
    os << R"SVG(<text transform="translate(26,)SVG"
       << plot::fixed((fig.plot_top() + fig.plot_bottom()) / 2.0, 2)
       << R"SVG() rotate(-90)" font-size=")SVG"
       << plot::fixed(plot::kAxisLabelSize, 1) << R"SVG(" fill=")SVG"
       << plot::kInk << R"SVG(" text-anchor="middle">)SVG"
       << "eigenvalue of the daily covariance (log scale)" << "</text>\n";
    fig.raw(os.str());
  }

  // The truth first, as a heavy dark reference line, so every estimator is
  // read as a departure from it rather than as one more series.
  std::vector<std::pair<double, double>> truth;
  for (int i = 0; i <= max_index; ++i) {
    for (const auto& p : points) {
      if (p.index == i && p.truth > 0.0) {
        truth.emplace_back(x(i), y(std::log10(p.truth)));
        break;
      }
    }
  }
  fig.polyline(truth, plot::kInk, 2.6);
  std::vector<double> label_ys;
  if (!truth.empty()) {
    const double ty = truth.back().second + 3.5;
    fig.text(truth.back().first + 12.0, ty, "truth", plot::kAnnotationSize,
             plot::kInk, "start");
    // Seed the collision list so an estimator label cannot land on top of
    // the reference line's label.
    label_ys.push_back(ty);
  }

  for (std::size_t e = 0; e < estimators.size(); ++e) {
    const std::string& name = estimators[e];
    const std::string color = plot::categorical_color(e);
    std::vector<std::pair<double, double>> line;
    for (int i = 0; i <= max_index; ++i) {
      for (const auto& p : points) {
        if (p.estimator == name && p.index == i && p.estimated > 0.0) {
          line.emplace_back(x(i), y(std::log10(p.estimated)));
          break;
        }
      }
    }
    if (line.empty()) continue;
    fig.polyline(line, color, 1.8, "none");
    for (const auto& [px, py] : line) fig.circle(px, py, 3.0, color);

    double label_y = line.back().second + 3.5;
    for (double const used : label_ys) {
      if (std::abs(label_y - used) < 12.0) label_y = used + 12.0;
    }
    label_ys.push_back(label_y);
    fig.text(line.back().first + 12.0, label_y, name, plot::kAnnotationSize,
             color, "start");
  }

  return fig.str(source);
}

// ---------------------------------------------------------------------------
// VaR backtest
// ---------------------------------------------------------------------------

std::string svg_var_backtest(const Eigen::VectorXd& portfolio_returns,
                             double var_level, double confidence,
                             const KupiecResult& kupiec,
                             const ChristoffersenResult& christoffersen,
                             const BaselResult& basel,
                             const std::string& source) {
  const Eigen::VectorXd& r = portfolio_returns;
  const int conf_pct = static_cast<int>(std::lround(confidence * 100.0));

  // The title reports the verdict the tests actually returned. Kupiec passing
  // while Christoffersen rejects is the interesting case and the one this
  // sample produces, but the chart must not claim it when it is not true.
  std::string headline;
  if (!kupiec.reject_at_95 && christoffersen.reject_independence_at_95) {
    headline = "Exceptions arrive at the right rate but cluster in time";
  } else if (kupiec.reject_at_95) {
    headline = "The exception rate is wrong: this VaR is miscalibrated";
  } else {
    headline = "Exceptions arrive at the right rate and show no clustering";
  }

  Figure fig(headline,
             "Daily portfolio returns against the " + std::to_string(conf_pct) +
                 "% VaR line. Red marks breach it; the shaded band is the "
                 "trailing Basel supervisory window.",
             plot::kDefaultWidth, plot::kDefaultHeight);
  fig.set_margins(92.0, 300.0, 92.0, 82.0);

  const auto n = static_cast<double>(r.size());
  double lo = std::min(r.minCoeff(), -var_level);
  double hi = r.maxCoeff();
  plot::pad_range(lo, hi, 0.07);

  const Scale x{.domain_lo = 0.0,
                .domain_hi = n - 1.0,
                .range_lo = fig.plot_left(),
                .range_hi = fig.plot_right()};
  const Scale y{.domain_lo = lo,
                .domain_hi = hi,
                .range_lo = fig.plot_bottom(),
                .range_hi = fig.plot_top()};

  // Basel window as a background band, drawn before the data.
  if (basel.window_complete) {
    const double start = n - static_cast<double>(basel.window);
    const char* zone_fill = "#eef7ee";
    if (basel.zone == BaselZone::Yellow) zone_fill = "#fdf6e3";
    if (basel.zone == BaselZone::Red) zone_fill = "#fdeeee";
    fig.rect(x(start), fig.plot_top(), fig.plot_right() - x(start),
             fig.plot_height(), zone_fill);
    // Bottom of the band, not the top: the statistics box starts at the top
    // right and the two labels collided.
    fig.text(x(start) + 6.0, fig.plot_bottom() - 8.0,
             std::string("Basel 250d: ") + to_string(basel.zone) + ", " +
                 std::to_string(basel.exceptions) + " exc.",
             plot::kAnnotationSize, plot::kMutedInk);
  }

  fig.y_axis(y, plot::nice_ticks(lo, hi, 7), "daily return (%)", &pct1);
  fig.x_axis(x, plot::nice_ticks(0.0, n - 1.0, 8), "trading day", nullptr);

  std::vector<std::pair<double, double>> path;
  path.reserve(static_cast<std::size_t>(r.size()));
  for (Eigen::Index i = 0; i < r.size(); ++i) {
    path.emplace_back(x(static_cast<double>(i)), y(r(i)));
  }
  fig.polyline(path, "#b9c6d2", 0.8);

  fig.line(fig.plot_left(), y(-var_level), fig.plot_right(), y(-var_level),
           plot::categorical_color(0), 1.8, "7 4");
  fig.text(fig.plot_left() + 6.0, y(-var_level) - 6.0,
           "-" + std::to_string(conf_pct) + "% VaR " + pct2(var_level),
           plot::kAnnotationSize, plot::categorical_color(0));

  const std::string breach = plot::categorical_color(1);
  for (Eigen::Index i = 0; i < r.size(); ++i) {
    if (r(i) < -var_level) {
      fig.circle(x(static_cast<double>(i)), y(r(i)), 3.0, breach);
    }
  }

  // The statistics box. This is what the previous version of this figure was
  // missing entirely: a picture of breaches with no test attached proves
  // nothing at all.
  const double bx = fig.plot_right() + 22.0;
  double by = fig.plot_top() + 16.0;
  auto row = [&](const std::string& label, const std::string& value,
                 const char* color = plot::kInk) {
    fig.text(bx, by, label, plot::kAnnotationSize, plot::kMutedInk);
    fig.text(bx + 258.0, by, value, plot::kAnnotationSize, color, "end");
    by += 15.0;
  };

  fig.text(bx, by, "Backtest", plot::kAxisLabelSize, plot::kInk, "start",
           "600");
  by += 21.0;
  row("observations", std::to_string(kupiec.observations));
  row("exceptions", std::to_string(kupiec.exceptions));
  row("expected",
      plot::fixed(
          kupiec.expected_rate * static_cast<double>(kupiec.observations), 1));
  row("observed rate", pct2(kupiec.observed_rate));
  by += 10.0;

  fig.text(bx, by, "Kupiec, unconditional coverage", plot::kAnnotationSize,
           plot::kInk, "start", "600");
  by += 16.0;
  row("LR statistic", plot::fixed(kupiec.lr_statistic, 3));
  row("p-value", plot::fixed(kupiec.p_value, 3));
  row("at 95%", kupiec.reject_at_95 ? "REJECT" : "pass",
      kupiec.reject_at_95 ? kFailColor : kPassColor);
  by += 10.0;

  fig.text(bx, by, "Christoffersen, independence", plot::kAnnotationSize,
           plot::kInk, "start", "600");
  by += 16.0;
  row("LR statistic", plot::fixed(christoffersen.lr_independence, 3));
  row("p-value", plot::fixed(christoffersen.p_value_independence, 3));
  row("at 95%", christoffersen.reject_independence_at_95 ? "REJECT" : "pass",
      christoffersen.reject_independence_at_95 ? kFailColor : kPassColor);
  by += 10.0;

  fig.text(bx, by, "Conditional coverage", plot::kAnnotationSize, plot::kInk,
           "start", "600");
  by += 16.0;
  row("LR statistic", plot::fixed(christoffersen.lr_conditional_coverage, 3));
  row("p-value", plot::fixed(christoffersen.p_value_conditional_coverage, 3));
  row("at 95%",
      christoffersen.reject_conditional_coverage_at_95 ? "REJECT" : "pass",
      christoffersen.reject_conditional_coverage_at_95 ? kFailColor
                                                       : kPassColor);

  return fig.str(source);
}

// ---------------------------------------------------------------------------
// Factor risk decomposition
// ---------------------------------------------------------------------------

std::string svg_factor_decomposition(const FactorDecomposition& d,
                                     const std::string& source) {
  struct Item {
    std::string label;
    double variance = 0.0;
  };
  std::vector<Item> items;
  items.reserve(d.factors.size() + 1);
  for (std::size_t k = 0; k < d.factors.size(); ++k) {
    items.push_back(
        {d.factors[k], d.variance_contribution(static_cast<Eigen::Index>(k))});
  }
  items.push_back({"specific", d.specific_variance});
  std::ranges::sort(items, [](const Item& a, const Item& b) {
    return a.variance > b.variance;
  });

  // Name the dominant factor and its share in the title, computed from the
  // decomposition rather than written down once and left to rot.
  std::ostringstream title;
  if (d.model_variance > 0.0 && !items.empty()) {
    title << items.front().label << " carries "
          << pct1(items.front().variance / d.model_variance)
          << " of the book's variance";
  } else {
    title << "Factor risk decomposition";
  }
  std::ostringstream sub;
  sub << "Portfolio variance split across factors plus specific risk. "
      << "A negative bar is a factor that hedges the rest of the book.";

  Figure fig(title.str(), sub.str(), plot::kDefaultWidth, 540);
  fig.set_margins(150.0, 250.0, 96.0, 78.0);

  if (!(d.model_variance > 0.0)) return fig.str(source);

  double lo = 0.0;
  double hi = 0.0;
  for (const auto& it : items) {
    lo = std::min(lo, it.variance / d.model_variance);
    hi = std::max(hi, it.variance / d.model_variance);
  }
  plot::pad_range(lo, hi, 0.10);

  const Scale x{.domain_lo = lo,
                .domain_hi = hi,
                .range_lo = fig.plot_left(),
                .range_hi = fig.plot_right()};
  for (double const t : plot::nice_ticks(lo, hi, 7)) {
    fig.line(x(t), fig.plot_top(), x(t), fig.plot_bottom(), plot::kGrid, 1.0);
    fig.text(x(t), fig.plot_bottom() + 17.0, pct1(t), plot::kTickSize,
             plot::kMutedInk, "middle");
  }
  fig.text((fig.plot_left() + fig.plot_right()) / 2.0, fig.plot_bottom() + 40.0,
           "share of model variance (%)", plot::kAxisLabelSize, plot::kInk,
           "middle");

  const double row = fig.plot_height() / static_cast<double>(items.size());
  const double zero_x = x(0.0);
  for (std::size_t k = 0; k < items.size(); ++k) {
    const double share = items[k].variance / d.model_variance;
    const double y = fig.plot_top() + row * static_cast<double>(k);
    const bool specific = items[k].label == "specific";
    const std::string color =
        specific ? std::string(plot::kFaintInk) : plot::categorical_color(k);
    fig.rect(zero_x, y + row * 0.2, x(share) - zero_x, row * 0.6, color);
    fig.text(fig.plot_left() - 12.0, y + row * 0.5 + 3.5, items[k].label,
             plot::kTickSize, plot::kInk, "end");
    const bool neg = share < 0.0;
    fig.text(x(share) + (neg ? -8.0 : 8.0), y + row * 0.5 + 3.5, pct1(share),
             plot::kAnnotationSize, plot::kMutedInk, neg ? "end" : "start");
  }

  const double bx = fig.plot_right() + 18.0;
  double by = fig.plot_top() + 14.0;
  auto note = [&](const std::string& s) {
    fig.text(bx, by, s, plot::kAnnotationSize, plot::kMutedInk);
    by += 16.0;
  };
  note("factor vol " +
       plot::percent(std::sqrt(std::max(0.0, d.factor_variance)), 3) +
       " daily");
  note("specific vol " +
       plot::percent(std::sqrt(std::max(0.0, d.specific_variance)), 3));
  note("model vol " + plot::percent(d.model_vol, 3));
  note("sample vol " +
       plot::percent(std::sqrt(std::max(0.0, d.sample_variance)), 3));
  by += 8.0;
  note("variance explained " + pct1(d.variance_explained));
  if (d.sample_variance > 0.0) {
    note("model error " + plot::percent(d.model_error / d.sample_variance, 1) +
         " of sample variance");
  }

  return fig.str(source);
}

}  // namespace risk
