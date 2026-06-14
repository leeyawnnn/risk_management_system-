#include "risk/figures.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace risk {

namespace {

std::string fmt(double v, int prec = 2) {
  std::ostringstream os;
  os.precision(prec);
  os << std::fixed << v;
  return os.str();
}

// Escape the handful of characters that matter inside SVG text nodes.
std::string esc(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      default: out += c;
    }
  }
  return out;
}

// Diverging blue-white-red colour for a correlation in [-1, 1].
std::string corr_color(double r) {
  r = std::clamp(r, -1.0, 1.0);
  int red, green, blue;
  if (r >= 0.0) {  // white -> red
    red = 255;
    green = static_cast<int>(255 * (1.0 - r));
    blue = static_cast<int>(255 * (1.0 - r));
  } else {  // white -> blue
    red = static_cast<int>(255 * (1.0 + r));
    green = static_cast<int>(255 * (1.0 + r));
    blue = 255;
  }
  std::ostringstream os;
  os << "rgb(" << red << "," << green << "," << blue << ")";
  return os.str();
}

}  // namespace

std::string svg_correlation_heatmap(const Eigen::MatrixXd& correlation,
                                    const std::vector<std::string>& names) {
  const int n = static_cast<int>(correlation.rows());
  const int cell = 70;
  const int margin = 90;
  const int w = margin + n * cell + 20;
  const int h = margin + n * cell + 20;

  std::ostringstream s;
  s << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\""
    << h << "\" font-family=\"sans-serif\">\n";
  s << "<rect width=\"" << w << "\" height=\"" << h << "\" fill=\"white\"/>\n";
  s << "<text x=\"" << margin << "\" y=\"30\" font-size=\"18\" font-weight=\"bold\">"
    << "Correlation matrix</text>\n";

  for (int i = 0; i < n; ++i) {
    // Row label (left) and column label (top).
    const int y = margin + i * cell;
    const int x = margin + i * cell;
    s << "<text x=\"" << (margin - 10) << "\" y=\"" << (y + cell / 2 + 4)
      << "\" font-size=\"13\" text-anchor=\"end\">" << esc(names[i])
      << "</text>\n";
    s << "<text x=\"" << (x + cell / 2) << "\" y=\"" << (margin - 10)
      << "\" font-size=\"13\" text-anchor=\"middle\">" << esc(names[i])
      << "</text>\n";
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const double r = correlation(i, j);
      const int x = margin + j * cell;
      const int y = margin + i * cell;
      s << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << cell
        << "\" height=\"" << cell << "\" fill=\"" << corr_color(r)
        << "\" stroke=\"#888\"/>\n";
      s << "<text x=\"" << (x + cell / 2) << "\" y=\"" << (y + cell / 2 + 4)
        << "\" font-size=\"12\" text-anchor=\"middle\">" << fmt(r, 2)
        << "</text>\n";
    }
  }
  s << "</svg>\n";
  return s.str();
}

std::string svg_risk_contribution_bars(const RiskAttribution& a) {
  const int n = static_cast<int>(a.percent.size());
  const int row = 40;
  const int margin_left = 90;
  const int margin_top = 60;
  const int plot_w = 360;
  const int w = margin_left + plot_w + 80;
  const int h = margin_top + n * row + 40;
  const int zero_x = margin_left + plot_w / 4;  // leave room for negative bars

  // Scale: max absolute percentage sets the full bar length.
  double maxabs = 0.0;
  for (int i = 0; i < n; ++i) maxabs = std::max(maxabs, std::abs(a.percent(i)));
  if (maxabs <= 0.0) maxabs = 1.0;
  const double scale = (plot_w * 0.7) / maxabs;

  std::ostringstream s;
  s << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\""
    << h << "\" font-family=\"sans-serif\">\n";
  s << "<rect width=\"" << w << "\" height=\"" << h << "\" fill=\"white\"/>\n";
  s << "<text x=\"20\" y=\"30\" font-size=\"18\" font-weight=\"bold\">"
    << "Percentage risk contribution by position</text>\n";
  s << "<line x1=\"" << zero_x << "\" y1=\"" << (margin_top - 10) << "\" x2=\""
    << zero_x << "\" y2=\"" << (margin_top + n * row) << "\" stroke=\"#333\"/>\n";

  for (int i = 0; i < n; ++i) {
    const double pct = a.percent(i);
    const int y = margin_top + i * row;
    const int len = static_cast<int>(std::abs(pct) * scale);
    const int x = (pct >= 0.0) ? zero_x : zero_x - len;
    const char* fill = (pct >= 0.0) ? "#d62728" : "#2ca02c";
    s << "<rect x=\"" << x << "\" y=\"" << (y + 6) << "\" width=\"" << len
      << "\" height=\"" << (row - 16) << "\" fill=\"" << fill << "\"/>\n";
    s << "<text x=\"" << (margin_left - 12) << "\" y=\"" << (y + row / 2 + 2)
      << "\" font-size=\"13\" text-anchor=\"end\">"
      << esc(a.names[static_cast<std::size_t>(i)]) << "</text>\n";
    const int lx = (pct >= 0.0) ? (zero_x + len + 6) : (zero_x - len - 6);
    const char* anchor = (pct >= 0.0) ? "start" : "end";
    s << "<text x=\"" << lx << "\" y=\"" << (y + row / 2 + 2)
      << "\" font-size=\"12\" text-anchor=\"" << anchor << "\">"
      << fmt(pct * 100.0, 1) << "%</text>\n";
  }
  s << "</svg>\n";
  return s.str();
}

std::string svg_return_histogram(const Eigen::VectorXd& r, double var,
                                 double cvar, double confidence, int bins) {
  if (r.size() == 0) throw std::invalid_argument("svg_return_histogram: empty");
  const double lo = r.minCoeff();
  const double hi = r.maxCoeff();
  const double span = (hi > lo) ? (hi - lo) : 1.0;
  const double bw = span / bins;

  std::vector<int> counts(bins, 0);
  for (Eigen::Index k = 0; k < r.size(); ++k) {
    int b = static_cast<int>((r(k) - lo) / bw);
    b = std::clamp(b, 0, bins - 1);
    counts[static_cast<std::size_t>(b)]++;
  }
  const int maxc = std::max(1, *std::max_element(counts.begin(), counts.end()));

  const int margin = 60;
  const int plot_w = 600;
  const int plot_h = 280;
  const int w = margin + plot_w + 30;
  const int h = margin + plot_h + 70;

  auto x_of = [&](double val) {
    return margin + (val - lo) / span * plot_w;
  };

  std::ostringstream s;
  s << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\""
    << h << "\" font-family=\"sans-serif\">\n";
  s << "<rect width=\"" << w << "\" height=\"" << h << "\" fill=\"white\"/>\n";
  s << "<text x=\"" << margin << "\" y=\"30\" font-size=\"18\" font-weight=\"bold\">"
    << "Portfolio return distribution (" << fmt(confidence * 100, 0)
    << "% VaR / CVaR)</text>\n";

  // Bars.
  for (int b = 0; b < bins; ++b) {
    const double x0 = margin + static_cast<double>(b) / bins * plot_w;
    const double bar_h = static_cast<double>(counts[static_cast<std::size_t>(b)]) /
                         maxc * plot_h;
    s << "<rect x=\"" << fmt(x0, 1) << "\" y=\"" << fmt(margin + plot_h - bar_h, 1)
      << "\" width=\"" << fmt(plot_w / static_cast<double>(bins) - 1.0, 1)
      << "\" height=\"" << fmt(bar_h, 1) << "\" fill=\"#9ecae1\"/>\n";
  }

  // Axis baseline.
  s << "<line x1=\"" << margin << "\" y1=\"" << (margin + plot_h) << "\" x2=\""
    << (margin + plot_w) << "\" y2=\"" << (margin + plot_h)
    << "\" stroke=\"#333\"/>\n";

  // VaR / CVaR markers at returns -var and -cvar. Labels are staggered
  // vertically (and anchored away from the line) so they don't collide when the
  // two levels sit close together.
  auto vline = [&](double val, const char* color, const std::string& label,
                   int label_y, const char* anchor, double dx) {
    const double x = x_of(val);
    s << "<line x1=\"" << fmt(x, 1) << "\" y1=\"" << margin << "\" x2=\""
      << fmt(x, 1) << "\" y2=\"" << (margin + plot_h) << "\" stroke=\"" << color
      << "\" stroke-width=\"2\" stroke-dasharray=\"5,3\"/>\n";
    s << "<text x=\"" << fmt(x + dx, 1) << "\" y=\"" << label_y
      << "\" font-size=\"12\" fill=\"" << color << "\" text-anchor=\"" << anchor
      << "\">" << esc(label) << "</text>\n";
  };
  // CVaR is the deeper (more negative) level, so its line is to the left: label
  // it to the left; label VaR to the right, one line lower.
  vline(-cvar, "#7f0000", "CVaR " + fmt(cvar * 100, 2) + "%", margin + 14,
        "end", -5);
  vline(-var, "#d62728", "VaR " + fmt(var * 100, 2) + "%", margin + 32, "start",
        5);

  // x-axis end labels.
  s << "<text x=\"" << margin << "\" y=\"" << (margin + plot_h + 20)
    << "\" font-size=\"11\" text-anchor=\"middle\">" << fmt(lo * 100, 1)
    << "%</text>\n";
  s << "<text x=\"" << (margin + plot_w) << "\" y=\"" << (margin + plot_h + 20)
    << "\" font-size=\"11\" text-anchor=\"middle\">" << fmt(hi * 100, 1)
    << "%</text>\n";
  s << "</svg>\n";
  return s.str();
}

namespace {
// A small fixed palette keyed by sector label.
std::string sector_color(const std::string& sector) {
  static const std::vector<std::pair<std::string, std::string>> palette = {
      {"Equity", "#1f77b4"},  {"Rates", "#2ca02c"},
      {"Credit", "#ff7f0e"},  {"Commodity", "#9467bd"}};
  for (const auto& [k, v] : palette)
    if (k == sector) return v;
  return "#7f7f7f";
}
}  // namespace

std::string svg_asset_volatility(const std::vector<std::string>& names,
                                 const std::vector<double>& annual_vol,
                                 const std::vector<std::string>& sectors) {
  const int n = static_cast<int>(names.size());
  // Sort indices by descending volatility.
  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
  std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return annual_vol[static_cast<std::size_t>(a)] >
                                       annual_vol[static_cast<std::size_t>(b)]; });

  const int row = 26;
  const int margin_left = 70;
  const int margin_top = 60;
  const int plot_w = 460;
  const int w = margin_left + plot_w + 70;
  const int h = margin_top + n * row + 60;

  double maxv = 0.0;
  for (double v : annual_vol) maxv = std::max(maxv, v);
  if (maxv <= 0.0) maxv = 1.0;
  const double scale = plot_w / maxv;

  std::ostringstream s;
  s << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\""
    << h << "\" font-family=\"sans-serif\">\n";
  s << "<rect width=\"" << w << "\" height=\"" << h << "\" fill=\"white\"/>\n";
  s << "<text x=\"20\" y=\"30\" font-size=\"18\" font-weight=\"bold\">"
    << "Annualized volatility by instrument</text>\n";

  for (int r = 0; r < n; ++r) {
    const int i = idx[static_cast<std::size_t>(r)];
    const double v = annual_vol[static_cast<std::size_t>(i)];
    const int y = margin_top + r * row;
    const int len = static_cast<int>(v * scale);
    s << "<rect x=\"" << margin_left << "\" y=\"" << (y + 4) << "\" width=\""
      << len << "\" height=\"" << (row - 8) << "\" fill=\""
      << sector_color(sectors[static_cast<std::size_t>(i)]) << "\"/>\n";
    s << "<text x=\"" << (margin_left - 8) << "\" y=\"" << (y + row / 2 + 3)
      << "\" font-size=\"12\" text-anchor=\"end\">"
      << esc(names[static_cast<std::size_t>(i)]) << "</text>\n";
    s << "<text x=\"" << (margin_left + len + 6) << "\" y=\"" << (y + row / 2 + 3)
      << "\" font-size=\"11\">" << fmt(v * 100, 1) << "%</text>\n";
  }

  // Legend.
  const std::vector<std::string> secs = {"Equity", "Rates", "Credit", "Commodity"};
  int lx = margin_left;
  const int ly = margin_top + n * row + 22;
  for (const auto& sec : secs) {
    s << "<rect x=\"" << lx << "\" y=\"" << (ly - 10) << "\" width=\"12\" "
      << "height=\"12\" fill=\"" << sector_color(sec) << "\"/>\n";
    s << "<text x=\"" << (lx + 16) << "\" y=\"" << ly << "\" font-size=\"11\">"
      << sec << "</text>\n";
    lx += 110;
  }
  s << "</svg>\n";
  return s.str();
}

std::string svg_weight_vs_risk(const std::vector<std::string>& names,
                               const std::vector<double>& weights,
                               const std::vector<double>& pct_risk) {
  const int n = static_cast<int>(names.size());
  const int row = 34;
  const int margin_left = 70;
  const int margin_top = 70;
  const int plot_w = 380;
  const int w = margin_left + plot_w + 190;
  const int h = margin_top + n * row + 30;

  double maxv = 0.0;
  for (int i = 0; i < n; ++i)
    maxv = std::max({maxv, weights[static_cast<std::size_t>(i)],
                     pct_risk[static_cast<std::size_t>(i)]});
  if (maxv <= 0.0) maxv = 1.0;
  const double scale = plot_w / maxv;

  std::ostringstream s;
  s << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\""
    << h << "\" font-family=\"sans-serif\">\n";
  s << "<rect width=\"" << w << "\" height=\"" << h << "\" fill=\"white\"/>\n";
  s << "<text x=\"20\" y=\"30\" font-size=\"18\" font-weight=\"bold\">"
    << "Capital weight vs. risk share</text>\n";
  // Legend.
  s << "<rect x=\"70\" y=\"42\" width=\"12\" height=\"12\" fill=\"#bbbbbb\"/>"
    << "<text x=\"88\" y=\"52\" font-size=\"11\">Weight</text>\n";
  s << "<rect x=\"160\" y=\"42\" width=\"12\" height=\"12\" fill=\"#d62728\"/>"
    << "<text x=\"178\" y=\"52\" font-size=\"11\">% of risk</text>\n";

  for (int i = 0; i < n; ++i) {
    const int y = margin_top + i * row;
    const int wlen = static_cast<int>(weights[static_cast<std::size_t>(i)] * scale);
    const int rlen = static_cast<int>(pct_risk[static_cast<std::size_t>(i)] * scale);
    s << "<rect x=\"" << margin_left << "\" y=\"" << (y + 3) << "\" width=\""
      << wlen << "\" height=\"11\" fill=\"#bbbbbb\"/>\n";
    s << "<rect x=\"" << margin_left << "\" y=\"" << (y + 16) << "\" width=\""
      << rlen << "\" height=\"11\" fill=\"#d62728\"/>\n";
    s << "<text x=\"" << (margin_left - 8) << "\" y=\"" << (y + 17)
      << "\" font-size=\"12\" text-anchor=\"end\">"
      << esc(names[static_cast<std::size_t>(i)]) << "</text>\n";
    s << "<text x=\"" << (margin_left + std::max(wlen, rlen) + 6) << "\" y=\""
      << (y + 17) << "\" font-size=\"10\" fill=\"#555\">"
      << fmt(pct_risk[static_cast<std::size_t>(i)] * 100, 0) << "% risk / "
      << fmt(weights[static_cast<std::size_t>(i)] * 100, 0) << "% wt</text>\n";
  }
  s << "</svg>\n";
  return s.str();
}

std::string svg_estimator_var_comparison(
    const std::vector<std::string>& methods, const std::vector<double>& var95,
    const std::vector<double>& var99) {
  const int g = static_cast<int>(methods.size());
  const int margin = 60;
  const int plot_h = 240;
  const int group_w = 130;
  const int w = margin + g * group_w + 40;
  const int h = margin + plot_h + 70;

  double maxv = 0.0;
  for (double v : var99) maxv = std::max(maxv, v);
  for (double v : var95) maxv = std::max(maxv, v);
  if (maxv <= 0.0) maxv = 1.0;
  const double scale = plot_h / (maxv * 1.15);

  std::ostringstream s;
  s << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\""
    << h << "\" font-family=\"sans-serif\">\n";
  s << "<rect width=\"" << w << "\" height=\"" << h << "\" fill=\"white\"/>\n";
  s << "<text x=\"20\" y=\"30\" font-size=\"18\" font-weight=\"bold\">"
    << "1-day VaR by covariance estimator</text>\n";
  // Legend.
  s << "<rect x=\"60\" y=\"40\" width=\"12\" height=\"12\" fill=\"#6baed6\"/>"
    << "<text x=\"76\" y=\"50\" font-size=\"11\">95%</text>\n";
  s << "<rect x=\"130\" y=\"40\" width=\"12\" height=\"12\" fill=\"#08519c\"/>"
    << "<text x=\"146\" y=\"50\" font-size=\"11\">99%</text>\n";

  const int base = margin + plot_h;
  s << "<line x1=\"" << margin << "\" y1=\"" << base << "\" x2=\""
    << (margin + g * group_w) << "\" y2=\"" << base << "\" stroke=\"#333\"/>\n";

  for (int i = 0; i < g; ++i) {
    const int gx = margin + i * group_w + 20;
    const int h95 = static_cast<int>(var95[static_cast<std::size_t>(i)] * scale);
    const int h99 = static_cast<int>(var99[static_cast<std::size_t>(i)] * scale);
    s << "<rect x=\"" << gx << "\" y=\"" << (base - h95) << "\" width=\"36\" "
      << "height=\"" << h95 << "\" fill=\"#6baed6\"/>\n";
    s << "<rect x=\"" << (gx + 42) << "\" y=\"" << (base - h99) << "\" width=\"36\" "
      << "height=\"" << h99 << "\" fill=\"#08519c\"/>\n";
    s << "<text x=\"" << (gx + 18) << "\" y=\"" << (base - h95 - 4)
      << "\" font-size=\"10\" text-anchor=\"middle\">"
      << fmt(var95[static_cast<std::size_t>(i)] * 100, 2) << "%</text>\n";
    s << "<text x=\"" << (gx + 60) << "\" y=\"" << (base - h99 - 4)
      << "\" font-size=\"10\" text-anchor=\"middle\">"
      << fmt(var99[static_cast<std::size_t>(i)] * 100, 2) << "%</text>\n";
    s << "<text x=\"" << (gx + 39) << "\" y=\"" << (base + 18)
      << "\" font-size=\"12\" text-anchor=\"middle\">"
      << esc(methods[static_cast<std::size_t>(i)]) << "</text>\n";
  }
  s << "</svg>\n";
  return s.str();
}

std::string svg_var_backtest(const Eigen::VectorXd& r, double var_level,
                             double confidence) {
  if (r.size() == 0) throw std::invalid_argument("svg_var_backtest: empty");
  const int T = static_cast<int>(r.size());
  const int margin = 55;
  const int plot_w = 760;
  const int plot_h = 240;
  const int w = margin + plot_w + 20;
  const int h = margin + plot_h + 50;

  double maxabs = var_level;
  for (Eigen::Index i = 0; i < r.size(); ++i)
    maxabs = std::max(maxabs, std::abs(r(i)));
  if (maxabs <= 0.0) maxabs = 1.0;
  const int mid = margin + plot_h / 2;
  const double yscale = (plot_h / 2.0) / (maxabs * 1.05);
  auto x_of = [&](int t) { return margin + static_cast<double>(t) / (T - 1) * plot_w; };
  auto y_of = [&](double v) { return mid - v * yscale; };

  std::ostringstream s;
  s << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\""
    << h << "\" font-family=\"sans-serif\">\n";
  s << "<rect width=\"" << w << "\" height=\"" << h << "\" fill=\"white\"/>\n";
  s << "<text x=\"20\" y=\"30\" font-size=\"18\" font-weight=\"bold\">"
    << "VaR backtest: daily returns vs " << fmt(confidence * 100, 0)
    << "% VaR</text>\n";

  // Zero line and -VaR line.
  s << "<line x1=\"" << margin << "\" y1=\"" << mid << "\" x2=\""
    << (margin + plot_w) << "\" y2=\"" << mid << "\" stroke=\"#ccc\"/>\n";
  const double yvar = y_of(-var_level);
  s << "<line x1=\"" << margin << "\" y1=\"" << fmt(yvar, 1) << "\" x2=\""
    << (margin + plot_w) << "\" y2=\"" << fmt(yvar, 1)
    << "\" stroke=\"#d62728\" stroke-width=\"1.5\" stroke-dasharray=\"6,3\"/>\n";
  s << "<text x=\"" << (margin + 4) << "\" y=\"" << fmt(yvar - 4, 1)
    << "\" font-size=\"11\" fill=\"#d62728\">-VaR " << fmt(var_level * 100, 2)
    << "%</text>\n";

  // Returns as thin vertical bars from zero; breaches in red.
  int breaches = 0;
  for (int t = 0; t < T; ++t) {
    const double v = r(t);
    const double x = x_of(t);
    const bool breach = v < -var_level;
    if (breach) ++breaches;
    const char* col = breach ? "#d62728" : "#9ecae1";
    s << "<line x1=\"" << fmt(x, 1) << "\" y1=\"" << mid << "\" x2=\"" << fmt(x, 1)
      << "\" y2=\"" << fmt(y_of(v), 1) << "\" stroke=\"" << col << "\" stroke-width=\"1\"/>\n";
  }
  s << "<text x=\"" << margin << "\" y=\"" << (margin + plot_h + 35)
    << "\" font-size=\"12\">" << breaches << " breaches in " << T
    << " days (expected " << fmt((1.0 - confidence) * T, 0)
    << " at " << fmt(confidence * 100, 0) << "% confidence)</text>\n";
  s << "</svg>\n";
  return s.str();
}

void write_figures(const Eigen::MatrixXd& correlation,
                   const std::vector<std::string>& names,
                   const RiskAttribution& attribution,
                   const Eigen::VectorXd& portfolio_returns, double var,
                   double cvar, double confidence, const std::string& dir) {
  auto dump = [&](const std::string& file, const std::string& content) {
    std::ofstream out(dir + "/" + file);
    if (!out) throw std::invalid_argument("write_figures: cannot write " + file);
    out << content;
  };
  dump("correlation.svg", svg_correlation_heatmap(correlation, names));
  dump("risk_contributions.svg", svg_risk_contribution_bars(attribution));
  dump("return_distribution.svg",
       svg_return_histogram(portfolio_returns, var, cvar, confidence));
}

}  // namespace risk
