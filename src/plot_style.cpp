#include "risk/plot_style.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <ranges>
#include <sstream>
#include <utility>

namespace risk::plot {

namespace {

std::string rgb(int r, int g, int b) {
  std::ostringstream os;
  os << "rgb(" << std::clamp(r, 0, 255) << "," << std::clamp(g, 0, 255) << ","
     << std::clamp(b, 0, 255) << ")";
  return os.str();
}

// Coordinates are emitted at two decimals. More is noise in an SVG and makes
// the diff between two regenerations unreadable.
std::string coord(double v) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(2) << v;
  std::string s = os.str();
  if (s == "-0.00") return "0.00";
  return s;
}

}  // namespace

const std::vector<std::string>& categorical_palette() {
  // Okabe & Ito's eight-colour qualitative set, minus pure black.
  static const std::vector<std::string> kPalette = {
      "#0072b2",  // blue
      "#d55e00",  // vermillion
      "#009e73",  // bluish green
      "#cc79a7",  // reddish purple
      "#e69f00",  // orange
      "#56b4e9",  // sky blue
      "#f0e442",  // yellow
  };
  return kPalette;
}

std::string categorical_color(std::size_t index) {
  const auto& p = categorical_palette();
  return p[index % p.size()];
}

std::string diverging_color(double value, double centre, double half_range) {
  if (!(half_range > 0.0)) return "#ffffff";
  double t = (value - centre) / half_range;
  t = std::clamp(t, -1.0, 1.0);
  // Blue (#2166ac) through white to red (#b2182b). The midpoint is pinned at
  // `centre`, so the neutral value is exactly white rather than wherever the
  // data happens to average out.
  if (t >= 0.0) {
    const int r = static_cast<int>(std::lround(255 + t * (178 - 255)));
    const int g = static_cast<int>(std::lround(255 + t * (24 - 255)));
    const int b = static_cast<int>(std::lround(255 + t * (43 - 255)));
    return rgb(r, g, b);
  }
  const double u = -t;
  const int r = static_cast<int>(std::lround(255 + u * (33 - 255)));
  const int g = static_cast<int>(std::lround(255 + u * (102 - 255)));
  const int b = static_cast<int>(std::lround(255 + u * (172 - 255)));
  return rgb(r, g, b);
}

std::string sequential_color(double t) {
  t = std::clamp(t, 0.0, 1.0);
  // Five-stop piecewise-linear approximation of viridis. Perceptually uniform
  // enough for a heatmap and far better than any rainbow.
  static constexpr std::array<std::array<double, 3>, 5> stops = {
      {{68, 1, 84},
       {59, 82, 139},
       {33, 145, 140},
       {94, 201, 98},
       {253, 231, 37}}};
  const double scaled = t * 4.0;
  const auto i = static_cast<std::size_t>(std::min(3.0, std::floor(scaled)));
  const double f = scaled - static_cast<double>(i);
  const int r = static_cast<int>(
      std::lround(stops[i][0] + f * (stops[i + 1][0] - stops[i][0])));
  const int g = static_cast<int>(
      std::lround(stops[i][1] + f * (stops[i + 1][1] - stops[i][1])));
  const int b = static_cast<int>(
      std::lround(stops[i][2] + f * (stops[i + 1][2] - stops[i][2])));
  return rgb(r, g, b);
}

std::string escape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char const c : s) {
    switch (c) {
      case '&':
        out += "&amp;";
        break;
      case '<':
        out += "&lt;";
        break;
      case '>':
        out += "&gt;";
        break;
      case '"':
        out += "&quot;";
        break;
      case '\'':
        out += "&apos;";
        break;
      default:
        out += c;
    }
  }
  return out;
}

std::string fixed(double v, int precision) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(precision) << v;
  std::string s = os.str();
  // Avoid printing "-0.0", which looks like a bug to a reader.
  if (s.find_first_not_of("-0.") == std::string::npos && !s.empty() &&
      s[0] == '-') {
    s.erase(0, 1);
  }
  return s;
}

std::string percent(double fraction, int precision) {
  return fixed(fraction * 100.0, precision) + "%";
}

std::string thousands(double v) {
  const bool negative = v < 0.0;
  std::ostringstream os;
  os << std::fixed << std::setprecision(0) << std::abs(v);
  std::string digits = os.str();
  std::string grouped;
  int count = 0;
  for (char const c : std::ranges::reverse_view(digits)) {
    if (count > 0 && count % 3 == 0) grouped.push_back(',');
    grouped.push_back(c);
    ++count;
  }
  std::ranges::reverse(grouped);
  return (negative ? "-" : "") + grouped;
}

std::string compact_money(double v) {
  const bool negative = v < 0.0;
  const double a = std::abs(v);
  std::string body;
  if (a >= 1e9) {
    body = fixed(a / 1e9, 1) + "bn";
  } else if (a >= 1e6) {
    body = fixed(a / 1e6, 1) + "M";
  } else if (a >= 1e3) {
    body = fixed(a / 1e3, 0) + "k";
  } else {
    body = fixed(a, 0);
  }
  return (negative ? "-$" : "$") + body;
}

double Scale::operator()(double v) const {
  const double span = domain_hi - domain_lo;
  if (std::abs(span) < 1e-300) return range_lo;
  return range_lo + (v - domain_lo) / span * (range_hi - range_lo);
}

std::vector<double> nice_ticks(double lo, double hi, int target) {
  std::vector<double> ticks;
  if (!(hi > lo) || target < 2) {
    ticks.push_back(lo);
    return ticks;
  }
  const double raw = (hi - lo) / static_cast<double>(target);
  const double magnitude = std::pow(10.0, std::floor(std::log10(raw)));
  const double normalized = raw / magnitude;
  // Snap the step to 1, 2, 5 or 10 times a power of ten so labels stay round.
  double step = magnitude;
  if (normalized > 5.0) {
    step = 10.0 * magnitude;
  } else if (normalized > 2.0) {
    step = 5.0 * magnitude;
  } else if (normalized > 1.0) {
    step = 2.0 * magnitude;
  }

  // Indexed by an integer rather than accumulating `t += step`: repeated
  // addition drifts, and a tick axis that drifts prints 0.30000000000000004.
  const double start = std::ceil(lo / step) * step;
  const auto count =
      static_cast<long>(std::floor((hi - start) / step + 1e-9)) + 1;
  for (long k = 0; k < count; ++k) {
    const double t = start + static_cast<double>(k) * step;
    const double snapped = (std::abs(t) < step * 1e-9) ? 0.0 : t;
    ticks.push_back(snapped);
  }
  return ticks;
}

void pad_range(double& lo, double& hi, double fraction) {
  const double span = hi - lo;
  if (span <= 0.0) {
    const double bump = (std::abs(hi) > 0.0) ? std::abs(hi) * 0.1 : 1.0;
    lo -= bump;
    hi += bump;
    return;
  }
  lo -= span * fraction;
  hi += span * fraction;
}

Figure::Figure(std::string title, std::string subtitle, int width, int height)
    : width_(width),
      height_(height),
      title_(std::move(title)),
      subtitle_(std::move(subtitle)) {}

void Figure::set_margins(double left, double right, double top, double bottom) {
  left_ = left;
  right_ = right;
  top_ = top;
  bottom_ = bottom;
}

void Figure::raw(const std::string& svg) {
  body_ += svg;
}

void Figure::line(double x1, double y1, double x2, double y2,
                  const std::string& stroke, double stroke_width,
                  const std::string& dash) {
  std::ostringstream os;
  os << "<line x1=\"" << coord(x1) << "\" y1=\"" << coord(y1) << "\" x2=\""
     << coord(x2) << "\" y2=\"" << coord(y2) << "\" stroke=\"" << stroke
     << "\" stroke-width=\"" << fixed(stroke_width, 2) << "\"";
  if (!dash.empty()) os << " stroke-dasharray=\"" << dash << "\"";
  os << "/>\n";
  body_ += os.str();
}

void Figure::rect(double x, double y, double w, double h,
                  const std::string& fill, const std::string& stroke,
                  double stroke_width) {
  if (w < 0.0) {
    x += w;
    w = -w;
  }
  if (h < 0.0) {
    y += h;
    h = -h;
  }
  std::ostringstream os;
  os << "<rect x=\"" << coord(x) << "\" y=\"" << coord(y) << "\" width=\""
     << coord(w) << "\" height=\"" << coord(h) << "\" fill=\"" << fill << "\"";
  if (!stroke.empty()) {
    os << " stroke=\"" << stroke << "\" stroke-width=\""
       << fixed(stroke_width, 2) << "\"";
  }
  os << "/>\n";
  body_ += os.str();
}

void Figure::circle(double cx, double cy, double r, const std::string& fill,
                    const std::string& stroke, double stroke_width) {
  std::ostringstream os;
  os << "<circle cx=\"" << coord(cx) << "\" cy=\"" << coord(cy) << "\" r=\""
     << fixed(r, 2) << "\" fill=\"" << fill << "\"";
  if (!stroke.empty()) {
    os << " stroke=\"" << stroke << "\" stroke-width=\""
       << fixed(stroke_width, 2) << "\"";
  }
  os << "/>\n";
  body_ += os.str();
}

void Figure::polyline(const std::vector<std::pair<double, double>>& points,
                      const std::string& stroke, double stroke_width,
                      const std::string& fill) {
  if (points.empty()) return;
  std::ostringstream os;
  os << "<polyline points=\"";
  for (const auto& [x, y] : points) os << coord(x) << "," << coord(y) << " ";
  os << "\" fill=\"" << fill << "\" stroke=\"" << stroke << "\" stroke-width=\""
     << fixed(stroke_width, 2)
     << "\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>\n";
  body_ += os.str();
}

void Figure::text(double x, double y, const std::string& content, double size,
                  const std::string& fill, const std::string& anchor,
                  const std::string& weight, const std::string& style) {
  std::ostringstream os;
  os << "<text x=\"" << coord(x) << "\" y=\"" << coord(y) << "\" font-size=\""
     << fixed(size, 1) << "\" fill=\"" << fill << "\" text-anchor=\"" << anchor
     << "\"";
  if (weight != "normal") os << " font-weight=\"" << weight << "\"";
  if (style != "normal") os << " font-style=\"" << style << "\"";
  os << ">" << escape(content) << "</text>\n";
  body_ += os.str();
}

void Figure::callout(double text_x, double text_y, double target_x,
                     double target_y, const std::vector<std::string>& lines,
                     const std::string& anchor) {
  line(text_x, text_y + 4.0, target_x, target_y, kFaintInk, 1.0, "3 2");
  circle(target_x, target_y, 2.6, kInk);
  double y = text_y;
  for (const auto& l : lines) {
    text(text_x, y, l, kAnnotationSize, kMutedInk, anchor);
    y -= 12.0;
  }
}

void Figure::y_axis(const Scale& y, const std::vector<double>& ticks,
                    const std::string& label,
                    std::string (*formatter)(double)) {
  for (double const t : ticks) {
    const double py = y(t);
    if (py < plot_top() - 0.5 || py > plot_bottom() + 0.5) continue;
    // Horizontal grid only, drawn under the data because axes are emitted
    // before any series.
    line(plot_left(), py, plot_right(), py, kGrid, 1.0);
    const std::string lab = (formatter != nullptr) ? formatter(t) : fixed(t, 2);
    text(plot_left() - 9.0, py + 3.5, lab, kTickSize, kMutedInk, "end");
  }
  // Left spine only: no top or right spine anywhere in this repository.
  line(plot_left(), plot_top(), plot_left(), plot_bottom(), kAxis, 1.0);
  if (!label.empty()) {
    std::ostringstream os;
    const double cx = 22.0;
    const double cy = (plot_top() + plot_bottom()) / 2.0;
    os << R"SVG(<text transform="translate()SVG" << coord(cx) << ","
       << coord(cy) << R"SVG() rotate(-90)" font-size=")SVG"
       << fixed(kAxisLabelSize, 1) << R"SVG(" fill=")SVG" << kInk
       << R"SVG(" text-anchor="middle">)SVG" << escape(label) << "</text>\n";
    body_ += os.str();
  }
}

void Figure::x_axis(const Scale& x, const std::vector<double>& ticks,
                    const std::string& label,
                    std::string (*formatter)(double)) {
  for (double const t : ticks) {
    const double px = x(t);
    if (px < plot_left() - 0.5 || px > plot_right() + 0.5) continue;
    line(px, plot_bottom(), px, plot_bottom() + 4.0, kAxis, 1.0);
    const std::string lab = (formatter != nullptr) ? formatter(t) : fixed(t, 2);
    text(px, plot_bottom() + 17.0, lab, kTickSize, kMutedInk, "middle");
  }
  line(plot_left(), plot_bottom(), plot_right(), plot_bottom(), kAxis, 1.0);
  if (!label.empty()) {
    text((plot_left() + plot_right()) / 2.0, plot_bottom() + 38.0, label,
         kAxisLabelSize, kInk, "middle");
  }
}

void Figure::x_category_axis(const std::vector<double>& centres,
                             const std::vector<std::string>& labels,
                             double rotate_degrees) {
  line(plot_left(), plot_bottom(), plot_right(), plot_bottom(), kAxis, 1.0);
  const std::size_t n = std::min(centres.size(), labels.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double px = centres[i];
    if (std::abs(rotate_degrees) < 1e-9) {
      text(px, plot_bottom() + 17.0, labels[i], kTickSize, kMutedInk, "middle");
    } else {
      std::ostringstream os;
      os << R"SVG(<text transform="translate()SVG" << coord(px) << ","
         << coord(plot_bottom() + 14.0) << ") rotate("
         << fixed(rotate_degrees, 0) << R"SVG()" font-size=")SVG"
         << fixed(kTickSize, 1) << R"SVG(" fill=")SVG" << kMutedInk
         << R"SVG(" text-anchor="end">)SVG" << escape(labels[i]) << "</text>\n";
      body_ += os.str();
    }
  }
}

std::string Figure::str(const std::string& source) const {
  std::ostringstream os;
  os << R"(<svg xmlns="http://www.w3.org/2000/svg" width=")" << width_
     << "\" height=\"" << height_ << "\" viewBox=\"0 0 " << width_ << " "
     << height_ << "\" font-family=\"" << kFontStack << "\">\n";
  // An explicit background: without it the figure is transparent and becomes
  // unreadable against GitHub's dark theme.
  os << "<rect width=\"" << width_ << "\" height=\"" << height_
     << "\" fill=\"#ffffff\"/>\n";

  os << R"(<text x="36" y="34" font-size=")" << fixed(kTitleSize, 1)
     << R"(" font-weight="600" fill=")" << kInk << "\">" << escape(title_)
     << "</text>\n";
  if (!subtitle_.empty()) {
    os << R"(<text x="36" y="53" font-size=")" << fixed(kSubtitleSize, 1)
       << "\" fill=\"" << kMutedInk << "\">" << escape(subtitle_)
       << "</text>\n";
  }

  os << body_;

  os << R"(<text x="36" y=")" << (height_ - 14) << "\" font-size=\""
     << fixed(kSourceSize, 1) << R"(" font-style="italic" fill=")" << kFaintInk
     << "\">" << escape(source) << "</text>\n";
  os << "</svg>\n";
  return os.str();
}

}  // namespace risk::plot
