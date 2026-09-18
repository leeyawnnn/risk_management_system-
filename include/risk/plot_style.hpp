#pragma once

#include <string>
#include <vector>

// One place that owns how every figure in this repository looks.
//
// The figures are emitted as SVG directly from C++ rather than handed to
// matplotlib. That keeps the reporting path dependency-free -- the engine
// produces its own charts with nothing installed beyond a compiler -- but it
// only works if there is a single style module rather than eleven hand-tuned
// chart functions. Everything below is that module: palette, type scale,
// canvas geometry, axis maths and the provenance footer.
//
// Conventions, applied everywhere:
//   * SVG output, 1200x750 default. GitHub renders README images at roughly
//     880px, so anything wider is downsampled and the thin lines break up.
//   * One sans-serif stack with a documented fallback chain.
//   * Okabe-Ito categorical palette: colourblind-safe, at most six in use.
//   * Titles state the finding. The subtitle carries sample, period, units.
//   * No top or right spine, horizontal grid only, no fills or shadows.
//   * Every figure carries a source line in 8pt italic at bottom left.

namespace risk::plot {

// ---------------------------------------------------------------------------
// Typography and geometry
// ---------------------------------------------------------------------------
inline constexpr const char* kFontStack =
    "Inter, 'Helvetica Neue', Helvetica, Arial, 'Liberation Sans', sans-serif";

inline constexpr double kTitleSize = 15.0;
inline constexpr double kSubtitleSize = 11.0;
inline constexpr double kAxisLabelSize = 12.0;
inline constexpr double kTickSize = 10.0;
inline constexpr double kAnnotationSize = 9.0;
inline constexpr double kSourceSize = 8.0;

inline constexpr int kDefaultWidth = 1200;
inline constexpr int kDefaultHeight = 750;

// Ink colours. Deliberately not pure black: #1a1a1a reads softer on white and
// keeps the data marks as the darkest thing on the page.
inline constexpr const char* kInk = "#1a1a1a";
inline constexpr const char* kMutedInk = "#5a5a5a";
inline constexpr const char* kFaintInk = "#8a8a8a";
inline constexpr const char* kGrid = "#e4e4e4";
inline constexpr const char* kAxis = "#4a4a4a";

// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------
// Okabe-Ito, which stays distinguishable under all three common forms of
// colour vision deficiency. Never encode meaning by colour alone -- every
// figure here pairs colour with a direct label, a shape or a position.
const std::vector<std::string>& categorical_palette();

// Stable colour for a named series, so an instrument keeps its colour across
// every figure in a report.
std::string categorical_color(std::size_t index);

// Diverging blue-white-red, pinned so that `value == centre` is exactly white.
// `half_range` is the distance from the centre that saturates the scale.
std::string diverging_color(double value, double centre = 0.0,
                            double half_range = 1.0);

// Perceptually uniform sequential ramp (a viridis approximation). Never jet,
// never rainbow.
std::string sequential_color(double t);

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------
std::string escape(const std::string& s);
std::string fixed(double v, int precision = 2);

// "12.3%", not "0.123". Percent axes are formatted, never left as fractions.
std::string percent(double fraction, int precision = 1);

// "1,234,567" with thousands separators, and a leading minus outside.
std::string thousands(double v);

// "$1.2M" / "$345k" for axis ticks where the full number would collide.
std::string compact_money(double v);

// ---------------------------------------------------------------------------
// Scales
// ---------------------------------------------------------------------------
// Maps a data interval onto a pixel interval. Construct with the data range
// and the pixel range; y-scales pass pixel_lo > pixel_hi so that larger data
// values map upward.
struct Scale {
  double domain_lo = 0.0;
  double domain_hi = 1.0;
  double range_lo = 0.0;
  double range_hi = 1.0;

  [[nodiscard]] double operator()(double v) const;
};

// Ticks at 1/2/5 x 10^k covering [lo, hi], aiming for `target` of them.
std::vector<double> nice_ticks(double lo, double hi, int target = 6);

// Expand a data range outward to the nearest nice tick so bars and points do
// not touch the frame.
void pad_range(double& lo, double& hi, double fraction = 0.05);

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------
// Accumulates SVG elements and closes the document with a provenance footer.
class Figure {
 public:
  Figure(std::string title, std::string subtitle, int width = kDefaultWidth,
         int height = kDefaultHeight);

  // Plot area, measured from the document edges.
  void set_margins(double left, double right, double top, double bottom);
  [[nodiscard]] double plot_left() const { return left_; }
  [[nodiscard]] double plot_right() const { return width_ - right_; }
  [[nodiscard]] double plot_top() const { return top_; }
  [[nodiscard]] double plot_bottom() const { return height_ - bottom_; }
  [[nodiscard]] double plot_width() const { return plot_right() - plot_left(); }
  [[nodiscard]] double plot_height() const {
    return plot_bottom() - plot_top();
  }
  [[nodiscard]] int width() const { return width_; }
  [[nodiscard]] int height() const { return height_; }

  // --- primitives ---------------------------------------------------------
  void raw(const std::string& svg);
  void line(double x1, double y1, double x2, double y2,
            const std::string& stroke = kAxis, double stroke_width = 1.0,
            const std::string& dash = "");
  void rect(double x, double y, double w, double h, const std::string& fill,
            const std::string& stroke = "", double stroke_width = 0.0);
  void circle(double cx, double cy, double r, const std::string& fill,
              const std::string& stroke = "", double stroke_width = 0.0);
  void polyline(const std::vector<std::pair<double, double>>& points,
                const std::string& stroke, double stroke_width = 1.5,
                const std::string& fill = "none");
  void text(double x, double y, const std::string& content, double size,
            const std::string& fill = kInk, const std::string& anchor = "start",
            const std::string& weight = "normal",
            const std::string& style = "normal");

  // A short callout with a leader line pointing at (tx, ty), for the one or
  // two figures where the reader should be told the takeaway directly.
  void callout(double text_x, double text_y, double target_x, double target_y,
               const std::vector<std::string>& lines,
               const std::string& anchor = "start");

  // --- axes ---------------------------------------------------------------
  // Horizontal gridlines plus a left axis with formatted ticks. `formatter`
  // turns a domain value into its tick label.
  void y_axis(const Scale& y, const std::vector<double>& ticks,
              const std::string& label,
              std::string (*formatter)(double) = nullptr);
  void x_axis(const Scale& x, const std::vector<double>& ticks,
              const std::string& label,
              std::string (*formatter)(double) = nullptr);

  // Categorical tick labels along the bottom, one per band centre.
  void x_category_axis(const std::vector<double>& centres,
                       const std::vector<std::string>& labels,
                       double rotate_degrees = 0.0);

  // --- output -------------------------------------------------------------
  // `source` is the provenance line: data source, as-of date, and SYNTHETIC
  // where it applies. Required; a figure without one does not ship.
  [[nodiscard]] std::string str(const std::string& source) const;

 private:
  int width_;
  int height_;
  std::string title_;
  std::string subtitle_;
  double left_ = 90.0;
  double right_ = 40.0;
  double top_ = 78.0;
  double bottom_ = 76.0;
  std::string body_;
};

}  // namespace risk::plot
