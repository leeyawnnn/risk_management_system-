#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "risk/attribution.hpp"
#include "risk/backtest.hpp"
#include "risk/covariance.hpp"
#include "risk/decomposition.hpp"
#include "risk/figures.hpp"
#include "risk/plot_style.hpp"
#include "risk/portfolio.hpp"
#include "test_support.hpp"

using Catch::Matchers::WithinAbs;
using namespace risk;

namespace {

bool well_formed(const std::string& svg) {
  return svg.starts_with("<svg") && svg.find("</svg>") != std::string::npos &&
         svg.find("xmlns=\"http://www.w3.org/2000/svg\"") != std::string::npos;
}

// Every figure must carry its provenance line. A chart that cannot say where
// its numbers came from is exactly the thing this repository is fixing.
bool carries_source(const std::string& svg, const std::string& source) {
  return svg.find(source) != std::string::npos &&
         svg.find("font-style=\"italic\"") != std::string::npos;
}

const std::string kSource = "Source: test fixture, as of 2026-09-11.";

Eigen::MatrixXd block_correlation() {
  // Two tight blocks -- {0,2,4} and {1,3} -- deliberately interleaved, so a
  // clustering that works has to reorder them and one that does nothing
  // cannot pass by accident.
  Eigen::MatrixXd c = Eigen::MatrixXd::Identity(5, 5);
  const std::vector<int> block_a = {0, 2, 4};
  const std::vector<int> block_b = {1, 3};
  for (int i : block_a) {
    for (int j : block_a) {
      if (i != j) c(i, j) = 0.9;
    }
  }
  for (int i : block_b) {
    for (int j : block_b) {
      if (i != j) c(i, j) = 0.85;
    }
  }
  return c;
}

Portfolio three_asset_book() {
  Eigen::VectorXd w(3);
  w << 0.5, 0.3, 0.2;
  return Portfolio({"AAA", "BBB", "CCC"}, w, 1'000'000.0);
}

Eigen::MatrixXd three_asset_cov() {
  Eigen::VectorXd sd(3);
  sd << 0.010, 0.006, 0.020;
  Eigen::MatrixXd c(3, 3);
  c << 1.0, 0.5, -0.2, 0.5, 1.0, 0.1, -0.2, 0.1, 1.0;
  return sd.asDiagonal() * c * sd.asDiagonal();
}

Eigen::VectorXd synthetic_returns(int n) {
  Eigen::VectorXd r(n);
  for (Eigen::Index i = 0; i < r.size(); ++i) {
    const auto x = static_cast<double>(i);
    r(i) = 0.004 * std::sin(x * 0.37) + 0.002 * std::sin(x * 1.13) - 0.0002;
  }
  return r;
}

}  // namespace

// ---------------------------------------------------------------------------
// Style primitives
// ---------------------------------------------------------------------------

TEST_CASE("percent axis labels are formatted, not left as fractions",
          "[figures][style]") {
  CHECK(plot::percent(0.123) == "12.3%");
  CHECK(plot::percent(-0.0456, 2) == "-4.56%");
  // A value that rounds to zero must not print as "-0.0%".
  CHECK(plot::percent(-1e-9, 1) == "0.0%");
}

TEST_CASE("money is grouped with thousands separators", "[figures][style]") {
  CHECK(plot::thousands(1234567) == "1,234,567");
  CHECK(plot::thousands(-1234567) == "-1,234,567");
  CHECK(plot::thousands(999) == "999");
  CHECK(plot::compact_money(1'260'000) == "$1.3M");
  CHECK(plot::compact_money(-45'000) == "-$45k");
  CHECK(plot::compact_money(2'400'000'000) == "$2.4bn");
  CHECK(plot::compact_money(750) == "$750");
  // 1.25 is exactly representable, so iostreams rounds the tie to even and
  // this prints $1.2M rather than $1.3M. Pinned so the behaviour is a
  // decision rather than a surprise the next time a figure looks off by a
  // tenth.
  CHECK(plot::compact_money(1'250'000) == "$1.2M");
}

TEST_CASE("nice ticks land on round numbers and cover the range",
          "[figures][style]") {
  const auto ticks = plot::nice_ticks(0.0, 0.47, 6);
  REQUIRE(ticks.size() >= 4);
  CHECK(ticks.front() >= 0.0);
  CHECK(ticks.back() <= 0.47);
  // Steps must be uniform and land on a 1/2/5 x 10^k grid; accumulating
  // t += step instead would drift and print 0.30000000000000004.
  const double step = ticks[1] - ticks[0];
  for (std::size_t i = 1; i < ticks.size(); ++i) {
    CHECK_THAT(ticks[i] - ticks[i - 1], WithinAbs(step, 1e-12));
  }
  const double mantissa = step / std::pow(10.0, std::floor(std::log10(step)));
  CHECK((std::abs(mantissa - 1.0) < 1e-9 || std::abs(mantissa - 2.0) < 1e-9 ||
         std::abs(mantissa - 5.0) < 1e-9));
}

TEST_CASE("the diverging scale is pinned so zero is exactly white",
          "[figures][style]") {
  // The midpoint must be the true neutral value, not wherever the data
  // happens to average out.
  CHECK(plot::diverging_color(0.0, 0.0, 1.0) == "rgb(255,255,255)");
  CHECK(plot::diverging_color(1.0, 0.0, 1.0) == "rgb(178,24,43)");
  CHECK(plot::diverging_color(-1.0, 0.0, 1.0) == "rgb(33,102,172)");
  // Values beyond the half-range saturate rather than wrapping.
  CHECK(plot::diverging_color(3.0, 0.0, 1.0) ==
        plot::diverging_color(1.0, 0.0, 1.0));
}

TEST_CASE("the categorical palette stays within six working colours",
          "[figures][style]") {
  const auto& p = plot::categorical_palette();
  CHECK(p.size() >= 6);
  // Colours must be stable across calls: an instrument keeps its colour in
  // every figure of a report.
  CHECK(plot::categorical_color(0) == plot::categorical_color(0));
  CHECK(plot::categorical_color(0) != plot::categorical_color(1));
  // The index wraps rather than reading off the end.
  CHECK(plot::categorical_color(p.size()) == plot::categorical_color(0));
}

TEST_CASE("text content is escaped into SVG", "[figures][style]") {
  CHECK(plot::escape("a & b < c") == "a &amp; b &lt; c");
  plot::Figure fig("Title & <tag>", "sub");
  const std::string out = fig.str("src");
  CHECK(out.find("Title &amp; &lt;tag&gt;") != std::string::npos);
  CHECK(out.find("<tag>") == std::string::npos);
}

TEST_CASE("every figure declares an explicit background", "[figures][style]") {
  // Without one the figure is transparent and becomes unreadable against
  // GitHub's dark theme.
  plot::Figure fig("T", "S");
  CHECK(fig.str("src").find("fill=\"#ffffff\"") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Clustering
// ---------------------------------------------------------------------------

TEST_CASE("correlation clustering groups the blocks together",
          "[figures][clustering]") {
  const auto order = correlation_cluster_order(block_correlation());
  REQUIRE(order.size() == 5);

  // Every index appears exactly once.
  std::vector<std::size_t> sorted = order;
  std::ranges::sort(sorted);
  CHECK(sorted == std::vector<std::size_t>{0, 1, 2, 3, 4});

  // {0,2,4} must end up contiguous, and so must {1,3}.
  auto position_of = [&](std::size_t v) {
    return static_cast<int>(
        std::distance(order.begin(), std::ranges::find(order, v)));
  };
  std::vector<int> a = {position_of(0), position_of(2), position_of(4)};
  std::vector<int> b = {position_of(1), position_of(3)};
  std::ranges::sort(a);
  std::ranges::sort(b);
  CHECK(a[2] - a[0] == 2);
  CHECK(b[1] - b[0] == 1);
}

TEST_CASE("clustering handles degenerate sizes", "[figures][clustering]") {
  CHECK(correlation_cluster_order(Eigen::MatrixXd(0, 0)).empty());
  CHECK(correlation_cluster_order(Eigen::MatrixXd::Identity(1, 1)).size() == 1);
}

// ---------------------------------------------------------------------------
// Figures
// ---------------------------------------------------------------------------

TEST_CASE("correlation heatmap is well formed and masks the upper triangle",
          "[figures]") {
  const Eigen::MatrixXd corr = cov_to_correlation(three_asset_cov());
  const std::string svg =
      svg_correlation_heatmap(corr, {"AAA", "BBB", "CCC"}, kSource);
  CHECK(well_formed(svg));
  CHECK(carries_source(svg, kSource));
  // Lower triangle including the diagonal is 6 cells for n = 3; a full grid
  // would be 9. Counting the cell rectangles catches an unmasked triangle.
  const std::string needle = "<rect";
  std::size_t cells = 0;
  for (std::size_t pos = svg.find(needle); pos != std::string::npos;
       pos = svg.find(needle, pos + 1)) {
    ++cells;
  }
  // 6 data cells + 1 background + 60 colour-key steps.
  CHECK(cells == 67);
}

TEST_CASE("risk contribution figure shows the 1/N reference and concentration",
          "[figures]") {
  const auto a = compute_attribution(three_asset_book(), three_asset_cov());
  const std::string svg = svg_risk_contribution_bars(a, kSource);
  CHECK(well_formed(svg));
  CHECK(carries_source(svg, kSource));
  CHECK(svg.find("equal contribution 1/N") != std::string::npos);
  CHECK(svg.find("effective bets") != std::string::npos);
}

TEST_CASE("return distribution overlays the fitted normal", "[figures]") {
  const Eigen::VectorXd r = synthetic_returns(800);
  const std::string svg = svg_return_histogram(r, 0.004, 0.0055, 0.95, kSource);
  CHECK(well_formed(svg));
  CHECK(carries_source(svg, kSource));
  CHECK(svg.find("VaR") != std::string::npos);
  CHECK(svg.find("CVaR") != std::string::npos);
  CHECK(svg.find("fitted normal") != std::string::npos);
  CHECK(svg.find("<polyline") != std::string::npos);
}

TEST_CASE("volatility figure draws the bootstrap intervals", "[figures]") {
  const std::vector<std::string> names = {"EQ", "RATES", "COMD"};
  const std::vector<std::string> sectors = {"Equity", "Rates", "Commodity"};
  const std::vector<double> vols = {0.16, 0.08, 0.41};
  const std::vector<double> lo = {0.148, 0.074, 0.372};
  const std::vector<double> hi = {0.174, 0.087, 0.452};

  const std::string svg =
      svg_asset_volatility(names, vols, lo, hi, sectors, kSource);
  CHECK(well_formed(svg));
  CHECK(carries_source(svg, kSource));
  CHECK(svg.find("Equity") != std::string::npos);
  CHECK(svg.find("Commodity") != std::string::npos);
  // Three bars, three error bars with two caps each: at least 9 line elements
  // beyond the axis furniture.
  CHECK(svg.find("41.0%") != std::string::npos);
}

TEST_CASE("weight versus risk labels both ends of the dumbbell", "[figures]") {
  const std::vector<std::string> names = {"EQ", "RATES", "COMD"};
  const std::vector<double> weights = {0.6, 0.3, 0.1};
  const std::vector<double> risk = {0.89, 0.06, 0.05};
  const std::string svg = svg_weight_vs_risk(names, weights, risk, kSource);
  CHECK(well_formed(svg));
  CHECK(carries_source(svg, kSource));
  CHECK(svg.find("capital") != std::string::npos);
  CHECK(svg.find("risk") != std::string::npos);
  CHECK(svg.find("risk over capital") != std::string::npos);
}

TEST_CASE("backtest figure carries the statistics, not just the picture",
          "[figures][backtest]") {
  const Eigen::VectorXd r = synthetic_returns(600);
  const double var_level = 0.004;
  const auto k = kupiec_pof(r, var_level, 0.95);
  const auto c = christoffersen(r, var_level, 0.95);
  const auto b = basel_traffic_light(r, var_level, 0.95, 250);

  const std::string svg =
      svg_var_backtest(r, var_level, 0.95, k, c, b, kSource);
  CHECK(well_formed(svg));
  CHECK(carries_source(svg, kSource));
  // This is the regression guard for the old figure, which drew breaches and
  // attached no test at all.
  CHECK(svg.find("Kupiec") != std::string::npos);
  CHECK(svg.find("Christoffersen") != std::string::npos);
  CHECK(svg.find("Conditional coverage") != std::string::npos);
  CHECK(svg.find("Basel 250d") != std::string::npos);
  CHECK(svg.find("p-value") != std::string::npos);
}

TEST_CASE("estimator error figure draws one labelled line per estimator",
          "[figures]") {
  std::vector<EstimatorErrorPoint> points;
  for (int n : {60, 250, 1000}) {
    points.push_back({"Sample", n, 0.01 / std::sqrt(n), 0.0005 / std::sqrt(n)});
    points.push_back({"EWMA", n, 0.02, 0.001});
  }
  const std::string svg = svg_estimator_error(
      points, "mean error", "Error falls with sample size", kSource);
  CHECK(well_formed(svg));
  CHECK(carries_source(svg, kSource));
  CHECK(svg.find("Sample") != std::string::npos);
  CHECK(svg.find("EWMA") != std::string::npos);
  CHECK(svg.find("log scale") != std::string::npos);
  // Confidence bands are drawn as polygons.
  CHECK(svg.find("<polygon") != std::string::npos);
}

TEST_CASE("eigenvalue spectrum figure names the estimator that is furthest off",
          "[figures][honesty]") {
  // The truth spans two decades; Ledoit-Wolf lifts the smallest eigenvalue
  // and Sample sits close to it.
  std::vector<SpectrumPoint> points;
  const std::vector<double> truth = {1e-6, 5e-6, 2e-5, 1e-4};
  for (std::size_t i = 0; i < truth.size(); ++i) {
    const int idx = static_cast<int>(i);
    points.push_back({"Sample", idx, truth[i] * 0.98, truth[i]});
    points.push_back(
        {"Ledoit-Wolf", idx, truth[i] * (i == 0 ? 3.0 : 1.05), truth[i]});
  }
  const std::string svg = svg_eigenvalue_spectrum(points, 60, kSource);
  CHECK(well_formed(svg));
  CHECK(carries_source(svg, kSource));
  CHECK(svg.find("truth") != std::string::npos);
  CHECK(svg.find("Ledoit-Wolf lifts the smallest eigenvalue 3.0x") !=
        std::string::npos);
  CHECK(svg.find("condition number") != std::string::npos);

  // When every estimator recovers the spectrum the title must say so rather
  // than keeping the dramatic version.
  std::vector<SpectrumPoint> accurate;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    accurate.push_back(
        {"Sample", static_cast<int>(i), truth[i] * 1.01, truth[i]});
  }
  CHECK(svg_eigenvalue_spectrum(accurate, 1260, kSource)
            .find("Every estimator recovers the spectrum") !=
        std::string::npos);
}

TEST_CASE("eigenvalue spectrum figure tolerates no data", "[figures]") {
  CHECK(well_formed(svg_eigenvalue_spectrum({}, 60, kSource)));
}

TEST_CASE("estimator error figure tolerates no data", "[figures]") {
  const std::string svg = svg_estimator_error({}, "y", "title", kSource);
  CHECK(well_formed(svg));
}

TEST_CASE("factor decomposition figure reports the model error", "[figures]") {
  FactorDecomposition d;
  d.factors = {"equity", "rates_level"};
  d.exposures = Eigen::VectorXd(2);
  d.exposures << 0.4, -2.0;
  d.variance_contribution = Eigen::VectorXd(2);
  d.variance_contribution << 6e-6, 1.2e-5;
  d.factor_variance = 1.8e-5;
  d.specific_variance = 2e-6;
  d.model_variance = 2e-5;
  d.model_vol = std::sqrt(2e-5);
  d.sample_variance = 2.1e-5;
  d.model_error = -1e-6;
  d.variance_explained = 0.9;
  d.percent_of_total = d.variance_contribution / d.model_variance;

  const std::string svg = svg_factor_decomposition(d, kSource);
  CHECK(well_formed(svg));
  CHECK(carries_source(svg, kSource));
  CHECK(svg.find("specific") != std::string::npos);
  CHECK(svg.find("model error") != std::string::npos);
  CHECK(svg.find("variance explained") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

namespace {

// Pull every x/y coordinate out of an SVG and check none of them escape the
// canvas. Eyeballing a thumbnail is not a test, and a label that renders one
// pixel outside the viewBox is invisible in a README with no error anywhere.
struct Bounds {
  double min_x = 0.0;
  double max_x = 0.0;
  double min_y = 0.0;
  double max_y = 0.0;
  double width = 0.0;
  double height = 0.0;
};

double attribute(const std::string& svg, const std::string& name,
                 std::size_t from, std::size_t* next) {
  const std::string key = name + "=\"";
  const std::size_t pos = svg.find(key, from);
  if (pos == std::string::npos) {
    *next = std::string::npos;
    return 0.0;
  }
  const std::size_t start = pos + key.size();
  const std::size_t end = svg.find('"', start);
  *next = end;
  return std::stod(svg.substr(start, end - start));
}

Bounds measure(const std::string& svg) {
  Bounds b;
  std::size_t next = 0;
  b.width = attribute(svg, "width", 0, &next);
  b.height = attribute(svg, "height", 0, &next);
  b.min_x = b.width;
  b.min_y = b.height;
  b.max_x = 0.0;
  b.max_y = 0.0;

  for (const std::string& key : {std::string("x"), std::string("x1"),
                                 std::string("x2"), std::string("cx")}) {
    std::size_t pos = 0;
    while (true) {
      const std::string needle = " " + key + "=\"";
      pos = svg.find(needle, pos);
      if (pos == std::string::npos) break;
      const std::size_t start = pos + needle.size();
      const std::size_t end = svg.find('"', start);
      const double v = std::stod(svg.substr(start, end - start));
      b.min_x = std::min(b.min_x, v);
      b.max_x = std::max(b.max_x, v);
      pos = end;
    }
  }
  for (const std::string& key : {std::string("y"), std::string("y1"),
                                 std::string("y2"), std::string("cy")}) {
    std::size_t pos = 0;
    while (true) {
      const std::string needle = " " + key + "=\"";
      pos = svg.find(needle, pos);
      if (pos == std::string::npos) break;
      const std::size_t start = pos + needle.size();
      const std::size_t end = svg.find('"', start);
      const double v = std::stod(svg.substr(start, end - start));
      b.min_y = std::min(b.min_y, v);
      b.max_y = std::max(b.max_y, v);
      pos = end;
    }
  }
  return b;
}

void check_inside_canvas(const std::string& svg) {
  const Bounds b = measure(svg);
  INFO("canvas " << b.width << "x" << b.height << ", content x [" << b.min_x
                 << ", " << b.max_x << "] y [" << b.min_y << ", " << b.max_y
                 << "]");
  CHECK(b.min_x >= 0.0);
  CHECK(b.min_y >= 0.0);
  CHECK(b.max_x <= b.width);
  CHECK(b.max_y <= b.height);
  // A figure whose content stops well short of its own right edge has a
  // margin that no longer matches its contents.
  CHECK(b.max_x > b.width * 0.5);
}

}  // namespace

TEST_CASE("every figure keeps its content inside the canvas",
          "[figures][geometry]") {
  const Eigen::MatrixXd corr = cov_to_correlation(three_asset_cov());
  const auto attribution =
      compute_attribution(three_asset_book(), three_asset_cov());
  const Eigen::VectorXd r = synthetic_returns(600);

  SECTION("correlation") {
    check_inside_canvas(
        svg_correlation_heatmap(corr, {"AAA", "BBB", "CCC"}, kSource));
  }
  SECTION("risk contributions") {
    check_inside_canvas(svg_risk_contribution_bars(attribution, kSource));
  }
  SECTION("return distribution") {
    check_inside_canvas(svg_return_histogram(r, 0.004, 0.0055, 0.95, kSource));
  }
  SECTION("asset volatility") {
    check_inside_canvas(svg_asset_volatility(
        {"EQ", "RATES", "COMD"}, {0.16, 0.08, 0.41}, {0.148, 0.074, 0.372},
        {0.174, 0.087, 0.452}, {"Equity", "Rates", "Commodity"}, kSource));
  }
  SECTION("weight vs risk") {
    check_inside_canvas(svg_weight_vs_risk(
        {"EQ", "RATES", "COMD"}, {0.6, 0.3, 0.1}, {0.89, 0.06, 0.05}, kSource));
  }
  SECTION("var backtest") {
    const auto k = kupiec_pof(r, 0.004, 0.95);
    const auto c = christoffersen(r, 0.004, 0.95);
    const auto b = basel_traffic_light(r, 0.004, 0.95, 250);
    check_inside_canvas(svg_var_backtest(r, 0.004, 0.95, k, c, b, kSource));
  }
  SECTION("eigenvalue spectrum") {
    std::vector<SpectrumPoint> points;
    const std::vector<double> truth = {1e-6, 5e-6, 2e-5, 1e-4};
    for (std::size_t i = 0; i < truth.size(); ++i) {
      points.push_back(
          {"Sample", static_cast<int>(i), truth[i] * 1.4, truth[i]});
      points.push_back(
          {"Ledoit-Wolf", static_cast<int>(i), truth[i] * 0.8, truth[i]});
    }
    check_inside_canvas(svg_eigenvalue_spectrum(points, 60, kSource));
  }
  SECTION("estimator error") {
    std::vector<EstimatorErrorPoint> points;
    for (int n : {60, 250, 1000}) {
      points.push_back({"Sample", n, 0.01 / std::sqrt(n), 0.0005});
      points.push_back({"EWMA", n, 0.02, 0.001});
    }
    check_inside_canvas(
        svg_estimator_error(points, "mean error", "title", kSource));
  }
}

TEST_CASE("figure titles follow the data rather than being hardcoded",
          "[figures][honesty]") {
  // A near-normal sample must not be captioned as heavy-tailed. This is the
  // regression guard for the class of bug this repository exists to fix: a
  // claim that stayed in the output after the data stopped supporting it.
  std::mt19937_64 gen(99);
    Eigen::VectorXd gaussian(4000);
  for (Eigen::Index i = 0; i < gaussian.size(); ++i) gaussian(i) = risk_test::normal(gen, 0.0, 0.01);
  const std::string normal_svg =
      svg_return_histogram(gaussian, 0.0165, 0.019, 0.95, kSource);
  CHECK(normal_svg.find("close to normal") != std::string::npos);
  CHECK(normal_svg.find("heavier than a normal") == std::string::npos);

  // A genuinely fat-tailed sample must be captioned as one.
    Eigen::VectorXd heavy(4000);
  for (Eigen::Index i = 0; i < heavy.size(); ++i) heavy(i) = risk_test::student_t(gen, 3) * 0.005;
  const std::string heavy_svg =
      svg_return_histogram(heavy, 0.02, 0.03, 0.95, kSource);
  CHECK(heavy_svg.find("heavier than a normal") != std::string::npos);
}

TEST_CASE("the backtest title reports the verdict that actually occurred",
          "[figures][honesty]") {
  const Eigen::VectorXd r = synthetic_returns(600);
  auto title_for = [&](long count, bool clustered) {
    const long n = 600;
    Eigen::VectorXd x = Eigen::VectorXd::Constant(n, -0.001);
    // Spread the exceptions evenly across the whole series when they are
    // meant to be independent. Hardcoding a stride overruns the vector as
    // soon as the count goes up.
    const long stride = std::max<long>(1, n / std::max<long>(1, count));
    for (long i = 0; i < count; ++i) {
      const long at = clustered ? 100 + i : i * stride;
      REQUIRE(at < n);
      x(at) = -0.05;
    }
    const auto k = kupiec_pof(x, 0.02, 0.95);
    const auto c = christoffersen(x, 0.02, 0.95);
    const auto b = basel_traffic_light(x, 0.02, 0.95, 250);
    return svg_var_backtest(x, 0.02, 0.95, k, c, b, kSource);
  };

  // 30 exceptions in 600 days is exactly 5%: Kupiec must pass. Clustered,
  // Christoffersen must reject, and the title must say so.
  CHECK(title_for(30, true).find("cluster in time") != std::string::npos);
  CHECK(title_for(30, false).find("no clustering") != std::string::npos);
  // 150 exceptions in 600 days is 25% against an advertised 5%.
  CHECK(title_for(150, false).find("miscalibrated") != std::string::npos);
  (void)r;
}
