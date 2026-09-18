#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <string>

#include "risk/return_series.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using risk::MissingPolicy;
using risk::ReturnSeries;
using risk::ReturnType;

namespace {
std::string data_path(const std::string& name) {
  return std::string(RISK_TEST_DATA_DIR) + "/" + name;
}

// Known fixture: 100, 102, 101, 105, 103.
const std::vector<std::string> kDates = {"d0", "d1", "d2", "d3", "d4"};
const std::vector<double> kPrices = {100.0, 102.0, 101.0, 105.0, 103.0};
}  // namespace

TEST_CASE("simple returns match hand-computed reference", "[return_series]") {
  auto rs = ReturnSeries::from_prices(kDates, kPrices, ReturnType::Simple);
  REQUIRE(rs.size() == 4);

  // Reference values computed by hand: P_t/P_{t-1} - 1.
  CHECK_THAT(rs.returns()(0), WithinRel(0.02, 1e-12));
  CHECK_THAT(rs.returns()(1), WithinRel(101.0 / 102.0 - 1.0, 1e-12));
  CHECK_THAT(rs.returns()(2), WithinRel(105.0 / 101.0 - 1.0, 1e-12));
  CHECK_THAT(rs.returns()(3), WithinRel(103.0 / 105.0 - 1.0, 1e-12));

  // The date attached to each return is the END-of-step date.
  CHECK(rs.dates()[0] == "d1");
  CHECK(rs.dates()[3] == "d4");
}

TEST_CASE("log returns match hand-computed reference", "[return_series]") {
  auto rs = ReturnSeries::from_prices(kDates, kPrices, ReturnType::Log);
  REQUIRE(rs.size() == 4);
  CHECK_THAT(rs.returns()(0), WithinRel(std::log(1.02), 1e-12));
  CHECK_THAT(rs.returns()(3), WithinRel(std::log(103.0 / 105.0), 1e-12));
}

TEST_CASE("log-return additivity: sum == log of cumulative gross return",
          "[return_series]") {
  auto rs = ReturnSeries::from_prices(kDates, kPrices, ReturnType::Log);

  // Sum of log returns must equal ln(P_last / P_first).
  const double expected = std::log(kPrices.back() / kPrices.front());
  CHECK_THAT(rs.sum(), WithinAbs(expected, 1e-12));

  // And the recovered simple cumulative return is P_last/P_first - 1 = 3%.
  CHECK_THAT(rs.cumulative_return(), WithinAbs(0.03, 1e-12));
}

TEST_CASE("cumulative return agrees across Log and Simple conventions",
          "[return_series]") {
  auto log_rs = ReturnSeries::from_prices(kDates, kPrices, ReturnType::Log);
  auto simple_rs =
      ReturnSeries::from_prices(kDates, kPrices, ReturnType::Simple);
  CHECK_THAT(log_rs.cumulative_return(),
             WithinAbs(simple_rs.cumulative_return(), 1e-12));
  CHECK_THAT(simple_rs.cumulative_return(), WithinAbs(0.03, 1e-12));
}

TEST_CASE("annualization factor scales mean and vol correctly",
          "[return_series]") {
  auto rs = ReturnSeries::from_prices(kDates, kPrices, ReturnType::Log, 252.0);
  CHECK_THAT(rs.annualized_mean(), WithinRel(rs.mean() * 252.0, 1e-12));
  CHECK_THAT(rs.annualized_vol(),
             WithinRel(rs.stdev() * std::sqrt(252.0), 1e-12));
}

TEST_CASE("sample variance uses the unbiased (n-1) divisor",
          "[return_series]") {
  auto rs = ReturnSeries::from_prices(kDates, kPrices, ReturnType::Simple);
  const auto& r = rs.returns();
  const double m = r.mean();
  double ss = 0.0;
  for (Eigen::Index i = 0; i < r.size(); ++i) ss += (r(i) - m) * (r(i) - m);
  const double unbiased = ss / static_cast<double>(r.size() - 1);
  CHECK_THAT(rs.variance(), WithinRel(unbiased, 1e-12));
}

TEST_CASE("CSV loader reproduces the in-memory series",
          "[return_series][csv]") {
  auto rs = ReturnSeries::from_csv(data_path("known_prices.csv"),
                                   ReturnType::Log, MissingPolicy::Skip);
  REQUIRE(rs.size() == 4);
  CHECK_THAT(rs.returns()(0), WithinRel(std::log(1.02), 1e-12));
  CHECK(rs.dates()[0] == "2020-01-03");
}

TEST_CASE("missing data: Skip closes the gap", "[return_series][missing]") {
  // missing_prices.csv: 100, <missing>, 101, 105
  auto rs = ReturnSeries::from_csv(data_path("missing_prices.csv"),
                                   ReturnType::Simple, MissingPolicy::Skip);
  // Present prices are 100, 101, 105 -> 2 returns.
  REQUIRE(rs.size() == 2);
  CHECK_THAT(rs.returns()(0), WithinRel(101.0 / 100.0 - 1.0, 1e-12));
  CHECK_THAT(rs.returns()(1), WithinRel(105.0 / 101.0 - 1.0, 1e-12));
}

TEST_CASE("missing data: FillForward carries last price (zero return)",
          "[return_series][missing]") {
  // 100, <carry 100>, 101, 105 -> 3 returns, first is exactly 0.
  auto rs =
      ReturnSeries::from_csv(data_path("missing_prices.csv"),
                             ReturnType::Simple, MissingPolicy::FillForward);
  REQUIRE(rs.size() == 3);
  CHECK_THAT(rs.returns()(0), WithinAbs(0.0, 1e-15));
  CHECK_THAT(rs.returns()(1), WithinRel(101.0 / 100.0 - 1.0, 1e-12));
  CHECK(rs.dates()[0] == "2020-01-03");
}

TEST_CASE("invalid input is rejected loudly", "[return_series][errors]") {
  // Fewer than two prices.
  CHECK_THROWS_AS(ReturnSeries::from_prices({"d0"}, {100.0}, ReturnType::Log),
                  std::invalid_argument);
  // Non-positive price under a log return.
  CHECK_THROWS_AS(
      ReturnSeries::from_prices({"d0", "d1"}, {100.0, -5.0}, ReturnType::Log),
      std::invalid_argument);
  // Size mismatch.
  CHECK_THROWS_AS(
      ReturnSeries::from_prices({"d0"}, {100.0, 101.0}, ReturnType::Simple),
      std::invalid_argument);
}
