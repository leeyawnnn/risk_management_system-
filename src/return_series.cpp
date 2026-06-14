#include "risk/return_series.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace risk {

namespace {

// Parse a price field. Returns false if the field is empty or not a finite
// number (i.e. a "missing" observation).
bool parse_price(const std::string& field, double& out) {
  if (field.empty()) return false;
  try {
    std::size_t consumed = 0;
    double v = std::stod(field, &consumed);
    if (consumed != field.size()) return false;  // trailing junk
    if (!std::isfinite(v)) return false;
    out = v;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// Trim ASCII whitespace from both ends.
std::string trim(const std::string& s) {
  const char* ws = " \t\r\n";
  const auto b = s.find_first_not_of(ws);
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

}  // namespace

ReturnSeries ReturnSeries::from_prices(const std::vector<std::string>& dates,
                                       const std::vector<double>& prices,
                                       ReturnType type,
                                       double annualization_factor) {
  if (dates.size() != prices.size()) {
    throw std::invalid_argument(
        "ReturnSeries::from_prices: dates and prices size mismatch");
  }
  if (prices.size() < 2) {
    throw std::invalid_argument(
        "ReturnSeries::from_prices: need at least two prices to form a return");
  }
  if (annualization_factor <= 0.0) {
    throw std::invalid_argument(
        "ReturnSeries::from_prices: annualization_factor must be positive");
  }

  const std::size_t n = prices.size();
  ReturnSeries rs;
  rs.type_ = type;
  rs.annualization_factor_ = annualization_factor;
  rs.returns_.resize(static_cast<Eigen::Index>(n - 1));
  rs.dates_.reserve(n - 1);

  for (std::size_t i = 1; i < n; ++i) {
    const double p0 = prices[i - 1];
    const double p1 = prices[i];
    if (type == ReturnType::Log && (p0 <= 0.0 || p1 <= 0.0)) {
      throw std::invalid_argument(
          "ReturnSeries::from_prices: non-positive price encountered for a log "
          "return (date " + dates[i] + ")");
    }
    const double r = (type == ReturnType::Log) ? std::log(p1 / p0)
                                               : (p1 / p0 - 1.0);
    rs.returns_(static_cast<Eigen::Index>(i - 1)) = r;
    rs.dates_.push_back(dates[i]);  // label = end-of-step date
  }
  return rs;
}

ReturnSeries ReturnSeries::from_csv(const std::string& path, ReturnType type,
                                    MissingPolicy missing,
                                    double annualization_factor) {
  std::ifstream in(path);
  if (!in) {
    throw std::invalid_argument("ReturnSeries::from_csv: cannot open " + path);
  }

  std::vector<std::string> dates;
  std::vector<double> prices;

  std::string line;
  bool first = true;
  double last_valid = 0.0;
  bool have_last_valid = false;

  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string date_field, price_field;
    std::getline(ss, date_field, ',');
    std::getline(ss, price_field, ',');
    date_field = trim(date_field);
    price_field = trim(price_field);

    double price = 0.0;
    const bool ok = parse_price(price_field, price);

    // Auto-detect and skip a header row: the first line whose price field does
    // not parse as a number is treated as a header.
    if (first) {
      first = false;
      if (!ok) continue;  // header line, skip
    }

    if (!ok) {
      // Missing observation — apply the chosen policy.
      switch (missing) {
        case MissingPolicy::Skip:
          continue;  // drop the row; gap closes up
        case MissingPolicy::FillForward:
          if (!have_last_valid) continue;  // nothing to carry yet
          dates.push_back(date_field);
          prices.push_back(last_valid);    // carry → zero return this day
          continue;
      }
    }

    dates.push_back(date_field);
    prices.push_back(price);
    last_valid = price;
    have_last_valid = true;
  }

  if (prices.size() < 2) {
    throw std::invalid_argument(
        "ReturnSeries::from_csv: fewer than two usable prices in " + path);
  }
  return from_prices(dates, prices, type, annualization_factor);
}

double ReturnSeries::mean() const {
  if (returns_.size() == 0) return 0.0;
  return returns_.mean();
}

double ReturnSeries::variance() const {
  const Eigen::Index n = returns_.size();
  if (n < 2) return 0.0;
  const double m = returns_.mean();
  // Unbiased sample variance, divide by (n-1).
  return (returns_.array() - m).square().sum() / static_cast<double>(n - 1);
}

double ReturnSeries::stdev() const { return std::sqrt(variance()); }

double ReturnSeries::annualized_mean() const {
  return mean() * annualization_factor_;
}

double ReturnSeries::annualized_vol() const {
  return stdev() * std::sqrt(annualization_factor_);
}

double ReturnSeries::sum() const { return returns_.sum(); }

double ReturnSeries::cumulative_return() const {
  if (type_ == ReturnType::Log) {
    // sum of log returns = ln(P_last / P_first); invert to simple return.
    return std::exp(returns_.sum()) - 1.0;
  }
  // Product of gross simple returns minus 1.
  double gross = 1.0;
  for (Eigen::Index i = 0; i < returns_.size(); ++i) {
    gross *= (1.0 + returns_(i));
  }
  return gross - 1.0;
}

}  // namespace risk
