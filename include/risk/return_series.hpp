#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

namespace risk {

// How to turn a price level series P_t into a return series r_t.
//
//   Simple:  r_t = P_t / P_{t-1} - 1
//   Log:     r_t = ln(P_t / P_{t-1})
//
// Log returns are time-additive (sum over a window = log of the cumulative
// gross return), which is why risk models usually prefer them. Simple returns
// are asset-additive across a portfolio at a single point in time.
enum class ReturnType : std::uint8_t { Simple, Log };

// Policy for a row whose price field is absent / non-numeric (a "missing"
// observation). Genuine non-trading days (weekends, holidays) are simply not
// present in the file and need no policy — we only ever form returns between
// consecutive *present* observations.
//
//   Skip:        drop the row entirely (the gap is closed up).
//   FillForward: carry the last valid price forward (return for that day = 0).
//
// There is deliberately no silent default: the caller must choose.
enum class MissingPolicy : std::uint8_t { Skip, FillForward };

// A single asset's daily return series: an Eigen vector of returns plus the
// date label attached to each return (the date of P_t, the *end* of the step).
//
// Invariant: dates_.size() == returns_.size(). For N price observations there
// are N-1 returns.
class ReturnSeries {
 public:
  ReturnSeries() = default;

  // Build directly from aligned dates + prices (mostly for tests). `dates` has
  // one entry per price. Throws std::invalid_argument on bad input (fewer than
  // two prices, non-positive price under Log, size mismatch).
  static ReturnSeries from_prices(const std::vector<std::string>& dates,
                                  const std::vector<double>& prices,
                                  ReturnType type,
                                  double annualization_factor = 252.0);

  // Load a "date,price" CSV (header auto-detected) and compute returns.
  // `missing` decides what happens to rows with an empty / unparseable price.
  static ReturnSeries from_csv(const std::string& path, ReturnType type,
                               MissingPolicy missing,
                               double annualization_factor = 252.0);

  // ---- accessors -----------------------------------------------------------
  [[nodiscard]] const Eigen::VectorXd& returns() const { return returns_; }
  [[nodiscard]] const std::vector<std::string>& dates() const { return dates_; }
  [[nodiscard]] Eigen::Index size() const { return returns_.size(); }
  [[nodiscard]] ReturnType type() const { return type_; }
  [[nodiscard]] double annualization_factor() const {
    return annualization_factor_;
  }

  // ---- summary statistics --------------------------------------------------
  [[nodiscard]] double mean() const;      // arithmetic mean of the returns
  [[nodiscard]] double variance() const;  // sample variance (unbiased, /(n-1))
  [[nodiscard]] double stdev() const;     // sqrt(variance)

  [[nodiscard]] double annualized_mean() const;  // mean * factor
  [[nodiscard]] double annualized_vol() const;   // stdev * sqrt(factor)

  // Sum of the (log) returns over the whole window. For a Log series this
  // equals ln(P_last / P_first) by additivity. Provided for both types but
  // only carries that interpretation for Log.
  [[nodiscard]] double sum() const;

  // Cumulative *simple* return over the window: P_last / P_first - 1.
  // Computed from whichever return convention this series stores.
  [[nodiscard]] double cumulative_return() const;

 private:
  std::vector<std::string> dates_;
  Eigen::VectorXd returns_;
  ReturnType type_ = ReturnType::Log;
  double annualization_factor_ = 252.0;
};

}  // namespace risk
