#pragma once

#include <cstdint>

#include "risk/var.hpp"

namespace risk {

// ===========================================================================
// Reproducible random numbers
// ===========================================================================
// The standard library's distributions are NOT portable. std::mt19937_64 is
// specified bit for bit, but std::normal_distribution, std::uniform_int_
// distribution and friends are only specified by their distribution, not by
// their algorithm. libstdc++ and libc++ implement them differently, so the
// same seed produces different draws on Linux and macOS.
//
// For this project that is a correctness problem, not a curiosity. A report
// whose Monte Carlo VaR changes depending on which standard library built it
// cannot be checked against a committed artifact, and CI regenerating the
// worked example on Linux would not reproduce numbers generated on a Mac.
//
// Everything below is defined in terms of the engine's raw 64-bit output and
// arithmetic this project controls, so a seed determines the draws exactly,
// on every platform and every standard library.

// Uniform on [0, 1) from one 64-bit draw. The top 53 bits are used because
// that is exactly the mantissa of a double; the result is a multiple of
// 2^-53 and is never 1.0.
inline double uniform_unit(std::uint64_t bits) {
  return static_cast<double>(bits >> 11) * 0x1.0p-53;
}

// Uniform on the OPEN interval (0, 1), for callers feeding an inverse CDF.
// Both endpoints have to be excluded: Phi^{-1}(0) is negative infinity and
// Phi^{-1}(1) is positive infinity, and normal_ppf rejects both.
//
// The obvious ((bits >> 11) + 0.5) * 2^-53 does not work. At the top of the
// range that is (2^53 - 0.5) * 2^-53, and 2^53 - 0.5 is not representable in
// a double, so it rounds up and the result is exactly 1.0. Using 52 bits
// instead keeps 2*(bits >> 12) + 1 at or below 2^53 - 1, which is exact, so
// the result is an odd multiple of 2^-53 and lies strictly inside (0, 1).
// One bit of resolution is a cheap price: normal_ppf is nowhere near that
// accurate anyway.
inline double uniform_open_unit(std::uint64_t bits) {
  return static_cast<double>(2 * (bits >> 12) + 1) * 0x1.0p-53;
}

// Uniform integer in [0, n), by Lemire's multiply-shift with rejection.
// Unbiased, and unlike std::uniform_int_distribution the rejection rule is
// fixed here rather than left to the implementation.
template <typename Engine>
std::uint64_t uniform_below(Engine& engine, std::uint64_t n) {
  if (n <= 1) return 0;
  // Reject the residue window that would otherwise make low values slightly
  // more likely. The expected number of rejections is under one.
  const std::uint64_t threshold = -n % n;
  while (true) {
    const std::uint64_t x = engine();
    const auto product =
        static_cast<__uint128_t>(x) * static_cast<__uint128_t>(n);
    const auto low = static_cast<std::uint64_t>(product);
    if (low >= threshold) return static_cast<std::uint64_t>(product >> 64);
  }
}

// Standard normal by inverse CDF, using this project's own normal_ppf.
// Slower than a ziggurat and entirely deterministic, which is the trade this
// project wants: every Monte Carlo figure in the README has to be
// reproducible from its stated seed.
template <typename Engine>
double standard_normal(Engine& engine) {
  return normal_ppf(uniform_open_unit(engine()));
}

}  // namespace risk
