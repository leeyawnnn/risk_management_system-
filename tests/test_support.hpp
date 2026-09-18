#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <random>

#include "risk/random.hpp"

// Test-side sampling helpers.
//
// The tests must not use std::normal_distribution or
// std::student_t_distribution for the same reason the library does not: their
// algorithms are implementation defined, so libstdc++ and libc++ turn the same
// seed into different samples. A test whose assertion sits anywhere near a
// threshold then passes on one platform and fails on the other, which is
// exactly what happened on the first CI run.
//
// These wrap risk::standard_normal, which is a pure function of the engine's
// bit stream, so every test sees identical data everywhere.
namespace risk_test {

inline double normal(std::mt19937_64& gen, double mean = 0.0,
                     double stdev = 1.0) {
  return mean + stdev * risk::standard_normal(gen);
}

// Student-t from its definition, Z / sqrt(V / v), with V a chi-square on v
// degrees of freedom built from v standard normals. `df` must be a whole
// number, which every caller here uses.
inline double student_t(std::mt19937_64& gen, int df) {
  const double z = risk::standard_normal(gen);
  double chi_square = 0.0;
  for (int k = 0; k < df; ++k) {
    const double g = risk::standard_normal(gen);
    chi_square += g * g;
  }
  return z / std::sqrt(chi_square / static_cast<double>(df));
}

}  // namespace risk_test
