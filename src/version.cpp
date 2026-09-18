#include "risk/version.hpp"

// Placeholder translation unit so risk_core has something to compile in
// Phase 0. Real estimators land in later phases.
namespace risk {

const char* version() {
  return kVersionString;
}

}  // namespace risk
