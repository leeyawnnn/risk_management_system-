#pragma once

namespace risk {

// Tracks the engine, not the data. Report artifacts additionally carry the
// git commit and the data as-of date, which is what actually identifies a run.
inline constexpr int kVersionMajor = 1;
inline constexpr int kVersionMinor = 0;
inline constexpr const char* kVersionString = "1.0.0";

}  // namespace risk
