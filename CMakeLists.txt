# 3.25 is the first release with FetchContent's SYSTEM keyword, which is what
# keeps -Werror from firing inside Eigen's headers.
cmake_minimum_required(VERSION 3.25)
project(risk_engine VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(RISK_OFFLINE     "Require system-installed dependencies; never download" OFF)
option(RISK_WERROR      "Treat compiler warnings as errors"                     ON)
option(RISK_BUILD_TESTS "Build the Catch2 test suite"                           ON)
set(RISK_SANITIZER "" CACHE STRING
    "Sanitizer to build with: empty, 'address,undefined', or 'thread'")

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
include(RiskDependencies)
risk_provide_eigen()
risk_provide_json()

# ---------------------------------------------------------------------------
# Warning + sanitizer interface
# ---------------------------------------------------------------------------
add_library(risk_flags INTERFACE)
target_compile_options(risk_flags INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast -Wnon-virtual-dtor
    -Wcast-align -Wunused -Woverloaded-virtual -Wdouble-promotion
    $<$<BOOL:${RISK_WERROR}>:-Werror>)

if(RISK_SANITIZER)
  # Sanitizer builds need frame pointers and no tail-call elision for usable
  # stack traces, and must propagate to the link line as well as compilation.
  target_compile_options(risk_flags INTERFACE
      -fsanitize=${RISK_SANITIZER} -fno-omit-frame-pointer
      -fno-optimize-sibling-calls -g)
  target_link_options(risk_flags INTERFACE -fsanitize=${RISK_SANITIZER})
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
# Sources are listed explicitly, not globbed: a GLOB silently omits new files
# until someone re-runs CMake, which is exactly the failure you notice last.
add_library(risk_core
    src/attribution.cpp
    src/backtest.cpp
    src/config.cpp
    src/covariance.cpp
    src/cvar.cpp
    src/decomposition.cpp
    src/figures.cpp
    src/portfolio.cpp
    src/reporter.cpp
    src/return_series.cpp
    src/stress_test.cpp
    src/var.cpp
    src/version.cpp)

target_include_directories(risk_core
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(risk_core
    PUBLIC  Eigen3::Eigen nlohmann_json::nlohmann_json
    PRIVATE risk_flags)

# Debug builds keep asserts (is_psd / is_symmetric) live; Release defines
# NDEBUG via CMake and compiles them out.

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
foreach(app compute_risk estimator_study tail_study)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE risk_core risk_flags)
endforeach()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(RISK_BUILD_TESTS)
  enable_testing()
  risk_provide_catch2()

  add_executable(risk_tests
      tests/test_attribution.cpp
      tests/test_backtest.cpp
      tests/test_covariance.cpp
      tests/test_covariance_reference.cpp
      tests/test_cvar.cpp
      tests/test_decomposition.cpp
      tests/test_portfolio.cpp
      tests/test_return_series.cpp
      tests/test_stress_test.cpp
      tests/test_var.cpp)
  target_link_libraries(risk_tests PRIVATE risk_core risk_flags Catch2::Catch2WithMain)
  target_compile_definitions(risk_tests PRIVATE
      RISK_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

  list(APPEND CMAKE_MODULE_PATH "${RISK_CATCH2_EXTRAS}")
  include(Catch)
  # An explicit per-test timeout: a hung test should fail CI, not stall it.
  catch_discover_tests(risk_tests PROPERTIES TIMEOUT 120)
endif()
