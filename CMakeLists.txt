cmake_minimum_required(VERSION 3.20)
project(risk_engine LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# In Debug builds we want the asserts (is_psd / is_symmetric) to fire.
# In Release builds NDEBUG is defined by CMake automatically, disabling them.
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra")

# ---------------------------------------------------------------------------
# Dependencies (vendored via FetchContent)
# ---------------------------------------------------------------------------
include(FetchContent)

# Eigen 3.4 — header-only linear algebra
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
# We only need the headers; skip Eigen's own tests/blas/etc.
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen)

# nlohmann/json — JSON (config in, report out)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# Catch2 — testing
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.6.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
file(GLOB RISK_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_library(risk_core ${RISK_SOURCES})
target_include_directories(risk_core PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(risk_core PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json)

# ---------------------------------------------------------------------------
# CLI application
# ---------------------------------------------------------------------------
add_executable(compute_risk apps/compute_risk.cpp)
target_link_libraries(compute_risk PRIVATE risk_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/test_*.cpp")

if(TEST_SOURCES)
  add_executable(risk_tests ${TEST_SOURCES})
  target_link_libraries(risk_tests PRIVATE risk_core Catch2::Catch2WithMain)
  target_compile_definitions(risk_tests PRIVATE
    RISK_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")

  list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(risk_tests)
endif()
