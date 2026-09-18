# Dependency resolution for the risk engine.
#
# Every dependency follows the same three-step policy:
#
#   1. find_package(... QUIET) — use a system/package-manager copy if present.
#   2. If RISK_OFFLINE is ON and step 1 failed, fail loudly. Offline builds must
#      never touch the network; a silent fallback to a download would defeat the
#      point of the flag.
#   3. Otherwise FetchContent from a *pinned release tarball* with a SHA256.
#      Never a git clone: shallow clones of a moving tag are not reproducible,
#      and a git transport is the first thing a corporate proxy blocks.
#
# Fetched dependencies are declared SYSTEM so their headers arrive via -isystem.
# Without that, -Werror fires on Eigen's own old-style casts and the project
# cannot be built strictly at all. Warnings-as-errors must police our code, not
# a vendored library's.
#
# The pinned hashes below were produced by downloading each tarball and running
# `shasum -a 256`. If a download's hash does not match, CMake aborts rather than
# building against unverified source.

include(FetchContent)

set(RISK_EIGEN_VERSION  3.4.0)
set(RISK_JSON_VERSION   3.12.0)
set(RISK_CATCH2_VERSION 3.8.1)

# Eigen has no official GitHub-hosted release tarball: upstream lives on GitLab
# and the old github.com/eigenteam mirror was archived at 3.3.7. We therefore
# pin the canonical GitLab archive. Builds on networks that block gitlab.com
# should install Eigen from a package manager and configure -DRISK_OFFLINE=ON.
set(RISK_EIGEN_URL
    "https://gitlab.com/libeigen/eigen/-/archive/${RISK_EIGEN_VERSION}/eigen-${RISK_EIGEN_VERSION}.tar.gz")
set(RISK_EIGEN_SHA256
    "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72")

# nlohmann/json ships a purpose-built release asset (114 KB) rather than the
# 9.7 MB repository archive. Same CMake targets, a fraction of the download.
set(RISK_JSON_URL
    "https://github.com/nlohmann/json/releases/download/v${RISK_JSON_VERSION}/json.tar.xz")
set(RISK_JSON_SHA256
    "42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa")

set(RISK_CATCH2_URL
    "https://github.com/catchorg/Catch2/archive/refs/tags/v${RISK_CATCH2_VERSION}.tar.gz")
set(RISK_CATCH2_SHA256
    "18b3f70ac80fccc340d8c6ff0f339b2ae64944782f8d2fca2bd705cf47cadb79")

function(_risk_require_offline dep hint)
  message(FATAL_ERROR
      "RISK_OFFLINE=ON but ${dep} was not found on the system.\n"
      "Install it and re-configure, e.g.:\n  ${hint}")
endfunction()

# --- Eigen -------------------------------------------------------------------
function(risk_provide_eigen)
  # NO_*_PACKAGE_REGISTRY: Eigen's exported config registers itself in
  # ~/.cmake/packages, so without this find_package happily resolves to another
  # project's build tree on the same machine. That is not a system copy.
  find_package(Eigen3 ${RISK_EIGEN_VERSION} QUIET NO_MODULE
      NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)
  if(Eigen3_FOUND)
    message(STATUS "Eigen: using system copy ${Eigen3_VERSION} (${Eigen3_DIR})")
    return()
  endif()
  if(RISK_OFFLINE)
    _risk_require_offline("Eigen3"
        "apt install libeigen3-dev   |   brew install eigen")
  endif()
  message(STATUS "Eigen: fetching pinned ${RISK_EIGEN_VERSION} tarball")
  set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(Eigen3
      URL       "${RISK_EIGEN_URL}"
      URL_HASH  "SHA256=${RISK_EIGEN_SHA256}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM)
  FetchContent_MakeAvailable(Eigen3)
endfunction()

# --- nlohmann/json -----------------------------------------------------------
function(risk_provide_json)
  find_package(nlohmann_json 3.11 QUIET
      NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)
  if(nlohmann_json_FOUND)
    message(STATUS "nlohmann_json: using system copy ${nlohmann_json_VERSION}")
    return()
  endif()
  if(RISK_OFFLINE)
    _risk_require_offline("nlohmann_json"
        "apt install nlohmann-json3-dev   |   brew install nlohmann-json")
  endif()
  message(STATUS "nlohmann_json: fetching pinned ${RISK_JSON_VERSION} tarball")
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  FetchContent_Declare(nlohmann_json
      URL       "${RISK_JSON_URL}"
      URL_HASH  "SHA256=${RISK_JSON_SHA256}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM)
  FetchContent_MakeAvailable(nlohmann_json)
endfunction()

# --- Catch2 ------------------------------------------------------------------
function(risk_provide_catch2)
  find_package(Catch2 3 QUIET
      NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)
  if(Catch2_FOUND)
    message(STATUS "Catch2: using system copy ${Catch2_VERSION} (${Catch2_DIR})")
    # A packaged Catch2 installs Catch.cmake next to Catch2Config.cmake, not
    # in a sibling extras/ directory the way the source tree lays it out.
    # Offering both and letting include(Catch) pick is more robust than
    # guessing which layout the distribution used.
    set(RISK_CATCH2_EXTRAS "${Catch2_DIR};${Catch2_DIR}/.." PARENT_SCOPE)
    return()
  endif()
  if(RISK_OFFLINE)
    _risk_require_offline("Catch2 (>= 3)"
        "apt install catch2   |   brew install catch2")
  endif()
  message(STATUS "Catch2: fetching pinned ${RISK_CATCH2_VERSION} tarball")
  set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
  set(CATCH_INSTALL_EXTRAS ON CACHE BOOL "" FORCE)
  FetchContent_Declare(Catch2
      URL       "${RISK_CATCH2_URL}"
      URL_HASH  "SHA256=${RISK_CATCH2_SHA256}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM)
  FetchContent_MakeAvailable(Catch2)
  set(RISK_CATCH2_EXTRAS "${catch2_SOURCE_DIR}/extras" PARENT_SCOPE)
endfunction()
