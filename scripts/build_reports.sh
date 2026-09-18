#!/usr/bin/env bash
# Regenerate every committed artifact in reports/ from the committed data.
#
# This is the single command CI runs to prove the README's numbers are real:
# it rebuilds the worked example end to end, and CI then fails if the result
# differs from what is committed. Output is deterministic — fixed seeds, and
# every timestamp comes from the data's as-of date, never the wall clock.
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR=${BUILD_DIR:-build}
if [[ ! -x "${BUILD_DIR}/compute_risk" ]]; then
  echo "error: ${BUILD_DIR}/compute_risk not found. Build first:" >&2
  echo "  cmake -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release" >&2
  echo "  cmake --build ${BUILD_DIR} -j" >&2
  exit 1
fi

mkdir -p reports/figures

echo "==> risk report"
"./${BUILD_DIR}/compute_risk" \
  --portfolio config/portfolio.json \
  --data data/returns/ \
  --output reports/

echo
echo "==> reports/ regenerated"
