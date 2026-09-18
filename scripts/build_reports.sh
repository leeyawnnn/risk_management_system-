#!/usr/bin/env bash
# Regenerate every committed artifact under reports/ from the committed data.
#
# This is the one command CI runs to prove the README's numbers are real: it
# rebuilds the worked example end to end, and CI then fails if the result
# differs from what is committed. Output is deterministic -- fixed seeds, and
# every timestamp comes from the data's as-of date rather than the wall clock.
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR=${BUILD_DIR:-build}
for tool in compute_risk estimator_study tail_study; do
  if [[ ! -x "${BUILD_DIR}/${tool}" ]]; then
    echo "error: ${BUILD_DIR}/${tool} not found. Build first:" >&2
    echo "  cmake -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release" >&2
    echo "  cmake --build ${BUILD_DIR} -j" >&2
    exit 1
  fi
done

mkdir -p reports/figures

echo "==> estimator ground-truth study"
"./${BUILD_DIR}/estimator_study" --replications 400 --output reports/

echo
echo "==> tail study: bootstrap intervals and the Student-t comparison"
"./${BUILD_DIR}/tail_study" --resamples 10000 --output reports/

echo
echo "==> risk report and figures"
"./${BUILD_DIR}/compute_risk" \
  --portfolio config/portfolio.json \
  --data data/returns/ \
  --factors data/factors.csv \
  --output reports/ \
  --estimator-csv reports/estimator_study.csv \
  --spectrum-csv reports/estimator_spectrum.csv

echo
echo "==> reports/ regenerated"
