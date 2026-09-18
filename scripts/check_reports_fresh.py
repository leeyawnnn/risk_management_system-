#!/usr/bin/env python3
"""Compare a freshly generated report against the committed one, numerically.

An earlier version of this check used `git diff --quiet`, which asserted that
a Linux runner reproduces macOS output byte for byte. That assertion was
wrong, and CI proved it. The pipeline is deterministic in everything this
project controls -- fixed seeds, no wall-clock, no standard-library
distribution whose algorithm varies by platform -- but it still runs through
`std::erfc` and Eigen's eigen-decomposition and Cholesky, and those differ in
the last bits between glibc and macOS libm, and between AVX2 and NEON
summation orders. One flipped ulp can move which observation sits at the 99%
quantile.

So the honest check is numerical, not textual: every number in the report has
to agree to a stated tolerance, and the worst disagreement is printed so any
drift is visible rather than merely under the bar.

Usage:
    python3 scripts/check_reports_fresh.py committed.json regenerated.json
    python3 scripts/check_reports_fresh.py committed.json regenerated.json --rtol 1e-6
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

# Fields that are expected to differ between runs and are not results.
IGNORED_KEYS = {"git_commit"}


def walk(node, path=""):
    """Yield (path, value) for every leaf in a nested JSON structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in IGNORED_KEYS:
                continue
            yield from walk(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from walk(value, f"{path}[{i}]")
    else:
        yield path, node


def relative_difference(a: float, b: float) -> float:
    """Relative difference, falling back to absolute near zero.

    A pure relative comparison explodes on quantities that are legitimately
    about zero -- the Euler residual is 1e-18 by design -- so anything below
    the scale floor is compared absolutely instead.
    """
    scale = max(abs(a), abs(b))
    if scale < 1e-12:
        return abs(a - b)
    return abs(a - b) / scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("committed", type=pathlib.Path)
    ap.add_argument("regenerated", type=pathlib.Path)
    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="maximum tolerated relative difference (default 1e-6)",
    )
    args = ap.parse_args()

    old = dict(walk(json.loads(args.committed.read_text())))
    new = dict(walk(json.loads(args.regenerated.read_text())))

    problems: list[str] = []
    worst = 0.0
    worst_path = ""
    compared = 0

    missing = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    for path in missing:
        problems.append(f"{path}: present in the committed report, absent now")
    for path in added:
        problems.append(f"{path}: new field, not in the committed report")

    for path in sorted(set(old) & set(new)):
        a, b = old[path], new[path]
        if isinstance(a, bool) or isinstance(b, bool):
            if a != b:
                problems.append(f"{path}: {a} -> {b}")
            compared += 1
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if math.isnan(a) and math.isnan(b):
                continue
            diff = relative_difference(float(a), float(b))
            compared += 1
            if diff > worst:
                worst, worst_path = diff, path
            if diff > args.rtol:
                problems.append(f"{path}: {a!r} -> {b!r} (rel. diff {diff:.3e})")
        elif a != b:
            problems.append(f"{path}: {a!r} -> {b!r}")
            compared += 1

    print(f"compared {compared} values at rtol={args.rtol:g}")
    if worst_path:
        print(f"largest relative difference: {worst:.3e} at {worst_path}")

    if problems:
        print(f"\n{len(problems)} value(s) disagree:", file=sys.stderr)
        for p in problems[:40]:
            print(f"  - {p}", file=sys.stderr)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more", file=sys.stderr)
        return 1

    print("the regenerated report matches the committed one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
