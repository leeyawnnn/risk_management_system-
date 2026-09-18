#!/usr/bin/env python3
"""Fail if a number quoted in the README no longer matches the artifacts.

The defect this repository was built to fix is a README asserting results the
code does not produce. Re-running the pipeline is not enough on its own: the
report regenerates, the README does not, and the two drift apart silently.

This script closes that loop. Every headline figure in README.md is listed
below with the artifact field it came from, and CI runs it on every push, so a
stale claim fails the build instead of surviving until an interviewer checks.

Usage:
    python3 scripts/check_readme_numbers.py
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
REPORT = REPO / "reports" / "report.json"
TAIL = REPO / "reports" / "tail_study.json"
STUDY = REPO / "reports" / "estimator_study.csv"
README = REPO / "README.md"


def normalise(text: str) -> str:
    """Collapse whitespace so a claim can be checked regardless of wrapping.

    The README is hard-wrapped, so a phrase the checker looks for is often
    split across two lines. Matching on collapsed whitespace means reflowing a
    paragraph does not break the build for no reason.
    """
    return " ".join(text.split())


class Checker:
    def __init__(self, readme: str) -> None:
        self.readme = normalise(readme)
        self.failures: list[str] = []
        self.checked = 0

    def must_say(self, label: str, text: str) -> None:
        """The README must contain this phrase, built from the artifact.

        There is deliberately no second "expected" argument. An earlier
        version of this script carried the README's wording alongside the
        artifact's value, which meant two places to update and a checker that
        could itself go stale. The artifact is the only source of truth here.
        """
        self.checked += 1
        if normalise(text) not in self.readme:
            self.failures.append(f"{label}: README does not say {text!r}")


def main() -> int:
    for path in (REPORT, TAIL, STUDY):
        if not path.exists():
            print(
                f"error: {path} missing. Run ./scripts/build_reports.sh",
                file=sys.stderr,
            )
            return 2

    report = json.loads(REPORT.read_text())
    tail = json.loads(TAIL.read_text())
    study = list(csv.DictReader(STUDY.open()))
    c = Checker(README.read_text())

    # --- headline volatility and moments -----------------------------------
    vol = report["volatility"]
    moments = report["portfolio_moments"]
    c.must_say("daily vol", f"{vol['daily'] * 100:.3f}% daily")
    c.must_say("annual vol", f"{vol['annual'] * 100:.2f}% annual")
    c.must_say(
        "excess kurtosis", f"excess kurtosis **{moments['excess_kurtosis']:.2f}**"
    )
    c.must_say(
        "observations",
        f"{report['backtests'][0]['kupiec']['observations']:,} daily returns",
    )
    c.must_say("as-of", report["provenance"]["as_of"])

    # --- the results table --------------------------------------------------
    one_day = {
        (v["method"], v["confidence"]): v
        for v in report["var_cvar"]
        if v["horizon_days"] == 1
    }
    for (method, conf), row in sorted(one_day.items()):
        tag = f"{method} {conf:.0%}"
        c.must_say(f"{tag} VaR", f"{row['var'] * 100:.3f}%")
        c.must_say(f"{tag} CVaR", f"{row['cvar'] * 100:.3f}%")
        c.must_say(f"{tag} VaR $", f"{row['var_dollar']:,.0f}")
        c.must_say(f"{tag} CVaR $", f"{row['cvar_dollar']:,.0f}")

    # --- backtest verdicts --------------------------------------------------
    backtests = {(b["method"], b["confidence"]): b for b in report["backtests"]}
    hist95 = backtests[("Historical", 0.95)]
    hist99 = backtests[("Historical", 0.99)]
    c.must_say("95% exceptions", f"{hist95['kupiec']['exceptions']} breaches")
    expected95 = hist95["kupiec"]["expected_rate"] * hist95["kupiec"]["observations"]
    c.must_say("95% expected", f"an expected {expected95:.1f}")
    c.must_say(
        "99% exceptions",
        f"{hist99['kupiec']['exceptions']} against "
        f"{hist99['kupiec']['expected_rate'] * hist99['kupiec']['observations']:.1f}",
    )
    # Every method must still pass Kupiec and fail Christoffersen, or finding
    # 1 is no longer true and the whole Results section needs rewriting.
    for (method, conf), b in backtests.items():
        c.checked += 1
        if b["kupiec"]["reject_at_95"]:
            c.failures.append(
                f"{method} {conf:.0%}: Kupiec now REJECTS; README claims every "
                "method passes"
            )
        if not b["christoffersen"]["reject_independence_at_95"]:
            c.failures.append(
                f"{method} {conf:.0%}: Christoffersen no longer rejects; README "
                "claims every method fails independence"
            )
        if b["basel"]["zone"] != "green":
            c.failures.append(
                f"{method} {conf:.0%}: Basel zone is {b['basel']['zone']}, "
                "README says green"
            )

    param99 = backtests[("Parametric", 0.99)]["acerbi_szekely"]
    # The README renders negatives with U+2212 MINUS SIGN, not ASCII hyphen.
    minus = "\u2212"
    c.must_say(
        "Acerbi-Szekely Z2",
        f"is {minus}{abs(param99['z2']):.3f} (p={param99['p_value_z2']:.3f})",
    )

    # --- bootstrap intervals -------------------------------------------------
    boot = {(b["measure"], b["confidence"]): b for b in tail["bootstrap"]}
    var99 = boot[("VaR", 0.99)]
    cvar99 = boot[("CVaR", 0.99)]
    c.must_say(
        "VaR99 interval",
        f"[{var99['ci_lower'] * 100:.3f}%, {var99['ci_upper'] * 100:.3f}%]",
    )
    c.must_say(
        "CVaR99 interval",
        f"[{cvar99['ci_lower'] * 100:.3f}%, {cvar99['ci_upper'] * 100:.3f}%]",
    )
    c.checked += 1
    if not var99["parametric_inside_ci"]:
        c.failures.append(
            "parametric 99% VaR is no longer inside the bootstrap interval; "
            "README finding 2 claims it is"
        )
    c.checked += 1
    if cvar99["parametric_inside_ci"]:
        c.failures.append(
            "parametric 99% CVaR is now inside the bootstrap interval; README "
            "finding 2 claims it falls outside"
        )

    # --- estimator study -----------------------------------------------------
    cells = {(r["estimator"], int(r["sample_size"])): r for r in study}
    c.must_say(
        "shrinkage conditioning",
        f"from {float(cells[('Sample', 60)]['correlation_condition_mean']):.0f} to "
        f"{float(cells[('Ledoit-Wolf', 60)]['correlation_condition_mean']):.0f}",
    )
    c.must_say(
        "estimator error range",
        f"from {float(cells[('Sample', 60)]['var99_abs_error']) * 1e4:.1f}bp to "
        f"{float(cells[('Sample', 1260)]['var99_abs_error']) * 1e4:.1f}bp",
    )
    c.checked += 1
    ewma = [float(cells[("EWMA", n)]["var99_abs_error"]) * 1e4 for n in (60, 1260)]
    if not all(9.5 < e < 12.5 for e in ewma):
        c.failures.append(
            f"EWMA error is {ewma}bp; README says it stays near 11bp at every "
            "sample size"
        )

    # The spectrum figure's headline is computed inside the figure, so check
    # the README quotes the same multiple the study produces.
    spectrum = REPO / "reports" / "estimator_spectrum.csv"
    if spectrum.exists():
        rows = [r for r in csv.DictReader(spectrum.open())]
        smallest = min(int(r["sample_size"]) for r in rows)
        lw = next(
            r
            for r in rows
            if r["estimator"] == "Ledoit-Wolf"
            and int(r["sample_size"]) == smallest
            and int(r["index"]) == 0
        )
        ratio = float(lw["estimated_eigenvalue"]) / float(lw["true_eigenvalue"])
        c.must_say("spectrum overshoot", f"{ratio:.1f}\u00d7 above it")

    # --- concentration and factor decomposition ------------------------------
    conc = report["volatility_attribution"]["concentration"]
    c.must_say("effective bets", f"{conc['effective_num_bets']:.1f} across 11")
    c.must_say(
        "top contributor share", f"{conc['max_contribution'] * 100:.1f}% of risk"
    )
    fd = report["factor_decomposition"]
    equity = next(f for f in fd["factors"] if f["factor"] == "equity")
    c.must_say(
        "equity variance share", f"({equity['percent_of_model_variance'] * 100:.1f}%)"
    )
    specific_share = fd["specific_variance"] / fd["model_variance"] * 100
    c.must_say("specific risk", f"specific risk is {specific_share:.1f}%")
    corr_cond = report["conditioning"]["correlation"]["condition_number"]
    c.must_say("conditioning", f"condition number of {corr_cond:.0f}")

    # --- reverse stress -------------------------------------------------------
    rs = report["reverse_stress"][0]
    c.must_say(
        "reverse stress distance",
        f"is {rs['mahalanobis_distance']:.1f}\nstandard deviations",
    )

    print(f"checked {c.checked} claims against reports/")
    if c.failures:
        print("\nREADME is out of date:", file=sys.stderr)
        for f in c.failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nRe-run ./scripts/build_reports.sh and update README.md.",
            file=sys.stderr,
        )
        return 1
    print("every checked claim matches the committed artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
