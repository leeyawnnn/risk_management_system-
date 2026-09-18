#!/usr/bin/env python3
"""Estimate each instrument's factor betas by OLS and write config/factor_betas.json.

The previous betas in that file were typed in by hand. These are regression
coefficients estimated on the committed sample, so the stress engine translates
a factor shock into asset shocks using loadings the data actually supports, and
the factor risk decomposition has a residual that means something.

For each instrument i:

    r_i(t) = alpha_i + sum_f beta_{i,f} * F_f(t) + e_i(t)

Factors are the observable series in data/factors.csv. Three of them (equity,
usd, oil) are log returns of series that are themselves book constituents, so
those constituents regress on themselves and come back with a beta of 1.0 and
an R-squared of 1.0. That is arithmetic, not a finding, and the output marks
those rows `self_referential` so nothing downstream reports them as insight.

Output records, per instrument: the betas, heteroskedasticity-consistent (HC1)
t-statistics, R-squared, and the annualised residual volatility that the factor
model leaves unexplained.

Usage:
    python3 scripts/estimate_factor_betas.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import subprocess

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
RETURNS = REPO / "data" / "returns"
FACTORS = REPO / "data" / "factors.csv"
OUT = REPO / "config" / "factor_betas.json"
MANIFEST = REPO / "data" / "manifest.json"

TRADING_DAYS = 252.0

# Which factor each instrument is definitionally identical to, if any.
SELF_FACTOR = {
    "EQ_US_LARGE": "equity",
    "FX_USD_BROAD": "usd",
    "COMD_WTI": "oil",
}


def load_returns() -> tuple[list[str], list[str], np.ndarray]:
    """Log returns for every instrument, aligned on the factor dates."""
    codes = sorted(p.stem for p in RETURNS.glob("*.csv"))
    dates: list[str] | None = None
    cols = []
    for code in codes:
        rows = list(csv.reader((RETURNS / f"{code}.csv").open()))[1:]
        d = [r[0] for r in rows[1:]]
        px = np.array([float(r[1]) for r in rows])
        cols.append(np.diff(np.log(px)))
        if dates is None:
            dates = d
        elif dates != d:
            raise RuntimeError(f"{code}: date column does not match the others")
    assert dates is not None
    return codes, dates, np.column_stack(cols)


def load_factors() -> tuple[list[str], list[str], np.ndarray]:
    rows = list(csv.reader(FACTORS.open()))
    names = rows[0][1:]
    dates = [r[0] for r in rows[1:]]
    values = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    return names, dates, values


def ols_hc1(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with HC1 robust standard errors. X must already carry an intercept.

    Daily financial returns are conspicuously heteroskedastic -- volatility
    clusters -- so classical standard errors overstate precision. HC1 is the
    small-sample-corrected White estimator.
    """
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    meat = X.T @ (X * (resid**2)[:, None])
    cov = xtx_inv @ meat @ xtx_inv * (n / (n - k))
    se = np.sqrt(np.diag(cov))
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / tss if tss > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)
    return beta, tstat, r2


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main() -> int:
    codes, rdates, R = load_returns()
    fnames, fdates, F = load_factors()
    if rdates != fdates:
        raise RuntimeError("return dates and factor dates are not aligned")

    X = np.column_stack([np.ones(len(F)), F])
    manifest = json.loads(MANIFEST.read_text())

    betas: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict] = {}

    print(f"{'instrument':14s} {'R2':>6s} {'resid vol':>10s}   betas")
    for i, code in enumerate(codes):
        y = R[:, i]
        beta, tstat, r2 = ols_hc1(y, X)
        resid = y - X @ beta
        resid_vol = float(resid.std(ddof=X.shape[1]) * np.sqrt(TRADING_DAYS))

        betas[code] = {f: float(beta[j + 1]) for j, f in enumerate(fnames)}
        diagnostics[code] = {
            "alpha_annual": float(beta[0] * TRADING_DAYS),
            "r_squared": r2,
            "residual_vol_annual": resid_vol,
            "t_stats": {f: float(tstat[j + 1]) for j, f in enumerate(fnames)},
            "self_referential": SELF_FACTOR.get(code),
        }
        shown = " ".join(f"{f}={beta[j + 1]:+.3f}" for j, f in enumerate(fnames))
        print(f"{code:14s} {r2:6.3f} {resid_vol * 100:9.2f}%   {shown}")

    out = {
        "_meta": {
            "generated_by": "scripts/estimate_factor_betas.py",
            "git_commit": git_sha(),
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "estimator": "OLS with HC1 robust standard errors",
            "sample_start": manifest["sample_start"],
            "sample_end": manifest["sample_end"],
            "observations": len(F),
            "factors": manifest["factors"],
            "note": (
                "equity, usd and oil are log returns of SP500, DTWEXBGS and "
                "DCOILWTICO, which are also book constituents. Those three "
                "instruments therefore regress on themselves; their unit beta "
                "and unit R-squared are definitional and are flagged "
                "self_referential in diagnostics."
            ),
        },
        "betas": betas,
        "diagnostics": diagnostics,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
