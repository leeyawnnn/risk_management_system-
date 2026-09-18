#!/usr/bin/env python3
"""Build the worked example's instrument universe from FRED daily observations.

Every series is a real market observation published by FRED. Nothing here is
simulated. Two kinds of instrument come out the other end:

*Price instruments* (equity indices, crude, FX) are used as published: the
level series is the instrument level, and the C++ loader forms log returns.

*Yield instruments* (Treasury constant maturities, Moody's seasoned corporate
yields) are not returns and cannot be treated as such. A 10-year yield moving
from 4.00% to 4.05% is not a 1.25% loss. They are converted into the total
return of a constant-maturity par bond: each day we hold a bond struck at
par on yesterday's quoted yield and reprice it one day later at today's yield,
collecting one day of accrued coupon. That is an exact bond-maths calculation,
not a duration approximation -- see par_bond_total_return below.

The output is one `date,price` CSV per instrument under data/returns/, plus
data/manifest.json recording the source series, the retrieval date, a SHA256 of
each raw payload, and every date the alignment step dropped.

Usage:
    python3 scripts/fetch_data.py                 # pull through today
    python3 scripts/fetch_data.py --as-of 2026-09-11
    python3 scripts/fetch_data.py --years 5
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import pathlib
import subprocess
import sys

import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "returns"
FACTOR_FILE = REPO / "data" / "factors.csv"
MANIFEST = REPO / "data" / "manifest.json"

# Trading days per year. Used only to convert one calendar step into a fraction
# of a coupon period; the return series itself is dated, not indexed.
TRADING_DAYS = 252.0


class Instrument:
    """One tradeable line in the book, and how to build it from a FRED series."""

    def __init__(
        self,
        code: str,
        series: str,
        asset_class: str,
        kind: str,
        description: str,
        maturity_years: float | None = None,
        publisher: str = "",
        terms: str = "",
    ) -> None:
        self.code = code
        self.series = series
        self.asset_class = asset_class
        self.kind = kind  # "price" or "par_yield"
        self.description = description
        self.maturity_years = maturity_years
        self.publisher = publisher
        self.terms = terms


PUBLIC_DOMAIN = "US Government work, public domain"
SPDJI = "S&P Dow Jones Indices LLC; redistribution restricted, see FRED series page"
NASDAQ_TERMS = "Nasdaq OMX; redistribution restricted, see FRED series page"
MOODYS = "Moody's / Ryan ALM; redistribution restricted, see FRED series page"

UNIVERSE = [
    Instrument(
        "EQ_US_LARGE",
        "SP500",
        "Equity",
        "price",
        "S&P 500 price index (excludes dividends)",
        publisher="S&P Dow Jones Indices",
        terms=SPDJI,
    ),
    Instrument(
        "EQ_US_TECH",
        "NASDAQ100",
        "Equity",
        "price",
        "Nasdaq-100 price index (excludes dividends)",
        publisher="Nasdaq OMX Group",
        terms=NASDAQ_TERMS,
    ),
    Instrument(
        "EQ_US_BLUE",
        "DJIA",
        "Equity",
        "price",
        "Dow Jones Industrial Average price index (excludes dividends)",
        publisher="S&P Dow Jones Indices",
        terms=SPDJI,
    ),
    Instrument(
        "UST_2Y",
        "DGS2",
        "Rates",
        "par_yield",
        "2-year Treasury constant maturity, held as a rolling par bond",
        maturity_years=2.0,
        publisher="Board of Governors of the Federal Reserve System (H.15)",
        terms=PUBLIC_DOMAIN,
    ),
    Instrument(
        "UST_10Y",
        "DGS10",
        "Rates",
        "par_yield",
        "10-year Treasury constant maturity, held as a rolling par bond",
        maturity_years=10.0,
        publisher="Board of Governors of the Federal Reserve System (H.15)",
        terms=PUBLIC_DOMAIN,
    ),
    Instrument(
        "UST_30Y",
        "DGS30",
        "Rates",
        "par_yield",
        "30-year Treasury constant maturity, held as a rolling par bond",
        maturity_years=30.0,
        publisher="Board of Governors of the Federal Reserve System (H.15)",
        terms=PUBLIC_DOMAIN,
    ),
    Instrument(
        "CREDIT_AAA",
        "DAAA",
        "Credit",
        "par_yield",
        "Moody's seasoned Aaa corporate yield, held as a rolling 20y par bond",
        maturity_years=20.0,
        publisher="Moody's",
        terms=MOODYS,
    ),
    Instrument(
        "CREDIT_BAA",
        "DBAA",
        "Credit",
        "par_yield",
        "Moody's seasoned Baa corporate yield, held as a rolling 20y par bond",
        maturity_years=20.0,
        publisher="Moody's",
        terms=MOODYS,
    ),
    Instrument(
        "COMD_WTI",
        "DCOILWTICO",
        "Commodity",
        "price",
        "West Texas Intermediate spot, USD per barrel",
        publisher="U.S. Energy Information Administration",
        terms=PUBLIC_DOMAIN,
    ),
    Instrument(
        "FX_USD_BROAD",
        "DTWEXBGS",
        "FX",
        "price",
        "Nominal broad USD index; long USD against a trade-weighted basket",
        publisher="Board of Governors of the Federal Reserve System (H.10)",
        terms=PUBLIC_DOMAIN,
    ),
    Instrument(
        "FX_EURUSD",
        "DEXUSEU",
        "FX",
        "price",
        "EUR/USD spot, USD per EUR; long EUR against USD",
        publisher="Board of Governors of the Federal Reserve System (H.10)",
        terms=PUBLIC_DOMAIN,
    ),
]


def par_bond_total_return(y_prev: float, y_now: float, maturity_years: float) -> float:
    """One day's total return on a constant-maturity par bond.

    A constant-maturity yield is a *par* yield: a bond of that maturity paying
    that coupon semiannually would price at 100. So yesterday we buy a bond
    with coupon c = y_prev at a dirty price of exactly 100. Today we reprice
    the same cashflows one day closer, at today's yield y_now, and the return
    is the change in dirty price. Accrued coupon is inside the dirty price, so
    carry is captured without adding it separately.

    Both yields are decimals (0.0401, not 4.01). The calculation is exact bond
    maths for the stated convention -- there is no duration or convexity
    approximation anywhere in it, so it stays correct for the large yield moves
    where a -D*dy expansion starts to drift.
    """
    if maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    n = int(round(maturity_years * 2))  # semiannual coupon count
    coupon = 100.0 * y_prev / 2.0  # semiannual coupon in points of face
    rate = y_now / 2.0  # per-period discount rate

    # One trading day as a fraction of a semiannual period. w is the remaining
    # fraction of the current period at today's close.
    day_fraction = (1.0 / TRADING_DAYS) / 0.5
    w = 1.0 - day_fraction

    if abs(rate) < 1e-12:
        # Zero-yield edge case: discounting is the identity.
        dirty = coupon * n + 100.0
    else:
        disc = 1.0 + rate
        dirty = sum(coupon / disc ** (k - 1 + w) for k in range(1, n + 1))
        dirty += 100.0 / disc ** (n - 1 + w)

    return dirty / 100.0 - 1.0


# Observable risk factors, built from the same FRED pulls. Equity, USD and oil
# are log returns of a series that is also a book constituent, so those three
# betas are close to definitional for the constituent itself; the informative
# numbers are the cross-asset loadings. The rate and spread factors are yield
# changes in decimal (0.0001 = 1bp), not returns.
FACTOR_DEFS = {
    "equity": "log return of SP500",
    "rates_level": "average daily change in the 2y/10y/30y CMT yields, decimal",
    "rates_slope": "daily change in (30y - 2y) CMT, decimal",
    "credit_spread": "daily change in the Baa yield over the 10y CMT, decimal",
    "usd": "log return of the broad nominal USD index",
    "oil": "log return of WTI spot",
}


def build_factors(
    raw: dict[str, dict[str, float]], dates: list[str]
) -> dict[str, list[float]]:
    """Daily realisations of each observable factor over the aligned calendar."""
    import math

    def logret(code: str) -> list[float]:
        obs = raw[code]
        return [
            math.log(obs[b] / obs[a]) for a, b in zip(dates[:-1], dates[1:], strict=True)
        ]

    def dyield(code: str) -> list[float]:
        obs = raw[code]
        return [
            (obs[b] - obs[a]) / 100.0
            for a, b in zip(dates[:-1], dates[1:], strict=True)
        ]

    d2, d10, d30 = dyield("UST_2Y"), dyield("UST_10Y"), dyield("UST_30Y")
    baa = dyield("CREDIT_BAA")
    return {
        "equity": logret("EQ_US_LARGE"),
        "rates_level": [(a + b + c) / 3.0 for a, b, c in zip(d2, d10, d30, strict=True)],
        "rates_slope": [c - a for a, c in zip(d2, d30, strict=True)],
        # Baa over the 10y Treasury: the standard corporate spread, and the
        # definition behind FRED's own BAA10Y. Defining it as Baa - Aaa instead
        # makes the Aaa leg's own yield change part of the factor, which hands
        # the Aaa instrument a spurious positive credit beta.
        "credit_spread": [b - t for t, b in zip(d10, baa, strict=True)],
        "usd": logret("FX_USD_BROAD"),
        "oil": logret("COMD_WTI"),
    }


def fetch_series(series_id: str, start: str, end: str) -> tuple[dict[str, float], str]:
    """Download one FRED series. Returns {date: value} and the payload SHA256.

    Dates whose value is blank or "." are market holidays or non-publication
    days for that series and are omitted rather than filled -- see align().
    """
    params = {"id": series_id, "cosd": start, "coed": end}
    resp = requests.get(FRED_CSV, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.content
    digest = hashlib.sha256(payload).hexdigest()

    text = payload.decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or len(rows[0]) < 2:
        raise RuntimeError(f"{series_id}: unexpected CSV shape from FRED")

    out: dict[str, float] = {}
    for row in rows[1:]:
        if len(row) < 2:
            continue
        date, raw = row[0].strip(), row[1].strip()
        if raw in ("", "."):
            continue
        out[date] = float(raw)
    if not out:
        raise RuntimeError(f"{series_id}: FRED returned no observations")
    return out, digest


def align(
    raw: dict[str, dict[str, float]],
) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Inner-join the series onto the dates where every instrument quotes.

    The alternative -- forward-filling a missing quote -- looks harmless and is
    not. A filled day contributes a zero return for that one instrument while
    the rest of the book moves, which drags every correlation involving it
    toward zero and understates portfolio risk. Dropping the date costs a
    handful of observations and biases nothing.
    """
    sets = [set(v) for v in raw.values()]
    common = sorted(set.intersection(*sets))
    union = sorted(set.union(*sets))

    dropped = []
    for date in union:
        absent = sorted(code for code, obs in raw.items() if date not in obs)
        if absent:
            dropped.append((date, absent))
    return common, dropped


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--as-of",
        default=dt.date.today().isoformat(),
        help="last observation date to request (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--years", type=float, default=5.0, help="length of history to request"
    )
    args = ap.parse_args()

    end = dt.date.fromisoformat(args.as_of)
    start = end - dt.timedelta(days=int(args.years * 365.25))

    raw: dict[str, dict[str, float]] = {}
    digests: dict[str, str] = {}
    for inst in UNIVERSE:
        obs, digest = fetch_series(inst.series, start.isoformat(), end.isoformat())
        raw[inst.code] = obs
        digests[inst.code] = digest
        print(f"  {inst.code:14s} {inst.series:12s} {len(obs):5d} observations")

    common, dropped = align(raw)
    if len(common) < 100:
        raise RuntimeError(f"only {len(common)} aligned dates; refusing to write")

    print(f"\naligned trading days: {len(common)}  ({common[0]} .. {common[-1]})")
    print(f"dates dropped by the inner join: {len(dropped)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for inst in UNIVERSE:
        obs = raw[inst.code]
        if inst.kind == "price":
            levels = [obs[d] for d in common]
        else:
            # Chain daily par-bond total returns into an index starting at 100.
            assert inst.maturity_years is not None
            levels = [100.0]
            for prev, now in zip(common[:-1], common[1:], strict=True):
                r = par_bond_total_return(
                    obs[prev] / 100.0, obs[now] / 100.0, inst.maturity_years
                )
                levels.append(levels[-1] * (1.0 + r))

        path = OUT_DIR / f"{inst.code}.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["date", "price"])
            for date, level in zip(common, levels, strict=True):
                w.writerow([date, f"{level:.6f}"])
        written.append(inst.code)

    factors = build_factors(raw, common)
    with FACTOR_FILE.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        names = list(FACTOR_DEFS)
        w.writerow(["date", *names])
        for i, date in enumerate(common[1:]):
            w.writerow([date, *(f"{factors[n][i]:.10f}" for n in names)])

    manifest = {
        "generated_by": "scripts/fetch_data.py",
        "git_commit": git_sha(),
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "as_of": common[-1],
        "requested_through": end.isoformat(),
        "sample_start": common[0],
        "sample_end": common[-1],
        "aligned_observations": len(common),
        "alignment_policy": "inner join on dates where every instrument quotes",
        "source": "Federal Reserve Bank of St. Louis (FRED)",
        "source_url": FRED_CSV,
        "instruments": [
            {
                "code": i.code,
                "fred_series": i.series,
                "asset_class": i.asset_class,
                "construction": i.kind,
                "maturity_years": i.maturity_years,
                "description": i.description,
                "publisher": i.publisher,
                "terms": i.terms,
                "raw_sha256": digests[i.code],
                "raw_observations": len(raw[i.code]),
            }
            for i in UNIVERSE
        ],
        "factors": FACTOR_DEFS,
        "dropped_dates": [{"date": d, "missing_in": codes} for d, codes in dropped],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {len(written)} instruments to {OUT_DIR}")
    print(f"wrote {FACTOR_FILE}")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
