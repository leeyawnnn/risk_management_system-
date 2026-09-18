#!/usr/bin/env python3
"""Measure historical stress scenarios from FRED long-history series.

Every shock size in config/stress_scenarios.json is the factor's *realised*
move over a named window, read from a FRED series that goes back far enough to
cover it. Nothing is estimated from memory and nothing is rounded to a
convenient number, so each scenario carries the series, the two dates, the two
levels and the resulting move.

Factor definitions match data/factors.csv so the same betas apply:

    equity         log return of the equity proxy
    rates_level    change in the 10y CMT yield, decimal
    credit_spread  change in the Baa yield over the 10y CMT, decimal
    usd            log return of the broad nominal USD index
    oil            log return of WTI spot

The equity proxy is the Nasdaq Composite (NASDAQCOM), the only equity series on
FRED with history back to 1971 -- S&P 500 and DJIA are restricted to a rolling
ten-year window. That has one documented consequence, noted on the scenario
itself: on 1987-10-19 the Nasdaq Composite closed down 11.35% while the S&P 500
fell 20.47%, because Nasdaq market makers stopped answering their phones and
quotes went stale. The Black Monday scenario therefore understates the equity
shock, and the file says so.

The dollar index changes source mid-history: DTWEXM (major currencies, 1973 to
its discontinuation at the end of 2019) before 2006, DTWEXBGS (broad, goods and
services, 2006 onward) after. Each scenario records which one it used.

Usage:
    python3 scripts/build_stress_scenarios.py
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import pathlib
import subprocess

import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "config" / "stress_scenarios.json"

# name -> (start, end, description)
EPISODES = [
    (
        "1987 Black Monday",
        "1987-10-16",
        "1987-10-19",
        "The single-session crash of 19 October 1987.",
    ),
    (
        "1998 Russia / LTCM",
        "1998-08-17",
        "1998-10-08",
        "Russia's default on 17 August 1998 through the flight-to-quality "
        "peak that preceded the LTCM recapitalisation.",
    ),
    (
        "2008 Lehman",
        "2008-09-12",
        "2008-11-20",
        "The last session before Lehman Brothers filed, through the "
        "November 2008 equity and credit trough.",
    ),
    (
        "2020 COVID crash",
        "2020-02-19",
        "2020-03-23",
        "The 2020 equity peak through the 23 March trough.",
    ),
    (
        "2022 rates repricing",
        "2022-01-03",
        "2022-10-24",
        "The 2022 tightening cycle, from the first session of the year to "
        "the peak in long-end yields.",
    ),
    (
        "2023 SVB",
        "2023-03-08",
        "2023-03-15",
        "The Silicon Valley Bank failure and the front-end rally it caused.",
    ),
]

SERIES = {
    "NASDAQCOM": "Nasdaq Composite index level",
    "DGS10": "10-year Treasury constant maturity, percent",
    "DBAA": "Moody's seasoned Baa corporate yield, percent",
    "DCOILWTICO": "WTI spot, USD per barrel",
    "DTWEXM": "Nominal major-currencies USD index (discontinued 2019-12-31)",
    "DTWEXBGS": "Nominal broad USD index (from 2006)",
}


def fetch(series_id: str) -> dict[str, float]:
    resp = requests.get(FRED_CSV, params={"id": series_id}, timeout=90)
    resp.raise_for_status()
    rows = list(csv.reader(io.StringIO(resp.text)))[1:]
    return {
        r[0].strip(): float(r[1])
        for r in rows
        if len(r) > 1 and r[1].strip() not in ("", ".")
    }


def on_or_before(obs: dict[str, float], date: str) -> tuple[str, float]:
    """Latest observation at or before `date`.

    Episode boundaries are chosen from market history, and a given series may
    not quote on that exact day -- Columbus Day, a bond holiday, a stale
    publication. Stepping back to the previous quote is the honest resolution;
    the returned date says which observation was actually used.
    """
    candidates = [d for d in obs if d <= date]
    if not candidates:
        raise RuntimeError(f"no observation on or before {date}")
    key = max(candidates)
    return key, obs[key]


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
    data = {s: fetch(s) for s in SERIES}
    for s, obs in data.items():
        print(f"  {s:12s} {len(obs):6d} observations  {min(obs)} .. {max(obs)}")

    scenarios = []
    for name, start, end, description in EPISODES:
        usd_series = "DTWEXBGS" if start >= "2006-01-03" else "DTWEXM"
        provenance: dict[str, dict] = {}
        shocks: dict[str, float] = {}

        def record(factor, series, transform, s=start, e=end, p=provenance, sh=shocks):
            d0, v0 = on_or_before(data[series], s)
            d1, v1 = on_or_before(data[series], e)
            move = transform(v0, v1)
            sh[factor] = move
            p[factor] = {
                "fred_series": series,
                "from_date": d0,
                "from_level": v0,
                "to_date": d1,
                "to_level": v1,
                "move": move,
            }

        record("equity", "NASDAQCOM", lambda a, b: math.log(b / a))
        record("rates_level", "DGS10", lambda a, b: (b - a) / 100.0)
        record("oil", "DCOILWTICO", lambda a, b: math.log(b / a))
        record("usd", usd_series, lambda a, b: math.log(b / a))

        # credit_spread is Baa over the 10y, so it needs both series.
        b0d, b0 = on_or_before(data["DBAA"], start)
        b1d, b1 = on_or_before(data["DBAA"], end)
        t0d, t0 = on_or_before(data["DGS10"], start)
        t1d, t1 = on_or_before(data["DGS10"], end)
        spread_move = ((b1 - t1) - (b0 - t0)) / 100.0
        shocks["credit_spread"] = spread_move
        provenance["credit_spread"] = {
            "fred_series": "DBAA minus DGS10",
            "from_date": f"{b0d}/{t0d}",
            "from_level": round(b0 - t0, 4),
            "to_date": f"{b1d}/{t1d}",
            "to_level": round(b1 - t1, 4),
            "move": spread_move,
        }

        # rates_slope is not separately sourceable across all episodes: DGS30
        # was not published between 2002-02 and 2006-02, which straddles two of
        # them. Leaving it out applies a zero slope shock, which is explicit
        # here rather than silently imputed.
        entry = {
            "name": name,
            "type": "historical",
            "description": description,
            "window": {"start": start, "end": end},
            "factor_shocks": shocks,
            "provenance": provenance,
        }
        if name.startswith("1987"):
            entry["caveat"] = (
                "The equity shock is the Nasdaq Composite's -11.35% close. The "
                "S&P 500 fell 20.47% that session; Nasdaq quotes went stale as "
                "market makers stopped answering. This scenario understates a "
                "1987-style equity shock by roughly half."
            )
        scenarios.append(entry)

    # A small set of round hypotheticals, labelled as such. These are not
    # measurements of anything and are named so nobody mistakes them for one.
    scenarios += [
        {
            "name": "Hypothetical: equity -20%",
            "type": "hypothetical",
            "description": "Broad equity down 20% with all other factors held.",
            "factor_shocks": {"equity": math.log(0.80)},
        },
        {
            "name": "Hypothetical: rates +100bp parallel",
            "type": "hypothetical",
            "description": "Parallel 100bp rise in yields, spreads unchanged.",
            "factor_shocks": {"rates_level": 0.01},
        },
        {
            "name": "Hypothetical: credit +200bp",
            "type": "hypothetical",
            "description": "Baa spread over Treasuries widens 200bp.",
            "factor_shocks": {"credit_spread": 0.02},
        },
    ]

    out = {
        "_meta": {
            "generated_by": "scripts/build_stress_scenarios.py",
            "git_commit": git_sha(),
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "source": "FRED long-history daily series",
            "series_used": SERIES,
            "equity_proxy_note": (
                "NASDAQCOM is the equity proxy because S&P 500 and DJIA on FRED "
                "are limited to a rolling ten-year window."
            ),
        },
        "scenarios": scenarios,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'scenario':26s} {'equity':>8s} {'rates':>8s} {'credit':>8s} "
          f"{'usd':>8s} {'oil':>9s}")
    for s in scenarios:
        f = s["factor_shocks"]
        print(
            f"{s['name']:26s} "
            f"{f.get('equity', 0) * 100:7.2f}% "
            f"{f.get('rates_level', 0) * 1e4:7.0f}bp "
            f"{f.get('credit_spread', 0) * 1e4:6.0f}bp "
            f"{f.get('usd', 0) * 100:7.2f}% "
            f"{f.get('oil', 0) * 100:8.2f}%"
        )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
