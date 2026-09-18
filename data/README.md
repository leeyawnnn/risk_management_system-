# Data

Real daily market observations from FRED (Federal Reserve Bank of St. Louis).
Nothing in this directory is simulated.

Regenerate with:

```bash
python3 scripts/fetch_data.py --as-of 2026-09-17 --years 5
```

`manifest.json` records, for every series, the FRED identifier, the publisher,
the licence terms, a SHA256 of the exact payload that was downloaded, and every
date the alignment step dropped.

## Sample

| | |
|---|---|
| Source | FRED, `https://fred.stlouisfed.org/graph/fredgraph.csv` |
| Requested through | 2026-09-17 |
| Effective as-of | 2026-09-11 |
| Sample | 2021-09-17 to 2026-09-11 |
| Aligned observations | 1,239 dates, 1,238 daily returns |
| Instruments | 11 across equity, rates, credit, commodity, FX |
| Alignment | inner join on dates where every instrument quotes |

The as-of date is 2026-09-11, not 2026-09-17, because the Fed's H.10 exchange
rate release lags: `DTWEXBGS` and `DEXUSEU` had not published beyond the 11th
when this snapshot was taken. The book is only as current as its slowest
series, and pretending otherwise would date the whole report six days early.

## Instruments

| Code | Class | FRED series | Construction |
|---|---|---|---|
| `EQ_US_LARGE` | Equity | `SP500` | index level, log returns |
| `EQ_US_TECH` | Equity | `NASDAQ100` | index level, log returns |
| `EQ_US_BLUE` | Equity | `DJIA` | index level, log returns |
| `UST_2Y` | Rates | `DGS2` | rolling 2y par bond, total return |
| `UST_10Y` | Rates | `DGS10` | rolling 10y par bond, total return |
| `UST_30Y` | Rates | `DGS30` | rolling 30y par bond, total return |
| `CREDIT_AAA` | Credit | `DAAA` | rolling 20y par bond at the Aaa yield |
| `CREDIT_BAA` | Credit | `DBAA` | rolling 20y par bond at the Baa yield |
| `COMD_WTI` | Commodity | `DCOILWTICO` | WTI spot, log returns |
| `FX_USD_BROAD` | FX | `DTWEXBGS` | broad USD index, log returns (long USD) |
| `FX_EURUSD` | FX | `DEXUSEU` | EUR/USD spot, log returns (long EUR) |

Each file is `date,price`. For the yield-based instruments the `price` column
is a total return index starting at 100 on the first date, so every instrument
is loaded through the same code path.

## Turning yields into returns

A constant-maturity yield is not a return. `DGS10` moving from 4.00 to 4.05 is
five basis points of yield, not a five-basis-point loss, and the position that
yield describes lost roughly 40bp. The conversion in
[`scripts/fetch_data.py`](../scripts/fetch_data.py) is:

A CMT yield is a *par* yield, so a bond of that maturity paying that coupon
semiannually prices at exactly 100. Each day we buy that bond at par on
yesterday's yield, then reprice the same cashflows one day closer to maturity
at today's yield. The change in dirty price is the day's total return; accrued
coupon sits inside the dirty price, so carry needs no separate term.

This is exact bond maths for the stated convention, not a `-D·Δy` expansion, so
it stays right through the large yield moves in 2022 where a linear
approximation drifts. Sanity checks, all reproduced in the test suite:

- With the yield unchanged, a 4% 10y returns 1.57bp against `y/252` of 1.59bp —
  the gap is one day of discounting at a semiannual rate.
- For a 1bp move, the 30y result of −14.59bp matches `carry − D_mod·Δy` of
  −14.61bp at a par-bond modified duration of 16.37.
- For a −100bp move, the 30y returns +19.70% against a linear +17.38%. The
  +2.32% difference is convexity, and it is exactly the term a duration-only
  model throws away.

The Moody's Aaa/Baa series are seasoned long-corporate yields with no single
published maturity. Holding them as 20-year par bonds is a modelling choice,
stated here rather than buried: it sets the credit legs' duration, and a
shorter assumption would scale their volatility and risk contribution down
roughly in proportion.

## Missing data and the trading calendar

Across the window the eleven series between them quote on 1,261 dates, but all
eleven quote together on only 1,239. The 22 dropped dates are real calendar
structure, not corrupt data:

- **16 dates** where equities traded and the bond, FX or commodity series did
  not. Columbus Day (2021-10-11) and Veterans Day (2021-11-11) are SIFMA bond
  market holidays and Federal Reserve holidays but ordinary NYSE sessions.
  2021-11-26, the half day after Thanksgiving, has no EIA crude settle.
- **6 dates** with mixed availability, including the run at the end of the
  sample where H.10 exchange rates had not yet published.

The policy is an inner join: a date is kept only if every instrument quotes.
Forward-filling the gap is the tempting alternative and it is wrong here. A
filled day contributes a zero return for one instrument while the rest of the
book moves, which drags every correlation involving that instrument toward zero
and understates portfolio risk — the error lands precisely on the quantity the
engine exists to estimate. Dropping 22 of 1,261 dates costs 1.7% of the sample
and biases nothing.

## Known properties and limitations of this sample

- **The equity series are price indices.** `SP500`, `NASDAQ100` and `DJIA`
  exclude dividends. Over this window that understates their annualised return
  by roughly the index dividend yield. It leaves daily volatility and
  correlation essentially untouched, so the risk numbers are unaffected while
  the return and Sharpe columns are biased low. They are reported anyway rather
  than silently adjusted.
- **No single-name equity, no emerging markets, no gold.** FRED carries no
  current daily gold fixing, and single names are not FRED series. The book is
  index-level.
- **The credit legs are yield-based, not index-based.** ICE BofA total return
  indices are on FRED but licensing restricts them to a rolling three-year
  window, which would have cut the sample by 40%.
- **Just under five years.** 1,238 returns is a short sample for a 99% tail:
  about twelve observations sit beyond the threshold. The README quantifies
  what that does to the 99% VaR estimate rather than reporting a point estimate
  as if it were precise.

## Licensing

FRED redistributes series from several publishers, and their terms differ. This
repository commits derived daily level series, not the raw vendor payloads.

| Series | Publisher | Terms |
|---|---|---|
| `DGS2`, `DGS10`, `DGS30` | Federal Reserve Board (H.15) | US Government work, public domain |
| `DTWEXBGS`, `DEXUSEU` | Federal Reserve Board (H.10) | US Government work, public domain |
| `DCOILWTICO` | U.S. Energy Information Administration | US Government work, public domain |
| `SP500`, `DJIA` | S&P Dow Jones Indices LLC | Redistribution restricted; see the FRED series page |
| `NASDAQ100` | Nasdaq OMX Group | Redistribution restricted; see the FRED series page |
| `DAAA`, `DBAA` | Moody's | Redistribution restricted; see the FRED series page |

See [`NOTICE`](../NOTICE) at the repository root. Anyone needing the raw levels
under their own licence should re-run `scripts/fetch_data.py`.
