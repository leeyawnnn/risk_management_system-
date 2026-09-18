# Risk Management Engine

A multi-asset portfolio risk engine in C++20: Value at Risk three ways,
Expected Shortfall, risk attribution, stress testing, and the regulatory
backtests that say whether any of it is calibrated.

Worked example: 11 instruments built from real daily FRED series across
equity, rates, credit, commodity and FX, 2021-09-17 to **2026-09-11**,
1,238 daily returns. Nothing in this repository is simulated market data.
Every number below was produced by `./scripts/build_reports.sh` and is
reproduced in [`reports/report.json`](reports/report.json).

## Results

$25M book, Ledoit-Wolf covariance, 1-day horizon. Portfolio volatility
**0.494% daily / 7.85% annual**. Return skewness 0.02, excess kurtosis **4.13**.

| Method | Conf | VaR | VaR ($) | CVaR | CVaR ($) | Kupiec | Christoffersen | Basel |
|---|---:|---:|---:|---:|---:|---|---|---|
| Historical | 95% | 0.842% | 210,418 | 1.146% | 286,543 | pass (p=0.91) | **reject** (p=0.0001) | green |
| Parametric | 95% | 0.801% | 200,238 | 1.008% | 251,885 | pass (p=0.60) | **reject** (p=0.0005) | green |
| Monte Carlo | 95% | 0.802% | 200,618 | 1.006% | 251,488 | pass (p=0.60) | **reject** (p=0.0005) | green |
| Historical | 99% | 1.288% | 322,077 | 1.717% | 429,367 | pass (p=0.91) | **reject** (p=0.004) | green |
| Parametric | 99% | 1.138% | 284,469 | 1.305% | 326,353 | pass (p=0.08) | **reject** (p=0.032) | green |
| Monte Carlo | 99% | 1.132% | 282,922 | 1.293% | 323,281 | pass (p=0.08) | **reject** (p=0.032) | green |

Five findings, each of which is a measurement rather than a claim:

1. **Every method passes Kupiec and every method fails Christoffersen.** The
   exception *counts* are right — 61 breaches against an expected 61.9 at 95%,
   12 against 12.4 at 99%. Their *timing* is not: exceptions cluster, at
   p=0.0001. Counting hits is the easy half of backtesting and it is the half
   that says nothing about whether you survive the bad week.
2. **The VaR gap between historical and parametric is noise; the CVaR gap is
   not.** Bootstrapping the historical estimator 10,000 times puts the 99% VaR
   at 1.288% with a 95% interval of [1.108%, 1.522%] — the parametric 1.138%
   sits comfortably inside. The 99% CVaR interval is [1.397%, 2.043%] and the
   parametric 1.305% falls **outside** it. Fat tails show up in the tail mean
   before the tail quantile.
3. **Expected Shortfall is where the Gaussian model actually breaks.** The
   Acerbi-Szekely Z2 statistic for parametric ES at 99% is −0.797 (p=0.005).
   Negative means realised tail losses were worse than the model said.
4. **Shrinkage buys conditioning, not accuracy.** Against a known covariance,
   Ledoit-Wolf and the sample estimator are indistinguishable on error, but at
   n=60 shrinkage cuts the correlation-matrix condition number from 399 to 35.
5. **EWMA's error does not improve with more data.** At λ=0.94 its effective
   sample is roughly 33 observations regardless of how much history you hand
   it. Its 99% VaR error stays near 11bp at every sample size from 60 to 1,260
   days while the other two converge below 2bp.

## Quickstart

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure    # 122 tests

./scripts/build_reports.sh                    # regenerates everything in reports/
```

Dependencies (Eigen 3.4.0, nlohmann/json 3.12.0, Catch2 3.8.1) are fetched
from pinned release tarballs verified against a SHA256. On a network that
blocks them, install the three from a package manager and configure
`-DRISK_OFFLINE=ON`, which requires system copies and never touches the
network. CI exercises both paths.

To refresh the data from source:

```bash
python3 scripts/fetch_data.py --as-of 2026-09-17 --years 5
python3 scripts/estimate_factor_betas.py
python3 scripts/build_stress_scenarios.py
```

## What this is

Value at Risk answers one question: over the next day, with confidence α, what
is the most I expect to lose? It is a single comparable number across desks,
which is why risk committees and regulators live on it, and its weakness is
that it says nothing about how bad the worst α% gets. Expected Shortfall — the
mean loss beyond VaR — answers that, and is why the Basel FRTB framework moved
to it.

Neither number means anything until someone checks it. A VaR model that is
quietly 30% too low still prints a number every morning, and the only thing
that catches it is a backtest. This engine computes VaR three ways and then
subjects each to the tests a supervisor would: Kupiec for whether the exception
rate is right, Christoffersen for whether exceptions cluster, Basel's
traffic-light zones for the supervisory verdict, and Acerbi-Szekely for ES,
which cannot be backtested by counting hits because it is not elicitable.

The estimate that matters most is not the VaR formula but the covariance matrix
underneath it. This book's correlation matrix has a condition number of 189
under shrinkage and 378 without, because three US equity indices and three
points on one yield curve are not six independent things. Anything that
inverts Σ — an optimiser, a Cholesky factor, a Mahalanobis distance — amplifies
estimation error in the smallest eigenvalue by that factor. That is the
concrete reason shrinkage estimators exist, and this repository measures the
effect rather than asserting it.

The data is real, which forces the problems synthetic data hides: bond markets
close on Columbus Day while equities trade, the Fed's exchange-rate release
lags by several days, and a constant-maturity yield is not a return. Those are
handled explicitly and written up in [`data/README.md`](data/README.md) rather
than smoothed over.

## Method

| Module | Responsibility |
|---|---|
| [`return_series`](src/return_series.cpp) | CSV to returns; explicit missing-data policy; annualization |
| [`covariance`](src/covariance.cpp) | Sample / EWMA / Ledoit-Wolf; PSD and conditioning diagnostics |
| [`portfolio`](src/portfolio.cpp) | `wᵀΣw`, marginal and component volatility |
| [`var`](src/var.cpp) / [`cvar`](src/cvar.cpp) | Three VaR methods and Expected Shortfall |
| [`backtest`](src/backtest.cpp) | Kupiec, Christoffersen, Basel zones, Acerbi-Szekely |
| [`decomposition`](src/decomposition.cpp) | Marginal/component/incremental VaR, factor risk |
| [`stress_test`](src/stress_test.cpp) | Scenario P&L and reverse stress |
| [`attribution`](src/attribution.cpp) | Risk contributions and concentration |
| [`plot_style`](src/plot_style.cpp) / [`figures`](src/figures.cpp) | SVG charts, no plotting dependency |
| [`reporter`](src/reporter.cpp) | Assembles `report.json` and `report.md` |

Three points worth stating explicitly.

**Yields are not returns.** `DGS10` moving from 4.00 to 4.05 is five basis
points of yield, and the position it describes lost about 40bp. Each rate and
credit instrument is held as a rolling constant-maturity par bond: buy at par
on yesterday's quoted yield, reprice the same cashflows one day closer at
today's. That is exact bond maths, not a `−D·Δy` expansion, which matters
through 2022 — on a −100bp move a 30-year returns +19.70% against a linear
+17.38%, and the 2.32% difference is convexity.

**Component VaR sums to total VaR exactly.** VaR is homogeneous of degree 1 in
the weight vector, so Euler's theorem gives an exact additive split. The report
prints the residual (−1.7e-18) rather than asserting the identity.

**Reverse stress is the more useful direction.** Forward stress asks what a
scenario costs. Reverse stress asks what it would take to lose a given amount,
solving `min f'Σ_F⁻¹f` subject to `b'f = −L` for the most likely factor move
that gets there, and reporting the answer as a distance in standard deviations.

## Figures

All nine regenerate from `./scripts/build_reports.sh`.

**Correlations cluster into blocks.** Rows and columns are ordered by
average-linkage clustering on `1 − ρ`, so the structure emerges from the data
rather than being pointed out in a caption. Read it as three blocks: equities
at 0.77–0.94, the rates and credit complex at 0.53–0.93, and EUR/USD against
the dollar index at −0.86. Note what is *not* there: the equity-Treasury
correlation is +0.02 to +0.08 over this sample, not the negative stock-bond
hedge of the pre-2022 regime.

![Correlation matrix](reports/figures/correlation.svg)

**Capital share and risk share come apart.** Each rule spans a position's share
of capital and its share of volatility, sorted by the gap. Read the top and
bottom rows: `EQ_US_LARGE` is 18% of capital and 33.7% of risk, `UST_2Y` is 15%
of capital and 1.3% of risk. This is the entire argument for risk-based rather
than capital-based position limits.

![Capital weight against risk share](reports/figures/weight_vs_risk.svg)

**The exceptions cluster.** Daily returns against the 95% VaR line with
breaches marked, the Basel supervisory window shaded, and the test statistics
printed on the chart. Read the box, not the dots: Kupiec passes at p=0.91 and
Christoffersen rejects at p<0.001. A backtest figure without its statistic is
decoration.

![VaR backtest with Kupiec and Christoffersen](reports/figures/var_backtest.svg)

**More history helps two estimators and not the third.** Mean absolute 99% VaR
error against a known covariance, log-log, with Monte Carlo bands. Sample and
Ledoit-Wolf fall from 8.7bp to 1.9bp as the sample grows; EWMA sits flat near
11bp. That flat line is the finding: exponential weighting caps its own
effective sample size.

![Estimator error against sample size](reports/figures/estimator_error.svg)

**Shrinkage lifts the small eigenvalues past the truth.** Mean estimated
eigenvalue against the known one at n=60, log scale, ordered smallest to
largest. Read the left-hand end: the sample estimator sits slightly below the
truth and Ledoit-Wolf sits 3.4× above it. That overshoot *is* the mechanism —
shrinkage compresses the spectrum toward its average, which is what buys the
conditioning improvement and what costs accuracy on any individual eigenvalue.

![Estimated eigenvalue spectrum against the truth](reports/figures/eigenvalue_spectrum.svg)

**The loss tail is heavier than a normal of the same variance.** Daily returns
against the fitted normal density, with VaR and CVaR marked and breaching bins
shaded. The peak is taller and the left tail fatter than the fitted curve —
excess kurtosis 4.13. This is what makes the ES backtest fail while the VaR
backtest passes.

![Return distribution against the fitted normal](reports/figures/return_distribution.svg)

**Volatility is an estimate.** Annualized volatility per instrument with 95%
bootstrap intervals, sorted and coloured by asset class. Read the error bars:
crude runs at 41% and 2-year Treasuries at 2.1%, a 19-fold range, and even on
1,238 observations the intervals are several points wide for the volatile
names.

![Instrument volatility with bootstrap intervals](reports/figures/asset_volatility.svg)

**A few positions carry the volatility.** Component contribution per position
with a reference line at 1/N. Read the gap between the top bar and the
reference: the effective number of bets is 4.9 across 11 holdings, so the book
is roughly half as diversified as its position count suggests.

![Risk contribution by position](reports/figures/risk_contributions.svg)

**Equity carries most of the variance.** Portfolio variance split across the
six estimated factors plus specific risk; equity alone accounts for one bar
(68.8%) and the rates level for the next (23.8%). Read the bottom bar: specific
risk is 1.4%, and the factor model reproduces the sample variance to within
0.9%, so almost nothing in this book is idiosyncratic.

![Factor risk decomposition](reports/figures/factor_risk.svg)

## Validation

**Against closed forms.** The chi-square survival function is checked against
`exp(−x/2)` for two degrees of freedom and `erfc(√(x/2))` for one. Monte Carlo
VaR is checked against the analytic Gaussian quantile. Parametric ES is checked
against `σ·φ(z)/(1−α)`.

**Against third-party implementations.** Sample covariance is pinned to
`numpy.cov` at 1e-12 relative. EWMA and Ledoit-Wolf are pinned to the reference
implementation in [`scripts/covariance_reference.py`](scripts/covariance_reference.py).

**Against the source paper.** The Ledoit-Wolf test recomputes π̂, ρ̂ and γ̂ from
their definitions and confirms they reproduce the returned shrinkage intensity,
so the test covers the formula and not just the code path. Verified against the
paper text: Ledoit & Wolf (2004), estimator at equation (2), intensity at
equation (5) of Appendix B.

**Against a known ground truth.** The estimator study simulates from a
covariance built from the factor model fitted to the real sample, so estimation
error is observable rather than arguable. The fitted rate betas recover the
instruments' analytic par-bond durations without being told them — `UST_30Y`
regresses to −17.16 against a modified duration of 16.37.

**Against the supervisory table.** Basel zones are derived from the binomial
CDF rather than hardcoded, and a test walks every boundary at the regulatory
250-day window to confirm the derivation reproduces green 0–4, yellow 5–9,
red 10+.

**Identities, with residuals printed.** Component volatility sums to portfolio
volatility; component VaR sums to total VaR (residual −1.7e-18); factor plus
specific variance sums to model variance; the incremental-VaR linearisation
converges to full revaluation as the trade shrinks. Marginal VaR is checked
against a central difference of the VaR function.

122 tests, run on Ubuntu and macOS under gcc and clang in Debug and Release,
plus address/undefined and thread sanitizer legs.

## Limitations

Written in first person, because these are my choices and my gaps.

- **Linear positions only.** There are no options in this book, so Monte Carlo
  has nothing to do that the parametric method cannot — a linear combination of
  multivariate-normal draws is univariate normal, and the two agree to
  simulation error by construction. Delta-normal VaR is exact here rather than
  an approximation. The Monte Carlo path exists so the machinery is in place,
  not because it earns its keep on this portfolio, and I would not present its
  agreement with the parametric number as validation of anything but the RNG.
- **The independence test fails and I have not fixed the model.** Every method
  rejects Christoffersen. The honest reading is that a constant-volatility
  model is wrong for daily returns, and the fix is a conditional
  volatility model — GARCH or an EWMA-driven VaR that updates daily rather than
  a single covariance over the whole sample. I have measured the problem and
  not solved it.
- **Single period, no liquidity horizon.** Everything is one-day or
  √time-scaled to ten. There is no liquidity-adjusted VaR, no bid-ask or
  market-impact haircut, and no position-size-dependent unwind assumption. For
  a book holding 41%-volatility crude that is a real omission.
- **The sample is short for the 99% tail.** 1,238 observations put about twelve
  returns beyond the 99% threshold, which is why the bootstrap interval on the
  99% VaR spans 1.10% to 1.52%. I report the interval rather than the point
  estimate alone, but no amount of reporting fixes twelve observations.
- **The bootstrap assumes independence and the data is not.** The i.i.d.
  bootstrap ignores volatility clustering, which the Christoffersen test says is
  present. The intervals above are therefore too narrow, not too wide. A
  stationary block bootstrap would be the right tool and I have not implemented
  one.
- **No FX translation risk despite two FX lines.** `FX_EURUSD` and
  `FX_USD_BROAD` are held as return series, not as currency exposure on the
  other ten positions. A real multi-currency book would need every position's
  P&L translated, and this one does not do that.
- **The equity series are price indices.** `SP500`, `NASDAQ100` and `DJIA`
  exclude dividends, so the return and Sharpe columns are biased low by roughly
  the index dividend yield. Daily volatility and correlation are essentially
  unaffected, so the risk numbers are not.
- **The credit legs carry an assumed duration.** Moody's seasoned Aaa/Baa
  yields have no single published maturity; I hold them as 20-year par bonds.
  That choice sets their volatility and their risk contribution, and a shorter
  assumption would scale both down roughly in proportion.
- **Reverse stress probabilities are not forecasts.** A 5% one-day loss is 10.1
  standard deviations of the book's factor distribution, and the Gaussian
  probability attached to that is meaningless given measured excess kurtosis of
  4.13. The distance is the useful output; the probability is there to show how
  badly the Gaussian assumption fails that far out.
- **No P&L attribution against realised returns, and no intraday risk.** The
  engine explains the risk of a static book. It does not reconcile predicted to
  realised P&L, which is the other half of what a risk desk does daily.
- **Three factor betas are definitional.** The equity, USD and oil factors are
  the return series of three book constituents, so those three instruments
  regress on themselves and come back with unit beta. The output flags them
  `self_referential`; the informative loadings are the cross-asset ones.
- **Hardware-dependent numbers: none are reported.** Nothing in this README is
  a timing or a throughput figure. The Monte Carlo results depend on the seed,
  which is stamped into every artifact.

## Data sources

| Series | Publisher | Used for | Terms |
|---|---|---|---|
| `SP500`, `DJIA` | S&P Dow Jones Indices | Equity indices | Redistribution restricted |
| `NASDAQ100`, `NASDAQCOM` | Nasdaq OMX | Equity indices, stress history | Redistribution restricted |
| `DGS2`, `DGS10`, `DGS30` | Federal Reserve Board (H.15) | Treasury par yields | Public domain |
| `DAAA`, `DBAA` | Moody's | Corporate yields | Redistribution restricted |
| `DCOILWTICO` | U.S. EIA | WTI crude | Public domain |
| `DTWEXBGS`, `DTWEXM`, `DEXUSEU` | Federal Reserve Board (H.10) | FX | Public domain |

All retrieved from FRED, `https://fred.stlouisfed.org/`. As-of **2026-09-11**,
limited by the H.10 publication lag. Per-series identifiers, payload SHA256s and
every date dropped by the calendar alignment are in
[`data/manifest.json`](data/manifest.json); licence detail is in
[`NOTICE`](NOTICE) and [`data/README.md`](data/README.md). Refresh with
`python3 scripts/fetch_data.py`.

## References

- Philippe Jorion, *Value at Risk: The New Benchmark for Managing Financial
  Risk*, 3rd ed., McGraw-Hill, 2006.
- Paul Kupiec, "Techniques for Verifying the Accuracy of Risk Measurement
  Models", *Journal of Derivatives* 3(2), 1995, 73–84.
- Peter Christoffersen, "Evaluating Interval Forecasts", *International
  Economic Review* 39(4), 1998, 841–862.
- Olivier Ledoit and Michael Wolf, "Honey, I Shrunk the Sample Covariance
  Matrix", *Journal of Portfolio Management* 30(4), 2004, 110–119. Estimator at
  equation (2), shrinkage intensity at equation (5) of Appendix B.
- Carlo Acerbi and Balazs Szekely, "Back-testing Expected Shortfall", *Risk*,
  December 2014.
- J.P. Morgan/Reuters, *RiskMetrics — Technical Document*, 4th ed., 1996.
  EWMA with λ=0.94 and the square-root-of-time rule.
- Basel Committee on Banking Supervision, *Supervisory Framework for the Use of
  "Backtesting" in Conjunction with the Internal Models Approach to Market Risk
  Capital Requirements*, 1996. The traffic-light zones.
- Basel Committee on Banking Supervision, *Minimum Capital Requirements for
  Market Risk* (FRTB), 2019. The VaR to Expected Shortfall shift.

## Licence

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
