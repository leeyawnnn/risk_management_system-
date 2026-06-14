# Risk Report: Diversified Multi-Asset Book

- **Notional:** 25000000
- **Covariance estimator:** ledoit_wolf
- **Annualization factor:** 252
- **Portfolio volatility:** 0.56% daily / 8.89% annual

## Positions

| Asset | Weight |
|---|---:|
| SPY | 15.00% |
| QQQ | 10.00% |
| EFA | 8.00% |
| AAPL | 5.00% |
| MSFT | 5.00% |
| JPM | 4.00% |
| XOM | 4.00% |
| TLT | 12.00% |
| IEF | 12.00% |
| LQD | 8.00% |
| HYG | 6.00% |
| GLD | 8.00% |
| USO | 3.00% |

## Asset comparison

Annualized statistics per instrument, and how much portfolio risk each one actually contributes. Note where **% of risk diverges from weight** — that gap is the whole point of risk attribution.

| Asset | Sector | Weight | Ann. vol | Ann. return | Sharpe | Corr. to ptf | % of risk |
|---|---|---:|---:|---:|---:|---:|---:|
| SPY | Equity | 15.00% | 15.87% | 12.81% | 0.81 | 0.83 | 22.08% |
| QQQ | Equity | 10.00% | 19.14% | 21.22% | 1.11 | 0.80 | 17.15% |
| EFA | Equity | 8.00% | 15.71% | 8.26% | 0.53 | 0.77 | 10.78% |
| AAPL | Equity | 5.00% | 25.35% | 26.80% | 1.06 | 0.69 | 9.78% |
| MSFT | Equity | 5.00% | 23.80% | 27.96% | 1.17 | 0.71 | 9.52% |
| JPM | Equity | 4.00% | 25.68% | 28.50% | 1.11 | 0.54 | 6.26% |
| XOM | Equity | 4.00% | 30.42% | 18.31% | 0.60 | 0.45 | 6.22% |
| TLT | Rates | 12.00% | 17.06% | -3.63% | -0.21 | 0.29 | 6.78% |
| IEF | Rates | 12.00% | 7.79% | -0.20% | -0.03 | 0.28 | 2.94% |
| LQD | Credit | 8.00% | 7.11% | 0.86% | 0.12 | 0.30 | 1.90% |
| HYG | Credit | 6.00% | 9.69% | 2.99% | 0.31 | 0.45 | 2.97% |
| GLD | Commodity | 8.00% | 16.09% | 5.40% | 0.34 | 0.14 | 2.11% |
| USO | Commodity | 3.00% | 31.36% | 28.83% | 0.92 | 0.14 | 1.52% |

## Value at Risk & Expected Shortfall

| Method | Confidence | Horizon | VaR | VaR ($) | CVaR | CVaR ($) |
|---|---:|---:|---:|---:|---:|---:|
| Historical | 95% | 1d | 0.86% | 213968 | 1.18% | 294013 |
| Parametric | 95% | 1d | 0.88% | 220084 | 1.11% | 278615 |
| MonteCarlo | 95% | 1d | 0.87% | 218240 | 1.11% | 277816 |
| Historical | 95% | 10d | 2.71% | 676627 | 3.72% | 929750 |
| Parametric | 95% | 10d | 2.50% | 625425 | 3.24% | 810516 |
| MonteCarlo | 95% | 10d | 2.50% | 624698 | 3.25% | 811309 |
| Historical | 99% | 1d | 1.39% | 348419 | 1.66% | 414072 |
| Parametric | 99% | 1d | 1.26% | 315543 | 1.45% | 363009 |
| MonteCarlo | 99% | 1d | 1.26% | 314465 | 1.45% | 363386 |
| Historical | 99% | 10d | 4.41% | 1101796 | 5.24% | 1309412 |
| Parametric | 99% | 10d | 3.71% | 927294 | 4.31% | 1077395 |
| MonteCarlo | 99% | 10d | 3.71% | 926697 | 4.32% | 1079841 |

## Covariance estimator comparison

The same portfolio, priced with three different covariance estimators. Divergence here is *model risk*: the VaR number you report depends on how you estimated `Sigma`.

| Estimator | Daily vol | Annual vol | 95% 1d VaR | 99% 1d VaR |
|---|---:|---:|---:|---:|
| Sample | 0.56% | 8.90% | 0.88% | 1.26% |
| EWMA | 0.71% | 11.33% | 1.13% | 1.62% |
| Ledoit-Wolf | 0.56% | 8.89% | 0.88% | 1.26% |

## VaR backtest (in-sample)

How often realized daily returns actually breached the historical VaR. A well-calibrated model breaches at roughly `(1 - confidence)` of days; far more means VaR is too optimistic.

| Confidence | Obs | Expected breaches | Actual breaches | Actual rate | Verdict |
|---:|---:|---:|---:|---:|---|
| 95% | 1260 | 63.0 | 63 | 5.00% | OK |
| 99% | 1260 | 12.6 | 12 | 0.95% | OK |

## Risk attribution

| Asset | Weight | Marginal | Component | % of risk |
|---|---:|---:|---:|---:|
| SPY | 15.00% | 0.0082 | 0.0012 | 22.08% |
| QQQ | 10.00% | 0.0096 | 0.0010 | 17.15% |
| EFA | 8.00% | 0.0076 | 0.0006 | 10.78% |
| AAPL | 5.00% | 0.0110 | 0.0005 | 9.78% |
| MSFT | 5.00% | 0.0107 | 0.0005 | 9.52% |
| JPM | 4.00% | 0.0088 | 0.0004 | 6.26% |
| XOM | 4.00% | 0.0087 | 0.0003 | 6.22% |
| TLT | 12.00% | 0.0032 | 0.0004 | 6.78% |
| IEF | 12.00% | 0.0014 | 0.0002 | 2.94% |
| LQD | 8.00% | 0.0013 | 0.0001 | 1.90% |
| HYG | 6.00% | 0.0028 | 0.0002 | 2.97% |
| GLD | 8.00% | 0.0015 | 0.0001 | 2.11% |
| USO | 3.00% | 0.0028 | 0.0001 | 1.52% |
| **Total** |  |  | **0.0056** | **100.00%** |

### By sector

| Sector | % of risk |
|---|---:|
| Commodity | 3.63% |
| Credit | 4.87% |
| Equity | 81.78% |
| Rates | 9.72% |

**Concentration:** effective number of bets = 8.09; largest contributor = SPY (22.08%).

## Stress tests

| Scenario | P&L | P&L ($) |
|---|---:|---:|
| 2008 GFC | -16.92% | -4230000 |
| 2020 COVID Crash | -19.19% | -4797500 |
| 1987 Black Monday | -9.93% | -2483750 |
| Equity -20% | -10.70% | -2675000 |
| Credit Spreads +200bps | -1.18% | -294000 |
| USD +10% | -2.11% | -527500 |

## Figures

**Asset volatility comparison** — annualized vol of every instrument, coloured by sector.

![Asset volatility](figures/asset_volatility.svg)

**Weight vs. risk share** — capital allocation against actual risk contribution. Where the red (risk) bar dwarfs the grey (weight) bar, a small position is punching above its weight.

![Weight vs risk](figures/weight_vs_risk.svg)

**Correlation matrix** — red positive, blue negative; the off-diagonal structure is what makes the book more (or less) than the sum of its parts.

![Correlation matrix](figures/correlation.svg)

**Risk contribution by position** — each bar is a share of total portfolio volatility (bars sum to 100%).

![Risk contributions](figures/risk_contributions.svg)

**Return distribution** — daily portfolio returns with the VaR and CVaR thresholds marked.

![Return distribution](figures/return_distribution.svg)

**Estimator comparison** — 95% / 99% VaR under each covariance estimator (model risk at a glance).

![Estimator comparison](figures/estimator_comparison.svg)

**VaR backtest** — daily returns over time against the 95% VaR line; red dots are breaches.

![VaR backtest](figures/var_backtest.svg)

## Commentary — what to look at

- **Tail vs. routine risk.** Daily 99% VaR is on the order of a single bad session, but the worst stress scenario, **2020 COVID Crash**, costs -19.19% (-4797500) — an order of magnitude larger. VaR and stress answer different questions; report both.
- **Concentration.** The largest single risk contributor is **SPY** at 22.08% of portfolio risk; the effective number of bets is 8.1 across 13 holdings.
- **Weight != risk.** **IEF** is 12.00% of capital but 2.94% of risk — the clearest case where position size misleads.
- **Model risk.** The 95% VaR ranges from 0.88% to 1.13% across covariance estimators — the same book, a 29% spread purely from estimation choice.
