# FIXES

What was broken, what changed, which claims were retracted, and which numbers
moved. This file is a hand-off note, not permanent documentation: the parts
worth keeping are already folded into the README's Limitations section, and
this file should be deleted once read.

---

## 1. The analysis was circular

**Broken.** `scripts/generate_sample_data.py` built all 13 return series from a
Gaussian factor model with hand-written loadings. The README then presented
those loadings back as discoveries:

- "equities vs. Treasuries are negative (SPY/TLT −0.11), the stock-bond hedge"
  was `riskoff=-0.60` for SPY and `riskoff=+1.20` for TLT.
- "HYG sits between the blocks — high yield is half credit, half equity" was
  `dict(rates=-0.30, credit=-1.30, eq=0.25, riskoff=-0.50)`.
- "SPY/EFA +0.85, SPY/QQQ +0.79" was `eq=1.00` on three series.

The generator's own docstring said "which reproduces the stock/bond hedge". The
data was also described as generated exactly once, on line 52, mid-sentence,
after which five sections narrated market structure as fact.

**Changed.** Path A, with a substitution. The prompt recommended yfinance; it is
unreachable from here — every Yahoo endpoint and host returns HTTP 429, with
and without the cookie/crumb flow, and Stooq is behind a JavaScript challenge.
I checked with you before choosing and we settled on FRED.

The universe is now 11 instruments from real daily FRED series across equity,
rates, credit, commodity and FX, 2021-09-17 to 2026-09-11, 1,239 aligned dates
and 1,238 returns. `scripts/generate_sample_data.py` is deleted.

Real data forced three things the synthetic set hid, all written up in
`data/README.md`:

- **Yields are not returns.** Treasury and Moody's corporate legs are converted
  to the total return of a rolling constant-maturity par bond. Exact bond
  maths, not `−D·Δy`: on a −100bp move a 30-year returns +19.70% against a
  linear +17.38%, and the 2.32% difference is convexity.
- **Calendar mismatch.** The eleven series quote on 1,261 dates between them
  and on only 1,239 together. Columbus Day and Veterans Day are bond and
  Federal Reserve holidays but ordinary NYSE sessions; the H.10 FX release lags
  several days. The policy is an inner join, because forward-filling injects a
  zero return for one instrument while the book moves and biases exactly the
  correlations the engine exists to estimate.
- **Licensing.** FRED aggregates publishers with different terms. `NOTICE` and
  `data/README.md` now record which series are US Government public domain and
  which are restricted.

**Retracted.** Every sentence narrating market structure as a finding. Note
what the real data says instead: the equity-Treasury correlation over this
sample is **+0.02 to +0.08**, not the negative stock-bond hedge the old README
claimed. That regime ended in 2022.

## 2. A factually wrong claim about fat tails

**Broken.** The README said that at 99% "the historical VaR pulls ahead of both
(1.39% vs 1.26%) because the real return distribution has fatter tails than the
Gaussian assumes." The generator used `rng.standard_normal()` for both the
factor returns and the idiosyncratic term. Excess kurtosis was zero by
construction. The 13bp gap was sampling noise in roughly the thirteenth-worst
observation of 1,260.

**Changed.** The sentence is deleted. `apps/tail_study.cpp` replaces it with two
measurements.

Bootstrapping the historical estimator 10,000 times:

| Measure | Conf | Historical | 95% interval | Parametric | Inside? |
|---|---:|---:|---|---:|---|
| VaR | 95% | 0.842% | [0.732%, 0.923%] | 0.801% | yes |
| CVaR | 95% | 1.146% | [1.030%, 1.268%] | 1.008% | **no** |
| VaR | 99% | 1.288% | [1.108%, 1.522%] | 1.138% | yes |
| CVaR | 99% | 1.717% | [1.397%, 2.043%] | 1.305% | **no** |

So the VaR gap is not distinguishable from noise, exactly as the prompt
predicted, and the CVaR gap is. Fat tails bite the tail mean before the tail
quantile. The real sample does have fat tails — measured excess kurtosis
**4.13** — which is why this is now a demonstrable result instead of a false
one.

The controlled Student-t experiment is promoted from a unit test to a
documented experiment. At matched variance, against a t(4) evaluated on
5,000,000 draws:

| Conf | Measure | Gaussian model | True t tail | Gaussian understates by |
|---:|---|---:|---:|---:|
| 95% | VaR | 0.802% | 0.735% | −8.5% |
| 95% | CVaR | 1.009% | 1.110% | +10.0% |
| 99% | VaR | 1.140% | 1.302% | +14.2% |
| 99% | CVaR | 1.308% | 1.817% | +39.0% |

The negative first row is the interesting one: at 95% the Gaussian is
*conservative*, because matching variance moves mass out of the shoulders into
the tails. Fat tails cross over, and "parametric underestimates the tail" is
only true far enough out.

**A mistake I made and corrected.** My first version of this comparison
estimated the parametric σ from the same 1,238-day fat-tailed draw. A sample
standard deviation computed over fat tails is dominated by its outliers, so the
parametric number came out inflated and the comparison showed the wrong sign.
Evaluating the reference tail on 5,000,000 draws fixed it.

## 3. Missing risk machinery

All added, all tested.

**VaR backtesting** (`src/backtest.cpp`). Kupiec proportion-of-failures,
Christoffersen independence and conditional coverage, Basel traffic-light
zones. Run for all three methods at both confidences. Basel zones are derived
from the binomial CDF rather than hardcoded, and a test walks every boundary to
confirm the derivation reproduces the published table (green 0–4, yellow 5–9,
red 10+ at 250 days and 99%).

**The result is worth reading.** Every method passes Kupiec and every method
fails Christoffersen. Exception counts are right; their timing is not. That is
the textbook demonstration of why the independence test exists, and it is a
real failure of this model that I have measured and not fixed — see
Limitations.

**ES backtesting.** Acerbi-Szekely Z1 and Z2 with a simulated null, plus the
explanation of why ES needs them: it is not elicitable, so it cannot be
backtested by counting hits. Parametric ES at 99% gives Z2 = −0.797, p = 0.005.

**Risk decomposition** (`src/decomposition.cpp`). Marginal VaR, component VaR
with the Euler residual printed (−1.7e-18), incremental VaR by full revaluation
against the linearisation, and factor-vs-specific decomposition with the model
error reported (0.9% of sample variance).

**Stress testing.** Historical scenarios are now measured from FRED
long-history series over named windows, with the series, both dates and both
levels recorded per factor. The 1987 entry carries a caveat: its equity shock
is the Nasdaq Composite's −11.35% close, and the S&P fell 20.47% that session,
because Nasdaq quotes went stale as market makers stopped answering. I use the
Nasdaq because FRED restricts S&P 500 and DJIA to a rolling ten-year window.

Reverse stress solves for the most likely factor move reaching a target loss
and reports the Mahalanobis distance.

## 4. Repository hygiene

- 28 tracked `.cache/clangd/index/*.idx` files removed.
- `output/` untracked. It was byte-identical to `reports/figures/`, so every
  figure was committed twice. `reports/sample_report.*` also removed, for the
  same reason.
- `.gitattributes` added so GitHub reports C++ rather than SVG.
- Eigen, nlohmann/json and Catch2 pinned to release tarballs with SHA256s I
  verified by downloading them. An offline mode requires system copies.
- Tracked content is 1.2 MB.

## 5. Figures

All rebuilt against `src/plot_style.cpp`, which is now the only thing that
decides how a chart looks. Specific changes the prompt asked for: hierarchical
clustering on the correlation matrix, the dumbbell sort on weight-vs-risk, the
1/N reference line and effective-number-of-bets annotation, the fitted Gaussian
overlay, bootstrap error bars on volatility, the estimator-error figure rebuilt
around the ground-truth study, and the backtest figure with statistics attached
instead of a bare picture of breaches.

Figure titles are computed from the data. The distribution chart claims a heavy
tail only when the measured excess kurtosis supports it, and the backtest title
reports whichever verdict the tests returned. Tests pin both, because a caption
that outlived its data is the defect this whole exercise is about.

## Additional findings, not in the prompt

1. **`std::normal_distribution` is not portable.** Its algorithm is
   implementation-defined, so libstdc++ and libc++ produce different draws from
   the same seed. Every Monte Carlo number would have differed between a local
   macOS run and CI on Linux, making the committed report uncheckable. Replaced
   with an inverse-CDF normal over the engine's raw bits
   (`include/risk/random.hpp`). Same for `uniform_int_distribution` and
   `student_t_distribution`.
2. **A bug in that replacement, caught by its own test.** The natural offset
   `((bits >> 11) + 0.5) * 2^-53` returns exactly 1.0 at the top of the range,
   because `2^53 − 0.5` is not representable and rounds up, and `normal_ppf(1)`
   throws. Fixed by using 52 bits.
3. **`find_package(Eigen3)` resolved to a different project's build tree.**
   Eigen's config registers itself in `~/.cmake/packages`, so the "system copy"
   probe found `../fixed_income_engine/build/_deps/eigen-build` on this
   machine. Fixed with `NO_CMAKE_PACKAGE_REGISTRY`.
4. **`-Wshadow` found a real shadow** in `to_json`: the asset-stats loop bound
   `a`, shadowing the `a` already bound to `rep.attribution` in the same scope.
5. **The credit factor was defined wrongly at first.** Defining it as Baa − Aaa
   puts the Aaa leg's own yield change inside the factor and handed
   `CREDIT_AAA` a spurious *positive* credit beta. Redefined as Baa over the
   10-year Treasury, which is FRED's own `BAA10Y` definition.
6. **The stress tests were coupled to the shipped config.** They asserted
   SPY's 1987 shock was −0.205 and GLD's USD beta was −0.60, so re-measuring an
   episode looked like a regression. Moved to fixtures; the shipped config gets
   a structural test instead, including that every scenario only shocks factors
   some instrument actually loads on.
7. **The README said 54 tests. There were 56.** There are now 127.
8. **`version.hpp` said `0.0-phase0-skeleton`** and `version.cpp` described
   itself as a placeholder "so risk_core has something to compile in Phase 0".
9. **The old backtest verdict was arbitrary.** It printed "OK" when the
   exception rate was within 1.5× of expected. That threshold has no basis;
   Kupiec does.

## Where the prompt and the repository disagreed

- **§0.4 asked to prefer a GitHub-hosted mirror for Eigen.** There is not one.
  Eigen is GitLab-hosted and the `github.com/eigenteam` mirror was archived at
  3.3.7. The pinned URL is the canonical GitLab archive, which is why the
  offline mode matters; CI proves the offline path works with apt-installed
  Eigen.
- **§1.2 asked for estimator error against a ground truth known analytically
  from the generator's loadings.** With the generator deleted there is no such
  analytic Σ. `apps/estimator_study.cpp` instead builds the ground truth from
  the factor model *fitted to the real sample*, which keeps a real book's
  eigenvalue structure. The result is the same kind of evidence and arguably
  better calibrated.
- **§1.2 also expected shrinkage to look good.** It does not, on accuracy:
  Ledoit-Wolf and the sample estimator are indistinguishable on Frobenius
  error, volatility error and VaR error at every sample size. What shrinkage
  buys here is conditioning — at n=60 it cuts the correlation-matrix condition
  number from 399 to 35. Reported as found.

## Numbers that moved, and why

| Quantity | Old README | Now | Why |
|---|---:|---:|---|
| Instruments | 13 | 11 | Real FRED universe replaced the synthetic tickers |
| Observations | 1,260 | 1,238 | Real trading calendar, inner-joined |
| Portfolio vol (annual) | 8.89% | 7.85% | Different book, real data |
| Historical 99% VaR | 1.39% | 1.288% | Different book, real data |
| Parametric 99% VaR | 1.26% | 1.138% | Different book, real data |
| Excess kurtosis | 0 by construction | 4.13 measured | Real returns have tails |
| Equity/Treasury correlation | −0.11 (asserted) | +0.02 to +0.08 | Post-2022 regime, measured |
| Effective number of bets | 8.1 | 4.9 | Real correlations are higher |
| Tests | 54 (claimed), 56 (actual) | 127 | Backtests, decomposition, figures, RNG |

## Not done

- The Christoffersen failure is measured, not fixed. A conditional volatility
  model (GARCH, or a rolling EWMA-driven VaR) is the right answer.
- The bootstrap is i.i.d. and the data is not, so the intervals are too narrow.
  A stationary block bootstrap would be correct.
- No liquidity-adjusted VaR, no FX translation on the non-FX legs, no P&L
  attribution against realised returns, no intraday risk.

All of these are in the README's Limitations section in first person.
