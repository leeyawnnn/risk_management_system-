#!/usr/bin/env python3
"""Generate a realistic multi-asset-class daily price universe for the worked
example. ONE-TIME generator, not a runtime dependency.

Returns are built from a linear FACTOR MODEL so the cross-asset correlation
structure is realistic and the covariance is guaranteed PSD:

    r_asset(t) = sum_f beta[asset][f] * factor_f(t) + idio[asset] * z(t)

Factors (broad equity, tech, financials, energy, the rates level, credit
spreads, oil, gold, USD) are independent daily normals scaled from annual vols.
Bond assets load NEGATIVELY on the rates factor (price falls when yields rise),
which reproduces the stock/bond hedge and the credit/duration split.

Universe (13 instruments):
  Equities:    SPY QQQ EFA      Single names: AAPL MSFT JPM XOM
  Rates:       TLT IEF          Credit:       LQD HYG
  Commodities: GLD USO

Output: data/returns/<TICKER>.csv  (date,price), ~1260 trading days (~5y).
Seed fixed for reproducibility.
"""
import numpy as np
from datetime import date, timedelta
import os

SEED = 20240101
N_DAYS = 1260
ANNUAL = 252.0
rng = np.random.default_rng(SEED)

# Factor annualized vols. `riskoff` is a flight-to-quality factor: equities and
# credit fall while Treasuries and gold rally, which creates the realistic
# NEGATIVE stock/bond correlation.
factor_vol = {
    "eq": 0.15, "tech": 0.10, "fin": 0.11, "energy": 0.18,
    "rates": 0.07, "credit": 0.05, "oil": 0.30, "gold": 0.14, "usd": 0.07,
    "riskoff": 0.07,
}
factors = list(factor_vol.keys())

# Per-asset factor loadings, idiosyncratic annual vol, annual drift, start price.
# beta dict omits zero loadings.
assets = {
    "SPY":  (dict(eq=1.00, riskoff=-0.60),                          0.04, 0.09, 400.0),
    "QQQ":  (dict(eq=1.00, tech=0.90, riskoff=-0.70),               0.06, 0.12, 350.0),
    "EFA":  (dict(eq=0.90, usd=-0.35, riskoff=-0.55),               0.07, 0.06,  75.0),
    "AAPL": (dict(eq=1.05, tech=1.15, riskoff=-0.60),               0.16, 0.15, 180.0),
    "MSFT": (dict(eq=1.00, tech=1.05, riskoff=-0.55),               0.14, 0.14, 330.0),
    "JPM":  (dict(eq=1.00, fin=1.20, rates=0.30, riskoff=-0.70),    0.15, 0.10, 150.0),
    "XOM":  (dict(eq=0.80, energy=1.20, oil=0.35, riskoff=-0.40),   0.15, 0.08, 105.0),
    "TLT":  (dict(rates=-2.00, riskoff=1.20),                       0.03, 0.01,  95.0),
    "IEF":  (dict(rates=-0.90, riskoff=0.55),                       0.02, 0.02, 100.0),
    "LQD":  (dict(rates=-0.80, credit=-0.60, riskoff=0.20),         0.03, 0.03, 110.0),
    "HYG":  (dict(rates=-0.30, credit=-1.30, eq=0.25, riskoff=-0.50), 0.05, 0.05,  80.0),
    "GLD":  (dict(gold=1.00, usd=-0.40, riskoff=0.45),              0.05, 0.05, 180.0),
    "USO":  (dict(oil=1.00, energy=0.25, riskoff=-0.20),            0.08, 0.04,  70.0),
}

# Daily factor returns: (N_DAYS, n_factors).
fvol_daily = np.array([factor_vol[f] for f in factors]) / np.sqrt(ANNUAL)
F = rng.standard_normal((N_DAYS, len(factors))) * fvol_daily

names = list(assets.keys())
log_rets = np.zeros((N_DAYS, len(names)))
for j, name in enumerate(names):
    betas, idio_vol, drift, _ = assets[name]
    beta_vec = np.array([betas.get(f, 0.0) for f in factors])
    common = F @ beta_vec
    idio = rng.standard_normal(N_DAYS) * (idio_vol / np.sqrt(ANNUAL))
    log_rets[:, j] = drift / ANNUAL + common + idio

# Build price paths.
log_levels = np.vstack([np.zeros(len(names)), np.cumsum(log_rets, axis=0)])
start = np.array([assets[n][3] for n in names])
prices = start * np.exp(log_levels)

# Trading-day dates (skip weekends).
dates = []
d = date(2019, 1, 2)
while len(dates) < prices.shape[0]:
    if d.weekday() < 5:
        dates.append(d.isoformat())
    d += timedelta(days=1)

out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "returns")
os.makedirs(out_dir, exist_ok=True)
for j, name in enumerate(names):
    with open(os.path.join(out_dir, f"{name}.csv"), "w") as f:
        f.write("date,price\n")
        for i, dt in enumerate(dates):
            f.write(f"{dt},{prices[i, j]:.4f}\n")

# Diagnostics.
realized_vol = log_rets.std(axis=0, ddof=1) * np.sqrt(ANNUAL)
print(f"wrote {len(names)} assets x {N_DAYS} days to {out_dir}")
print("annualized vols:")
for n, v in zip(names, realized_vol):
    print(f"  {n:5s} {v*100:5.1f}%")
corr = np.corrcoef(log_rets, rowvar=False)
print("SPY/TLT corr =", round(corr[names.index("SPY"), names.index("TLT")], 3),
      "| HYG/SPY =", round(corr[names.index("HYG"), names.index("SPY")], 3),
      "| GLD/SPY =", round(corr[names.index("GLD"), names.index("SPY")], 3))
