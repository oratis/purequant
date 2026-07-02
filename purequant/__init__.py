"""purequant — a zero-dependency quantitative-finance toolkit in pure Python.

Everything here is standard-library only: no numpy, no pandas, no C extensions.
Sized for personal-to-desk-scale portfolios; swap the internals for numpy if you
outgrow it (the interfaces take plain ``list``/``dict``).

Modules
-------
- ``linalg``       : vectors/matrices, Gaussian solve, inverse, PSD ridge
- ``stats``        : mean/var/cov/corr, z-score, winsorize, quantile, OLS, shrinkage
- ``types``        : domain dataclasses (Instrument / Position / Portfolio / PriceSeries)
- ``data``         : return-series alignment by trading date
- ``risk``         : volatility, VaR/CVaR, drawdown, beta, Euler risk contributions
- ``optimize``     : min-variance, mean-variance, risk parity (constrained)
- ``derivatives``  : Black-Scholes-Merton price + Greeks + implied vol
- ``bonds``        : price/YTM, Macaulay & modified duration, convexity, DV01
- ``futures``      : cost-of-carry fair value, basis, roll yield, index hedge
- ``hedge``        : beta hedge, min-variance hedge ratio, pairs/cointegration
- ``factor``       : multi-factor scoring, cross-sectional standardisation, IC
- ``attribution``  : factor (regression) & Brinson sector attribution
- ``backtest``     : rebalanced & cross-sectional momentum backtests

Quick start
-----------
>>> from purequant import derivatives as dv
>>> dv.bsm_price(spot=100, strike=100, t=1, r=0.03, sigma=0.2, right="call")
>>> from purequant import risk
>>> risk.historical_var([-0.01, 0.02, -0.03, 0.015], confidence=0.95)
"""
from __future__ import annotations

from . import (attribution, backtest, bonds, data, derivatives, factor,
               futures, hedge, linalg, optimize, risk, stats, types)

__all__ = ["linalg", "stats", "types", "data", "risk", "optimize",
           "derivatives", "bonds", "futures", "hedge", "factor",
           "attribution", "backtest"]
__version__ = "0.1.0"
