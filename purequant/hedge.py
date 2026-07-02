"""Hedge & risk-neutral engine: beta hedging, minimum-variance hedge ratios,
pairs/cointegration screening, and hedge-effectiveness measurement.

See docs/03 §2. Outputs are hedge *suggestions* (instrument, direction, size) —
never orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import stats
from .types import Portfolio
from . import risk


@dataclass
class BetaHedgePlan:
    benchmark: str
    current_beta: float          # portfolio beta (weighted, signed)
    target_beta: float
    portfolio_value: float
    beta_dollar_gap: float       # beta-$ that needs neutralising
    hedge_notional: float        # signed $ notional of benchmark to add (-=short)
    hedge_units: Optional[float] # signed units, if benchmark price provided
    note: str = ""


def beta_hedge(portfolio: Portfolio, rets: Dict[str, List[float]],
               benchmark: str, bench_returns: List[float],
               target_beta: float = 0.0, bench_price: Optional[float] = None
               ) -> BetaHedgePlan:
    """Compute the benchmark hedge needed to move portfolio beta to target.

    Uses the linear book's dollar-beta. A short benchmark position (negative
    notional) reduces a positive net beta.
    """
    weights = portfolio.weights()
    pv = portfolio.total_value()
    cur_beta = risk.portfolio_beta(weights, rets, bench_returns)
    beta_gap = cur_beta - target_beta
    # Dollar-beta to neutralise; hedge has beta ~1 to itself.
    hedge_notional = -beta_gap * pv
    hedge_units = (hedge_notional / bench_price) if bench_price else None
    direction = "short" if hedge_notional < 0 else "long"
    return BetaHedgePlan(
        benchmark=benchmark, current_beta=cur_beta, target_beta=target_beta,
        portfolio_value=pv, beta_dollar_gap=beta_gap * pv,
        hedge_notional=hedge_notional, hedge_units=hedge_units,
        note=f"{direction} {abs(hedge_notional):,.0f} {portfolio.base_currency} of {benchmark} "
             f"to move beta {cur_beta:.2f} -> {target_beta:.2f}",
    )


def min_variance_hedge_ratio(asset_returns: List[float], hedge_returns: List[float]) -> float:
    """Optimal hedge ratio h* = Cov(asset, hedge)/Var(hedge): units of hedge per
    unit of asset that minimise variance of (asset - h*hedge)."""
    n = min(len(asset_returns), len(hedge_returns))
    a, h = asset_returns[-n:], hedge_returns[-n:]
    var_h = stats.variance(h)
    if var_h == 0:
        return 0.0
    return stats.covariance(a, h) / var_h


def hedge_effectiveness(asset_returns: List[float], hedge_returns: List[float],
                        hedge_ratio: float) -> Dict[str, object]:
    """Variance reduction from applying the hedge ratio.

    NOTE: this is an **in-sample** measure — the hedge ratio and the variance
    reduction are computed on the same window, so it overstates what the hedge
    will deliver out-of-sample. For a realistic estimate, fit the ratio on one
    window and measure effectiveness on a later (walk-forward) window.
    """
    n = min(len(asset_returns), len(hedge_returns))
    a, h = asset_returns[-n:], hedge_returns[-n:]
    hedged = [a[i] - hedge_ratio * h[i] for i in range(n)]
    var_unhedged = stats.variance(a)
    var_hedged = stats.variance(hedged)
    reduction = 1 - var_hedged / var_unhedged if var_unhedged > 0 else 0.0
    return {
        "vol_unhedged": stats.annualize_vol(stats.std(a)),
        "vol_hedged": stats.annualize_vol(stats.std(hedged)),
        "variance_reduction": reduction,
        "caveat": "in-sample; overstates out-of-sample effectiveness",
    }


# ---- Pairs / statistical arbitrage ---------------------------------------

@dataclass
class PairCandidate:
    a: str
    b: str
    hedge_ratio: float           # units of b per unit of a (from OLS)
    correlation: float
    spread_zscore: float         # current standardised spread (entry signal)
    half_life: Optional[float]   # mean-reversion half-life in days (OU estimate)
    adf_like_stat: float         # residual stationarity heuristic (more negative = better)


def _half_life(spread: List[float]) -> Optional[float]:
    """Estimate OU mean-reversion half-life from the spread:
    d(spread)_t = a + b*spread_{t-1} + e ;  half-life = -ln(2)/b  (b<0)."""
    import math
    if len(spread) < 10:
        return None
    y = [spread[i] - spread[i - 1] for i in range(1, len(spread))]
    x = spread[:-1]
    res = stats.ols(y, [x])
    b = res.beta[0]
    if b >= 0:
        return None
    return -math.log(2) / b


def _stationarity_stat(resid: List[float]) -> float:
    """Cheap ADF-like statistic: t-stat of the lagged-level coefficient in
    d(resid) = rho*resid_{t-1} + e. More negative => more mean-reverting.
    (For rigorous testing use statsmodels.adfuller — docs/03.)"""
    if len(resid) < 10:
        return 0.0
    y = [resid[i] - resid[i - 1] for i in range(1, len(resid))]
    x = resid[:-1]
    res = stats.ols(y, [x], add_intercept=True)
    rho = res.beta[0]
    n = len(y)
    se = stats.std(res.resid) / (stats.std(x) * (n ** 0.5)) if stats.std(x) > 0 else 1e-9
    return rho / se if se else 0.0


def find_pairs(prices: Dict[str, "object"], symbols: List[str],
               min_corr: float = 0.7, entry_z: float = 2.0
               ) -> List[PairCandidate]:
    """Screen symbol pairs for mean-reverting spread candidates.

    ``prices`` maps symbol -> PriceSeries. Uses log-price OLS to get the hedge
    ratio, then evaluates spread stationarity, half-life and current z-score.
    """
    import math
    candidates: List[PairCandidate] = []
    seen = set()
    syms = [s for s in symbols
            if s in prices and len(prices[s].closes) > 30 and not (s in seen or seen.add(s))]
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            sa, sb = syms[i], syms[j]
            ca = [math.log(x) for x in prices[sa].closes]
            cb = [math.log(x) for x in prices[sb].closes]
            n = min(len(ca), len(cb))
            ca, cb = ca[-n:], cb[-n:]
            ra = [ca[k] - ca[k - 1] for k in range(1, n)]
            rb = [cb[k] - cb[k - 1] for k in range(1, n)]
            corr = stats.correlation(ra, rb)
            if abs(corr) < min_corr:
                continue
            reg = stats.ols(ca, [cb])           # log_a = alpha + beta*log_b
            beta = reg.beta[0]
            spread = [ca[k] - (reg.intercept + beta * cb[k]) for k in range(n)]
            sd = stats.std(spread)
            z = (spread[-1] - stats.mean(spread)) / sd if sd > 0 else 0.0
            candidates.append(PairCandidate(
                a=sa, b=sb, hedge_ratio=beta, correlation=corr,
                spread_zscore=z, half_life=_half_life(spread),
                adf_like_stat=_stationarity_stat(spread),
            ))
    # Prioritise tradeable spreads: currently stretched and mean-reverting.
    candidates.sort(key=lambda c: (abs(c.spread_zscore) >= entry_z, -c.adf_like_stat,
                                   abs(c.spread_zscore)), reverse=True)
    return candidates


# ---- Factor-neutral target (thin wrapper over optimize) -------------------

@dataclass
class NeutralTarget:
    weights: Dict[str, float]
    achieved_beta: float
    note: str = ""


def market_neutral_overlay(portfolio: Portfolio, rets: Dict[str, List[float]],
                           benchmark: str, bench_returns: List[float],
                           bench_price: Optional[float] = None) -> Tuple[BetaHedgePlan, dict]:
    """Convenience: produce a beta-neutralising hedge plan plus before/after vol,
    so the report can show what the hedge buys you."""
    from .data import portfolio_returns
    plan = beta_hedge(portfolio, rets, benchmark, bench_returns, 0.0, bench_price)
    weights = portfolio.weights()
    before = portfolio_returns(weights, rets)
    # Approximate hedged returns: add benchmark weight = hedge_notional/pv.
    hw = dict(weights)
    hw[benchmark] = hw.get(benchmark, 0.0) + plan.hedge_notional / (plan.portfolio_value or 1.0)
    after_rets = dict(rets)
    after = portfolio_returns(hw, after_rets)
    eff = {
        "vol_before": risk.ann_volatility(before),
        "vol_after": risk.ann_volatility(after),
        "beta_before": plan.current_beta,
        "beta_after": 0.0,
    }
    return plan, eff
