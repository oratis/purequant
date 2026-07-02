"""Return attribution (docs/03 §0.4).

Two approaches:

* Factor (regression) attribution — decompose a portfolio's realised return into
  exposures to market / value / momentum style factors plus alpha, via OLS.
* Brinson attribution — allocation vs selection effects against a benchmark,
  by sector.

Style-factor return series are built from the universe as long/short tercile
spreads. Rankings here use the full sample (in-sample) for illustration; a
production attribution would use point-in-time factor returns (docs/06 §2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from . import stats
from .types import PriceSeries


def _long_short_spread(rets: Dict[str, List[float]], ranking: List[str],
                       frac: float = 0.34) -> List[float]:
    """Daily return of a long-top / short-bottom equally weighted tercile spread.
    ``ranking`` is symbols sorted best-first."""
    avail = [s for s in ranking if s in rets]
    if len(avail) < 3:
        return []
    k = max(1, int(len(avail) * frac))
    longs, shorts = avail[:k], avail[-k:]
    n = min(min(len(rets[s]) for s in longs), min(len(rets[s]) for s in shorts))
    out = []
    for t in range(n):
        lr = sum(rets[s][t] for s in longs) / len(longs)
        sr = sum(rets[s][t] for s in shorts) / len(shorts)
        out.append(lr - sr)
    return out


def build_style_factor_returns(rets: Dict[str, List[float]],
                               fundamentals: Dict[str, Dict[str, float]],
                               prices: Dict[str, PriceSeries],
                               bench_returns: List[float],
                               mom_lookback: int = 126) -> Dict[str, List[float]]:
    """Construct market / value / momentum factor return series."""
    syms = [s for s in rets if s in fundamentals]
    factors: Dict[str, List[float]] = {}
    if bench_returns:
        factors["market"] = list(bench_returns)
    # Value: earnings yield = 1/PE, high is cheap (long).
    ey = {s: (1.0 / fundamentals[s]["pe"]) if fundamentals[s].get("pe") else 0.0 for s in syms}
    value_rank = sorted(syms, key=lambda s: ey[s], reverse=True)
    vs = _long_short_spread(rets, value_rank)
    if vs:
        factors["value"] = vs
    # Momentum: trailing return over lookback (using prices).
    mom = {}
    for s in syms:
        c = prices[s].closes if s in prices else []
        mom[s] = (c[-1] / c[-mom_lookback] - 1.0) if len(c) > mom_lookback else 0.0
    mom_rank = sorted(syms, key=lambda s: mom[s], reverse=True)
    ms = _long_short_spread(rets, mom_rank)
    if ms:
        factors["momentum"] = ms
    return factors


@dataclass
class FactorAttribution:
    exposures: Dict[str, float]          # regression beta to each factor
    contributions: Dict[str, float]      # annualised return contribution per factor
    alpha_annual: float                  # annualised intercept (unexplained)
    r2: float


def factor_attribution(port_returns: List[float],
                       factor_returns: Dict[str, List[float]]) -> FactorAttribution:
    names = [k for k in factor_returns if factor_returns[k]]
    n = min([len(port_returns)] + [len(factor_returns[k]) for k in names])
    if n < 5 or not names:
        return FactorAttribution({}, {}, 0.0, 0.0)
    y = port_returns[-n:]
    cols = [factor_returns[k][-n:] for k in names]
    res = stats.ols(y, cols, add_intercept=True)
    exposures = {names[i]: res.beta[i] for i in range(len(names))}
    contributions = {names[i]: res.beta[i] * stats.mean(cols[i]) * stats.TRADING_DAYS
                     for i in range(len(names))}
    alpha = res.intercept * stats.TRADING_DAYS
    return FactorAttribution(exposures=exposures, contributions=contributions,
                             alpha_annual=alpha, r2=res.r2)


# ---- Brinson sector attribution -------------------------------------------

@dataclass
class BrinsonResult:
    allocation: Dict[str, float]      # effect of over/under-weighting sectors
    selection: Dict[str, float]       # effect of picking better names within sectors
    interaction: Dict[str, float]
    total_active: float               # portfolio return - benchmark return
    allocation_total: float = 0.0
    selection_total: float = 0.0
    interaction_total: float = 0.0


def brinson(port_weights: Dict[str, float], port_sector_returns: Dict[str, float],
            bench_weights: Dict[str, float], bench_sector_returns: Dict[str, float]
            ) -> BrinsonResult:
    """Brinson-Hood-Beebower attribution by sector.

    allocation_i = (wp_i - wb_i) * (rb_i - rb_total)
    selection_i  = wb_i * (rp_i - rb_i)
    interaction_i= (wp_i - wb_i) * (rp_i - rb_i)
    """
    sectors = set(port_weights) | set(bench_weights)
    rb_total = sum(bench_weights.get(s, 0.0) * bench_sector_returns.get(s, 0.0) for s in sectors)
    rp_total = sum(port_weights.get(s, 0.0) * port_sector_returns.get(s, 0.0) for s in sectors)
    alloc, sel, inter = {}, {}, {}
    for s in sectors:
        wp, wb = port_weights.get(s, 0.0), bench_weights.get(s, 0.0)
        rp, rb = port_sector_returns.get(s, 0.0), bench_sector_returns.get(s, 0.0)
        alloc[s] = (wp - wb) * (rb - rb_total)
        sel[s] = wb * (rp - rb)
        inter[s] = (wp - wb) * (rp - rb)
    return BrinsonResult(
        allocation=alloc, selection=sel, interaction=inter,
        total_active=rp_total - rb_total,
        allocation_total=sum(alloc.values()), selection_total=sum(sel.values()),
        interaction_total=sum(inter.values()))
