"""Return-series alignment helpers (pure stdlib)."""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

from .types import PriceSeries


def align_returns(series: Dict[str, PriceSeries], symbols: List[str]
                  ) -> Tuple[List[date], Dict[str, List[float]]]:
    """Align return series of the given symbols onto their common **dates**.

    Intersects each symbol's calendar by actual trading date — not by tail
    position — so cross-market books (US/HK/CN have different holidays) line up
    day-for-day. When calendars share a single grid (the sample provider) the
    intersection is the full set, so behaviour is unchanged.

    Returns prices on dates only one symbol is missing are dropped; the surviving
    return then spans that gap (a multi-day return), which is the correct,
    leak-free choice when one market was closed.
    """
    present = [s for s in symbols if s in series and len(series[s].closes) > 1]
    if not present:
        return [], {}
    # Intersect trading dates across all present symbols.
    common = set(series[present[0]].dates)
    for s in present[1:]:
        common &= set(series[s].dates)
    if len(common) < 2:
        return [], {}
    dates = sorted(common)
    rets: Dict[str, List[float]] = {}
    for s in present:
        by_date = dict(zip(series[s].dates, series[s].closes))
        closes = [by_date[d] for d in dates]
        rets[s] = [(closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes))]
    return dates[1:], rets



def portfolio_returns(weights: Dict[str, float], rets: Dict[str, List[float]]) -> List[float]:
    """Weighted portfolio return series from per-asset returns (weights need not
    sum to 1; they are applied as-is, supporting long/short books)."""
    syms = [s for s in weights if s in rets and weights[s] != 0]
    if not syms:
        return []
    n = min(len(rets[s]) for s in syms)
    return [sum(weights[s] * rets[s][t] for s in syms) for t in range(n)]

