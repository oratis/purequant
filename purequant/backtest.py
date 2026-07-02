"""Lightweight backtest framework (pure Python).

Includes performance statistics, a rebalanced fixed-weight backtest, and a
self-contained cross-sectional momentum strategy that uses only past data at
each rebalance (no look-ahead). For production-grade backtests (richer costs,
borrow, point-in-time fundamentals) see docs/03 §0.1 and docs/06 §2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from . import stats
from .types import PriceSeries

TRADING_DAYS = stats.TRADING_DAYS


@dataclass
class PerfStats:
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    n_days: int


def performance_stats(returns: List[float], rf_daily: float = 0.0) -> PerfStats:
    if not returns:
        return PerfStats(0, 0, 0, 0, 0, 0, 0)
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for r in returns:
        equity *= (1 + r)
        peak = max(peak, equity)
        mdd = max(mdd, (peak - equity) / peak)
    total = equity - 1.0
    n = len(returns)
    cagr = (equity ** (TRADING_DAYS / n) - 1.0) if n > 0 else 0.0
    vol = stats.annualize_vol(stats.std(returns)) if n > 1 else 0.0
    excess = stats.mean([r - rf_daily for r in returns])
    sharpe = (excess / stats.std(returns) * math.sqrt(TRADING_DAYS)) if n > 1 and stats.std(returns) > 0 else 0.0
    calmar = (cagr / mdd) if mdd > 0 else 0.0
    return PerfStats(total, cagr, vol, sharpe, mdd, calmar, n)


@dataclass
class BacktestResult:
    returns: List[float]
    equity_curve: List[float]
    stats: PerfStats
    turnover: float = 0.0
    n_rebalances: int = 0


def backtest_rebalanced(weights: Dict[str, float], rets: Dict[str, List[float]],
                        rebalance_every: int = 21, cost_bps: float = 5.0) -> BacktestResult:
    """Backtest a fixed target-weight portfolio, rebalanced every N days, with
    proportional transaction costs. Between rebalances weights drift with prices.
    """
    syms = [s for s in weights if s in rets and weights[s] != 0]
    if not syms:
        return BacktestResult([], [1.0], performance_stats([]))
    n = min(len(rets[s]) for s in syms)
    target = {s: weights[s] for s in syms}
    w = dict(target)
    port_rets: List[float] = []
    equity = [1.0]
    total_turnover = 0.0
    n_reb = 0
    cost = cost_bps / 10000.0
    for t in range(n):
        # Apply one day of returns; weights drift.
        day_ret = sum(w[s] * rets[s][t] for s in syms)
        new_w = {}
        for s in syms:
            new_w[s] = w[s] * (1 + rets[s][t])
        gross = sum(new_w.values())
        if gross != 0:
            new_w = {s: v / gross * sum(w.values()) for s, v in new_w.items()}
        w = new_w
        # Rebalance.
        if (t + 1) % rebalance_every == 0:
            turn = sum(abs(target[s] - w[s]) for s in syms)
            total_turnover += turn
            day_ret -= turn * cost
            w = dict(target)
            n_reb += 1
        port_rets.append(day_ret)
        equity.append(equity[-1] * (1 + day_ret))
    return BacktestResult(port_rets, equity, performance_stats(port_rets),
                          turnover=total_turnover, n_rebalances=n_reb)


def backtest_momentum(prices: Dict[str, PriceSeries], symbols: Sequence[str],
                      lookback: int = 126, skip: int = 21, top_n: int = 3,
                      rebalance_every: int = 21, cost_bps: float = 5.0,
                      warmup: Optional[int] = None) -> BacktestResult:
    """Cross-sectional momentum: every ``rebalance_every`` days, rank by past
    ``lookback``-day return (skipping the most recent ``skip`` days), hold the
    top ``top_n`` equally weighted. Uses only past data at each decision point.
    """
    syms = [s for s in symbols if s in prices and len(prices[s].closes) > lookback + 2]
    if not syms:
        return BacktestResult([], [1.0], performance_stats([]))
    n = min(len(prices[s].closes) for s in syms)
    closes = {s: prices[s].closes[-n:] for s in syms}
    start = warmup if warmup is not None else lookback + skip + 1
    port_rets: List[float] = []
    equity = [1.0]
    held: List[str] = []
    total_turnover = 0.0
    n_reb = 0
    cost = cost_bps / 10000.0
    for t in range(start, n):
        # Daily return of currently held basket (decided at previous rebalance).
        if held:
            day_ret = sum((closes[s][t] / closes[s][t - 1] - 1.0) for s in held) / len(held)
        else:
            day_ret = 0.0
        # Rebalance using only information up to t-1.
        if (t - start) % rebalance_every == 0:
            scores = []
            for s in syms:
                past = closes[s]
                mom = past[t - 1 - skip] / past[t - 1 - lookback] - 1.0
                scores.append((s, mom))
            scores.sort(key=lambda kv: kv[1], reverse=True)
            new_held = [s for s, _ in scores[:top_n]]
            turn = len(set(new_held) ^ set(held)) / max(top_n, 1)
            total_turnover += turn
            day_ret -= turn * cost
            held = new_held
            n_reb += 1
        port_rets.append(day_ret)
        equity.append(equity[-1] * (1 + day_ret))
    return BacktestResult(port_rets, equity, performance_stats(port_rets),
                          turnover=total_turnover, n_rebalances=n_reb)
