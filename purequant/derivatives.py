"""Derivatives engine: Black-Scholes-Merton pricing, Greeks, implied vol, and
multi-leg option strategy evaluation (payoff, breakevens, aggregate Greeks).

Pure Python (uses invest.core.stats for the normal distribution). For American
options / dividends-as-discrete / exotic payoffs, swap in QuantLib (docs/03 §3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from . import stats
from .types import OptionRight


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float    # per 1.00 (100%) change in vol; divide by 100 for per-1%-vol
    theta: float   # per year; divide by 365 for per-calendar-day
    rho: float


def _d1_d2(s: float, k: float, t: float, r: float, sigma: float, q: float):
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        return None, None
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return d1, d2


def bsm_price(s: float, k: float, t: float, r: float, sigma: float,
              right: OptionRight, q: float = 0.0) -> float:
    """Black-Scholes-Merton price. s=spot, k=strike, t=years, r=rate, q=dividend
    yield, sigma=vol."""
    if t <= 0:  # intrinsic value at/after expiry
        return max(s - k, 0.0) if right == OptionRight.CALL else max(k - s, 0.0)
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    nd1, nd2 = stats.normal_cdf(d1), stats.normal_cdf(d2)
    if right == OptionRight.CALL:
        return s * math.exp(-q * t) * nd1 - k * math.exp(-r * t) * nd2
    return k * math.exp(-r * t) * stats.normal_cdf(-d2) - s * math.exp(-q * t) * stats.normal_cdf(-d1)


def bsm_greeks(s: float, k: float, t: float, r: float, sigma: float,
               right: OptionRight, q: float = 0.0) -> Greeks:
    price = bsm_price(s, k, t, r, sigma, right, q)
    if t <= 0 or sigma <= 0:
        # Degenerate: delta is 0/1 at expiry, other Greeks ~0.
        if right == OptionRight.CALL:
            delta = 1.0 if s > k else 0.0
        else:
            delta = -1.0 if s < k else 0.0
        return Greeks(price, delta, 0.0, 0.0, 0.0, 0.0)
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    pdf = stats.normal_pdf(d1)
    sqrt_t = math.sqrt(t)
    disc_q = math.exp(-q * t)
    gamma = disc_q * pdf / (s * sigma * sqrt_t)
    vega = s * disc_q * pdf * sqrt_t
    if right == OptionRight.CALL:
        delta = disc_q * stats.normal_cdf(d1)
        theta = (-s * disc_q * pdf * sigma / (2 * sqrt_t)
                 - r * k * math.exp(-r * t) * stats.normal_cdf(d2)
                 + q * s * disc_q * stats.normal_cdf(d1))
        rho = k * t * math.exp(-r * t) * stats.normal_cdf(d2)
    else:
        delta = -disc_q * stats.normal_cdf(-d1)
        theta = (-s * disc_q * pdf * sigma / (2 * sqrt_t)
                 + r * k * math.exp(-r * t) * stats.normal_cdf(-d2)
                 - q * s * disc_q * stats.normal_cdf(-d1))
        rho = -k * t * math.exp(-r * t) * stats.normal_cdf(-d2)
    return Greeks(price, delta, gamma, vega, theta, rho)


def implied_vol(price: float, s: float, k: float, t: float, r: float,
                right: OptionRight, q: float = 0.0,
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-6) -> Optional[float]:
    """Solve for implied volatility by bisection. Returns None if no solution in
    the bracket (e.g. price below intrinsic)."""
    if t <= 0:
        return None
    intrinsic = (max(s - k, 0.0) if right == OptionRight.CALL else max(k - s, 0.0))
    if price < intrinsic - 1e-8:
        return None
    f_lo = bsm_price(s, k, t, r, lo, right, q) - price
    f_hi = bsm_price(s, k, t, r, hi, right, q) - price
    if f_lo * f_hi > 0:
        return None
    a, b = lo, hi
    for _ in range(200):
        mid = 0.5 * (a + b)
        fm = bsm_price(s, k, t, r, mid, right, q) - price
        if abs(fm) < tol:
            return mid
        if f_lo * fm < 0:
            b = mid
            f_hi = fm
        else:
            a = mid
            f_lo = fm
    return 0.5 * (a + b)


def year_fraction(expiry: date, as_of: date) -> float:
    return max((expiry - as_of).days / 365.0, 0.0)


# ---- Multi-leg strategy evaluation ----------------------------------------

@dataclass
class OptionLeg:
    right: OptionRight
    strike: float
    expiry: date
    quantity: float          # +long / -short, in contracts
    premium: float = 0.0     # entry premium per share (paid if long, received if short)
    multiplier: float = 100.0
    sigma: float = 0.25      # vol used for Greeks


@dataclass
class StrategyEval:
    legs: List[OptionLeg]
    spot: float
    net_premium: float                  # total cash flow at entry (negative = net debit)
    max_profit: Optional[float]
    max_loss: Optional[float]
    breakevens: List[float]
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega: float = 0.0
    net_theta: float = 0.0
    payoff_curve: List[tuple] = field(default_factory=list)  # (spot, pnl) samples


def _leg_payoff_at_expiry(leg: OptionLeg, spot: float) -> float:
    if leg.right == OptionRight.CALL:
        intrinsic = max(spot - leg.strike, 0.0)
    else:
        intrinsic = max(leg.strike - spot, 0.0)
    # PnL = (value at expiry - premium paid) * qty * multiplier
    return (intrinsic - leg.premium) * leg.quantity * leg.multiplier


def evaluate_strategy(legs: List[OptionLeg], spot: float, as_of: date,
                      r: float = 0.04, q: float = 0.0,
                      price_range: float = 0.5, n_points: int = 81) -> StrategyEval:
    """Evaluate a multi-leg option position: payoff diagram, breakevens,
    max profit/loss, and net Greeks at the current spot."""
    net_premium = -sum(leg.premium * leg.quantity * leg.multiplier for leg in legs)
    lo = spot * (1 - price_range)
    hi = spot * (1 + price_range)
    curve = []
    for i in range(n_points):
        sp = lo + (hi - lo) * i / (n_points - 1)
        pnl = sum(_leg_payoff_at_expiry(leg, sp) for leg in legs)
        curve.append((sp, pnl))
    pnls = [p for _, p in curve]
    max_profit = max(pnls)
    max_loss = min(pnls)
    # Breakevens: sign changes along the payoff curve (linear interpolation).
    breakevens = []
    for i in range(1, len(curve)):
        p0, p1 = curve[i - 1][1], curve[i][1]
        if (p0 <= 0 <= p1) or (p1 <= 0 <= p0):
            if p1 != p0:
                s0, s1 = curve[i - 1][0], curve[i][0]
                be = s0 + (s1 - s0) * (0 - p0) / (p1 - p0)
                breakevens.append(round(be, 2))
    # Net Greeks at current spot.
    nd = ng = nv = nt = 0.0
    for leg in legs:
        t = year_fraction(leg.expiry, as_of)
        g = bsm_greeks(spot, leg.strike, t, r, leg.sigma, leg.right, q)
        scale = leg.quantity * leg.multiplier
        nd += g.delta * scale
        ng += g.gamma * scale
        nv += g.vega * scale / 100.0    # per 1% vol
        nt += g.theta * scale / 365.0   # per calendar day
    return StrategyEval(
        legs=legs, spot=spot, net_premium=net_premium,
        max_profit=max_profit, max_loss=max_loss, breakevens=sorted(set(breakevens)),
        net_delta=nd, net_gamma=ng, net_vega=nv, net_theta=nt, payoff_curve=curve,
    )
