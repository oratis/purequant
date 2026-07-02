"""Risk engine: volatility, beta, VaR/CVaR, drawdown, exposures, concentration,
correlation clustering and risk-contribution decomposition.

Operates on aligned return series (pure Python). Return-based metrics use the
linear book (equities/ETFs); option/future non-linear risk is handled by the
derivatives & monitor engines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from . import linalg, stats
from .types import AssetClass, Portfolio

TRADING_DAYS = stats.TRADING_DAYS


def ann_volatility(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return stats.annualize_vol(stats.std(returns))


def asset_beta(asset_returns: List[float], bench_returns: List[float]) -> float:
    n = min(len(asset_returns), len(bench_returns))
    a, b = asset_returns[-n:], bench_returns[-n:]
    var_b = stats.variance(b)
    if var_b == 0:
        return 0.0
    return stats.covariance(a, b) / var_b


def portfolio_beta(weights: Dict[str, float], rets: Dict[str, List[float]],
                   bench_returns: List[float]) -> float:
    """Weighted sum of per-asset betas (signed weights -> shorts reduce beta)."""
    total = 0.0
    for sym, w in weights.items():
        if sym in rets and w != 0:
            total += w * asset_beta(rets[sym], bench_returns)
    return total


def historical_var(returns: List[float], confidence: float = 0.95) -> float:
    """1-period historical VaR as a positive loss fraction."""
    if not returns:
        return 0.0
    q = stats.quantile(returns, 1 - confidence)
    return max(-q, 0.0)


def parametric_var(returns: List[float], confidence: float = 0.95) -> float:
    if len(returns) < 2:
        return 0.0
    mu, sd = stats.mean(returns), stats.std(returns)
    z = stats.normal_ppf(1 - confidence)
    return max(-(mu + z * sd), 0.0)


def cvar(returns: List[float], confidence: float = 0.95) -> float:
    """Expected shortfall: mean loss beyond the VaR threshold."""
    if not returns:
        return 0.0
    cutoff = stats.quantile(returns, 1 - confidence)
    tail = [r for r in returns if r <= cutoff]
    if not tail:
        return historical_var(returns, confidence)
    return max(-stats.mean(tail), 0.0)


def max_drawdown(returns: List[float]) -> float:
    """Max peak-to-trough drawdown of the cumulative return path (positive)."""
    if not returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for r in returns:
        equity *= (1 + r)
        peak = max(peak, equity)
        mdd = max(mdd, (peak - equity) / peak)
    return mdd


def current_drawdown(returns: List[float]) -> float:
    if not returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    for r in returns:
        equity *= (1 + r)
        peak = max(peak, equity)
    return (peak - equity) / peak if peak > 0 else 0.0


# ---- Exposure breakdowns --------------------------------------------------

@dataclass
class ExposureBreakdown:
    by_market: Dict[str, float] = field(default_factory=dict)
    by_sector: Dict[str, float] = field(default_factory=dict)
    by_asset_class: Dict[str, float] = field(default_factory=dict)
    by_currency: Dict[str, float] = field(default_factory=dict)
    long_value: float = 0.0
    short_value: float = 0.0
    gross: float = 0.0
    net: float = 0.0


def exposures(portfolio: Portfolio) -> ExposureBreakdown:
    """Signed exposures as fractions of total portfolio value."""
    tv = portfolio.total_value() or 1.0
    eb = ExposureBreakdown()
    for pos in portfolio.positions:
        v = portfolio.position_value_base(pos)
        inst = pos.instrument
        eb.by_market[inst.market.value] = eb.by_market.get(inst.market.value, 0.0) + v / tv
        sector = inst.sector or "Unknown"
        eb.by_sector[sector] = eb.by_sector.get(sector, 0.0) + v / tv
        eb.by_asset_class[inst.asset_class.value] = \
            eb.by_asset_class.get(inst.asset_class.value, 0.0) + v / tv
        eb.by_currency[inst.currency] = eb.by_currency.get(inst.currency, 0.0) + v / tv
        if v >= 0:
            eb.long_value += v
        else:
            eb.short_value += v
    eb.gross = (eb.long_value - eb.short_value) / tv
    eb.net = (eb.long_value + eb.short_value) / tv
    return eb


# ---- Concentration --------------------------------------------------------

@dataclass
class Concentration:
    hhi: float                       # Herfindahl index of |weights|
    effective_n: float               # 1/HHI: effective number of positions
    top_weights: List[Tuple[str, float]]


def concentration(portfolio: Portfolio, top_n: int = 5) -> Concentration:
    weights = portfolio.weights()
    abs_w = {s: abs(w) for s, w in weights.items()}
    total = sum(abs_w.values()) or 1.0
    norm = {s: w / total for s, w in abs_w.items()}
    hhi = sum(w * w for w in norm.values())
    eff_n = 1.0 / hhi if hhi > 0 else 0.0
    top = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return Concentration(hhi=hhi, effective_n=eff_n, top_weights=top)


# ---- Correlation clustering ----------------------------------------------

def high_correlation_pairs(rets: Dict[str, List[float]], threshold: float = 0.8
                           ) -> List[Tuple[str, str, float]]:
    syms = sorted(rets.keys())
    out: List[Tuple[str, str, float]] = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            c = stats.correlation(rets[syms[i]], rets[syms[j]])
            if abs(c) >= threshold:
                out.append((syms[i], syms[j], c))
    return sorted(out, key=lambda t: abs(t[2]), reverse=True)


# ---- Risk contribution ----------------------------------------------------

@dataclass
class RiskContribution:
    total_vol: float                     # annualised portfolio vol
    contributions: Dict[str, float]      # symbol -> share of total risk (sums ~1)


def risk_contributions(weights: Dict[str, float], rets: Dict[str, List[float]]
                       ) -> RiskContribution:
    """Euler/component risk contributions: RC_i = w_i * (Cov w)_i / sigma_p.

    Tells you which positions consume the risk budget — compare against their
    return contribution (docs/03 §0.4).
    """
    syms = [s for s in weights if s in rets and weights[s] != 0]
    if not syms:
        return RiskContribution(0.0, {})
    n = min(len(rets[s]) for s in syms)
    series = [rets[s][-n:] for s in syms]
    cov = stats.cov_matrix(series)
    w = [weights[s] for s in syms]
    cov_w = linalg.matvec(cov, w)
    var_p = linalg.dot(w, cov_w)
    if var_p <= 0:
        return RiskContribution(0.0, {s: 0.0 for s in syms})
    sigma_p = var_p ** 0.5
    rc = {syms[i]: (w[i] * cov_w[i]) / var_p for i in range(len(syms))}
    return RiskContribution(total_vol=stats.annualize_vol(sigma_p), contributions=rc)


# ---- Bundle for the linear book ------------------------------------------

@dataclass
class RiskReport:
    ann_vol: float
    beta: float
    var_95_1d: float
    cvar_95_1d: float
    max_drawdown: float
    current_drawdown: float
    exposures: ExposureBreakdown
    concentration: Concentration
    high_corr_pairs: List[Tuple[str, str, float]]
    risk_contributions: RiskContribution


def analyze_linear_book(portfolio: Portfolio, rets: Dict[str, List[float]],
                        bench_returns: List[float], confidence: float = 0.95
                        ) -> RiskReport:
    from .data import portfolio_returns
    # Linear book weights (equities + ETFs) for return-based metrics.
    weights = portfolio.weights()
    lin_syms = {p.instrument.symbol for p in portfolio.positions
                if p.instrument.asset_class in (AssetClass.EQUITY, AssetClass.ETF)}
    lin_weights = {s: w for s, w in weights.items() if s in lin_syms}
    port_ret = portfolio_returns(lin_weights, rets)
    return RiskReport(
        ann_vol=ann_volatility(port_ret),
        beta=portfolio_beta(lin_weights, rets, bench_returns),
        var_95_1d=historical_var(port_ret, confidence),
        cvar_95_1d=cvar(port_ret, confidence),
        max_drawdown=max_drawdown(port_ret),
        current_drawdown=current_drawdown(port_ret),
        exposures=exposures(portfolio),
        concentration=concentration(portfolio),
        high_corr_pairs=high_correlation_pairs({s: rets[s] for s in lin_weights if s in rets}),
        risk_contributions=risk_contributions(lin_weights, rets),
    )
