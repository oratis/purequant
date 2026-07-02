"""Futures analytics (pure Python).

Cost-of-carry fair value, basis, annualised roll yield, contract roll for
continuous backtest series, and index-futures sizing for beta hedging. Covers
the futures sleeve (index/commodity/rates) of the mandate — see docs/03 §3.4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


def fair_value(spot: float, r: float, t: float, q: float = 0.0,
               storage: float = 0.0, convenience: float = 0.0) -> float:
    """Cost-of-carry fair value of a futures contract.

    F = S * exp((r - q + storage - convenience) * t).
    For equity index futures use q = dividend yield, storage = convenience = 0.
    For commodities use storage cost and convenience yield.
    """
    return spot * math.exp((r - q + storage - convenience) * t)


@dataclass
class BasisAnalysis:
    spot: float
    futures: float
    basis: float                  # futures - spot
    basis_pct: float
    annualized_basis: float       # annualised carry implied by the basis
    structure: str                # "contango" / "backwardation" / "flat"


def basis_analysis(spot: float, futures: float, t: float) -> BasisAnalysis:
    basis = futures - spot
    basis_pct = basis / spot if spot else 0.0
    ann = (math.log(futures / spot) / t) if (spot > 0 and futures > 0 and t > 0) else 0.0
    if basis > 1e-9:
        structure = "contango"
    elif basis < -1e-9:
        structure = "backwardation"
    else:
        structure = "flat"
    return BasisAnalysis(spot=spot, futures=futures, basis=basis, basis_pct=basis_pct,
                         annualized_basis=ann, structure=structure)


def roll_yield(near_price: float, far_price: float, days_between: float) -> float:
    """Annualised roll yield from rolling a near contract into a far one.

    Positive in backwardation (near > far) for a long position. Approximated as
    (near/far - 1) annualised over the gap between expiries.
    """
    if far_price <= 0 or days_between <= 0:
        return 0.0
    return (near_price / far_price - 1.0) * (365.0 / days_between)


def build_continuous(contract_series: Sequence[Sequence[float]],
                     roll_indices: Sequence[int],
                     method: str = "ratio") -> List[float]:
    """Splice consecutive contract price series into a continuous series for
    backtesting, adjusting at each roll so the join is gap-free.

    ``contract_series`` is an ordered list of equal-length price lists (front,
    next, ...). ``roll_indices[k]`` is the index at which we roll OUT of
    contract k into contract k+1. ``method`` is 'ratio' (multiplicative) or
    'diff' (additive back-adjustment).
    """
    if not contract_series:
        return []
    n = len(contract_series[0])
    out: List[float] = [0.0] * n
    bounds = [-1] + list(roll_indices) + [n - 1]
    cum = 1.0 if method == "ratio" else 0.0
    for k in range(len(contract_series)):
        start, end = bounds[k] + 1, bounds[k + 1]
        series = contract_series[k]
        for i in range(start, min(end + 1, n)):
            out[i] = series[i] * cum if method == "ratio" else series[i] + cum
        # Update the cumulative adjustment so the next contract joins gap-free.
        if k + 1 < len(contract_series):
            roll_i = bounds[k + 1]
            cur, nxt = contract_series[k][roll_i], contract_series[k + 1][roll_i]
            if method == "ratio":
                cum *= (cur / nxt) if nxt else 1.0
            else:
                cum += (cur - nxt)
    return out


@dataclass
class IndexHedge:
    benchmark: str
    beta_dollar_gap: float        # beta-$ to neutralise
    contract_multiplier: float
    futures_price: float
    contracts: float              # signed; negative = short
    note: str


def index_futures_hedge(beta_dollar_gap: float, futures_price: float,
                        contract_multiplier: float, benchmark: str = "index") -> IndexHedge:
    """Number of index-futures contracts to neutralise a dollar-beta gap.

    contracts = -beta_$ / (futures_price * multiplier). Negative => short.
    """
    notional_per_contract = futures_price * contract_multiplier
    contracts = -beta_dollar_gap / notional_per_contract if notional_per_contract else 0.0
    direction = "short" if contracts < 0 else "long"
    return IndexHedge(
        benchmark=benchmark, beta_dollar_gap=beta_dollar_gap,
        contract_multiplier=contract_multiplier, futures_price=futures_price,
        contracts=contracts,
        note=f"{direction} {abs(contracts):.1f} {benchmark} futures "
             f"(notional/contract {notional_per_contract:,.0f}) to neutralise "
             f"beta-$ {beta_dollar_gap:,.0f}",
    )


def margin_utilisation(contracts: float, futures_price: float, multiplier: float,
                       margin_rate: float, available_margin: float) -> dict:
    """Initial margin required vs available; flags over-utilisation."""
    notional = abs(contracts) * futures_price * multiplier
    required = notional * margin_rate
    util = required / available_margin if available_margin else float("inf")
    return {"notional": notional, "initial_margin": required,
            "utilisation": util, "ok": util <= 1.0}
