"""Factor engine: multi-factor stock scoring, cross-sectional standardisation,
sector neutralisation, composite scoring, selection, and single-factor IC tests.

All factors are constructed so that a HIGHER value implies a HIGHER expected
return (sign flips applied at construction). Standardisation and compositing
then use positive weights throughout. See docs/03 §1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import stats
from .types import PriceSeries

# Default factor weights for the composite score (sum need not be 1).
DEFAULT_WEIGHTS = {
    "value_ey": 1.0,     # earnings yield = 1/PE
    "value_bp": 0.7,     # book-to-price = 1/PB
    "quality_roe": 1.0,  # return on equity
    "quality_lowlev": 0.5,  # -debt/equity
    "growth": 0.8,       # earnings growth
    "momentum": 1.0,     # 12-1 price momentum
    "lowvol": 0.6,       # -realised volatility
}


@dataclass
class ScoredStock:
    symbol: str
    score: float
    rank: int
    factor_scores: Dict[str, float] = field(default_factory=dict)  # standardised z per factor


def compute_raw_factors(symbols: Sequence[str], fundamentals: Dict[str, Dict[str, float]],
                        prices: Dict[str, PriceSeries],
                        mom_lookback: int = 252, mom_skip: int = 21,
                        vol_window: int = 63) -> Dict[str, Dict[str, float]]:
    """Build the raw factor panel (higher = better) per symbol."""
    out: Dict[str, Dict[str, float]] = {}
    for s in symbols:
        f = fundamentals.get(s, {})
        row: Dict[str, float] = {}
        pe = f.get("pe")
        pb = f.get("pb")
        row["value_ey"] = 1.0 / pe if pe and pe > 0 else 0.0
        row["value_bp"] = 1.0 / pb if pb and pb > 0 else 0.0
        row["quality_roe"] = f.get("roe", 0.0)
        row["quality_lowlev"] = -f.get("debt_to_equity", 0.0)
        row["growth"] = f.get("earnings_growth", 0.0)
        # Price-based factors.
        ps = prices.get(s)
        if ps and len(ps.closes) > mom_lookback:
            c = ps.closes
            row["momentum"] = c[-mom_skip] / c[-mom_lookback] - 1.0
        else:
            row["momentum"] = 0.0
        if ps and len(ps.closes) > vol_window:
            rets = ps.returns()[-vol_window:]
            row["lowvol"] = -stats.annualize_vol(stats.std(rets)) if len(rets) > 1 else 0.0
        else:
            row["lowvol"] = 0.0
        out[s] = row
    return out


def _standardize_cross_section(panel: Dict[str, Dict[str, float]], factor: str,
                               winsor: bool = True) -> Dict[str, float]:
    syms = list(panel.keys())
    vals = [panel[s].get(factor, 0.0) for s in syms]
    # A z-score needs cross-sectional spread; with 0 or 1 names it is undefined
    # (and stats.std raises). Return neutral zeros so a thin/empty universe — now
    # reachable via the portfolio-holdings universe fallback — degrades instead of
    # crashing.
    if len(vals) < 2:
        return {s: 0.0 for s in syms}
    if winsor and len(vals) >= 5:
        vals = stats.winsorize(vals)
    z = stats.zscore(vals)
    return dict(zip(syms, z))


def _neutralize(zmap: Dict[str, float], groups: Dict[str, str]) -> Dict[str, float]:
    """Subtract the group (e.g. sector) mean from each z-score, then re-standardise."""
    by_group: Dict[str, List[str]] = {}
    for s, g in groups.items():
        by_group.setdefault(g, []).append(s)
    out: Dict[str, float] = {}
    for g, members in by_group.items():
        vals = [zmap[s] for s in members if s in zmap]
        gmean = stats.mean(vals) if vals else 0.0
        for s in members:
            if s in zmap:
                out[s] = zmap[s] - gmean
    # Re-standardise across the whole cross-section.
    syms = list(out.keys())
    z = stats.zscore([out[s] for s in syms]) if len(syms) > 1 else [0.0] * len(syms)
    return dict(zip(syms, z))


@dataclass
class FactorModel:
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    neutralize_by_sector: bool = True

    def score(self, symbols: Sequence[str], fundamentals: Dict[str, Dict[str, float]],
              prices: Dict[str, PriceSeries], sectors: Optional[Dict[str, str]] = None
              ) -> List[ScoredStock]:
        panel = compute_raw_factors(symbols, fundamentals, prices)
        syms = list(panel.keys())
        # Standardise each factor cross-sectionally.
        z_by_factor: Dict[str, Dict[str, float]] = {}
        for factor in self.weights:
            zmap = _standardize_cross_section(panel, factor)
            if self.neutralize_by_sector and sectors:
                zmap = _neutralize(zmap, {s: sectors.get(s, "Unknown") for s in syms})
            z_by_factor[factor] = zmap
        # Composite score.
        results: List[ScoredStock] = []
        for s in syms:
            fs = {f: z_by_factor[f].get(s, 0.0) for f in self.weights}
            score = sum(self.weights[f] * fs[f] for f in self.weights)
            results.append(ScoredStock(symbol=s, score=score, rank=0, factor_scores=fs))
        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results


def select(scored: List[ScoredStock], top_n: Optional[int] = None,
           top_quantile: Optional[float] = None) -> List[ScoredStock]:
    if top_n is not None:
        return scored[:top_n]
    if top_quantile is not None:
        k = max(1, int(len(scored) * top_quantile))
        return scored[:k]
    return scored


# ---- Single-factor evaluation --------------------------------------------

@dataclass
class ICResult:
    ic: float        # Pearson IC: corr(factor, forward return)
    rank_ic: float   # Spearman-style IC: corr(rank(factor), rank(forward))
    n: int


def information_coefficient(factor_values: Dict[str, float],
                            forward_returns: Dict[str, float]) -> ICResult:
    """Cross-sectional IC between a factor snapshot and forward returns."""
    syms = [s for s in factor_values if s in forward_returns]
    if len(syms) < 3:
        return ICResult(0.0, 0.0, len(syms))
    fv = [factor_values[s] for s in syms]
    fr = [forward_returns[s] for s in syms]
    ic = stats.correlation(fv, fr)
    rank_ic = stats.correlation(stats.rank(fv), stats.rank(fr))
    return ICResult(ic=ic, rank_ic=rank_ic, n=len(syms))
