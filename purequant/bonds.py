"""Bond analytics (pure Python).

Fixed-coupon bond pricing, yield-to-maturity, Macaulay / modified duration,
convexity, DV01, and portfolio-level interest-rate sensitivity. Covers the
bond sleeve of the US/HK/CN mandate (docs/03). Day-count is simplified to
periodic coupons; for production use an exact day-count calendar (ACT/ACT,
30/360) per market convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class Bond:
    face: float = 100.0
    coupon_rate: float = 0.05      # annual coupon as a fraction of face
    years_to_maturity: float = 5.0
    freq: int = 2                  # coupon payments per year
    name: str = ""

    def cashflows(self) -> List[tuple]:
        """List of (time_in_years, cashflow). Final period includes face."""
        n = max(int(round(self.years_to_maturity * self.freq)), 1)
        c = self.face * self.coupon_rate / self.freq
        out = []
        for i in range(1, n + 1):
            t = i / self.freq
            cf = c + (self.face if i == n else 0.0)
            out.append((t, cf))
        return out


def price_from_yield(bond: Bond, ytm: float) -> float:
    """Present value of cashflows discounted at the (annual, compounded ``freq``
    times) yield to maturity."""
    m = bond.freq
    pv = 0.0
    for t, cf in bond.cashflows():
        pv += cf / (1 + ytm / m) ** (t * m)
    return pv


def yield_to_maturity(bond: Bond, price: float,
                      lo: float = -0.9, hi: float = 5.0, tol: float = 1e-8) -> Optional[float]:
    """Solve for YTM by bisection (price is monotone decreasing in yield).

    The default bracket [-90%, +500%] covers negative-yield and distressed/
    deep-discount bonds. Prices outside the bracket return None (caller should
    treat as "not solvable" rather than fabricate a yield)."""
    f_lo = price_from_yield(bond, lo) - price
    f_hi = price_from_yield(bond, hi) - price
    if f_lo * f_hi > 0:
        return None
    a, b = lo, hi
    for _ in range(200):
        mid = 0.5 * (a + b)
        fm = price_from_yield(bond, mid) - price
        if abs(fm) < tol:
            return mid
        if f_lo * fm < 0:
            b = mid
        else:
            a, f_lo = mid, fm
    return 0.5 * (a + b)


@dataclass
class BondRisk:
    price: float
    ytm: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float            # dollar value of 1bp, per unit face


def analyze(bond: Bond, ytm: float) -> BondRisk:
    """Full risk profile at a given yield."""
    m = bond.freq
    price = price_from_yield(bond, ytm)
    weighted_t = 0.0
    convex = 0.0
    for t, cf in bond.cashflows():
        disc = cf / (1 + ytm / m) ** (t * m)
        weighted_t += t * disc
        convex += t * (t + 1.0 / m) * cf / (1 + ytm / m) ** (t * m + 2)
    macaulay = weighted_t / price if price else 0.0
    modified = macaulay / (1 + ytm / m)
    convexity = convex / price if price else 0.0
    dv01 = modified * price * 1e-4
    return BondRisk(price=price, ytm=ytm, macaulay_duration=macaulay,
                    modified_duration=modified, convexity=convexity, dv01=dv01)


def price_change_estimate(risk: BondRisk, dy: float) -> float:
    """Estimated price change for a yield move ``dy`` using duration+convexity:
    dP ≈ -ModDur*P*dy + 0.5*Convexity*P*dy^2."""
    return (-risk.modified_duration * risk.price * dy
            + 0.5 * risk.convexity * risk.price * dy * dy)


@dataclass
class PortfolioRates:
    market_value: float
    dollar_duration: float        # sum of MV_i * modified_duration_i
    portfolio_duration: float     # MV-weighted modified duration
    portfolio_dv01: float
    portfolio_convexity: float


def portfolio_rate_risk(holdings: Sequence[tuple]) -> PortfolioRates:
    """Aggregate interest-rate risk across bond holdings.

    ``holdings`` is a sequence of (Bond, ytm, market_value) tuples.
    """
    total_mv = 0.0
    dollar_dur = 0.0
    dollar_conv = 0.0
    dv01 = 0.0
    for bond, ytm, mv in holdings:
        r = analyze(bond, ytm)
        total_mv += mv
        dollar_dur += mv * r.modified_duration
        dollar_conv += mv * r.convexity
        dv01 += mv / 100.0 * r.dv01 if bond.face else 0.0
    port_dur = dollar_dur / total_mv if total_mv else 0.0
    port_conv = dollar_conv / total_mv if total_mv else 0.0
    return PortfolioRates(market_value=total_mv, dollar_duration=dollar_dur,
                          portfolio_duration=port_dur, portfolio_dv01=dv01,
                          portfolio_convexity=port_conv)
