"""Portfolio optimization (pure Python).

Implements minimum-variance, mean-variance and risk-parity portfolios with box
constraints (per-name bounds) and a budget constraint (weights sum to a target).
Constrained problems are solved by projected gradient descent with projection
onto the capped simplex — adequate for the small, well-conditioned problems here.

For large universes or richer constraints (turnover, sector neutrality as hard
constraints), swap in cvxpy (see docs/03 §0.3 and requirements.txt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import linalg, stats

Vector = List[float]
Matrix = List[List[float]]


@dataclass
class Constraints:
    weight_min: float = 0.0          # per-name lower bound (0 => long-only)
    weight_max: float = 1.0          # per-name upper bound
    total: float = 1.0               # weights must sum to this (budget)
    long_only: bool = True
    per_name_min: Dict[str, float] = field(default_factory=dict)
    per_name_max: Dict[str, float] = field(default_factory=dict)

    def bounds(self, symbols: Sequence[str]) -> (Vector, Vector):  # type: ignore
        lo = []
        hi = []
        for s in symbols:
            lo_i = self.per_name_min.get(s, self.weight_min if self.long_only else -abs(self.weight_max))
            hi_i = self.per_name_max.get(s, self.weight_max)
            lo.append(lo_i)
            hi.append(hi_i)
        return lo, hi


def project_capped_simplex(v: Sequence[float], lo: Sequence[float], hi: Sequence[float],
                           total: float = 1.0, iters: int = 100) -> Vector:
    """Project v onto {w : sum(w) = total, lo_i <= w_i <= hi_i} via bisection on
    a single Lagrange multiplier tau, where w_i = clip(v_i - tau, lo_i, hi_i)."""
    n = len(v)

    def w_of(tau: float) -> Vector:
        return [min(max(v[i] - tau, lo[i]), hi[i]) for i in range(n)]

    def s_of(tau: float) -> float:
        return sum(w_of(tau))

    # Bracket tau: increasing tau decreases the sum.
    lo_tau = min(v[i] - hi[i] for i in range(n)) - 1.0
    hi_tau = max(v[i] - lo[i] for i in range(n)) + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo_tau + hi_tau)
        if s_of(mid) > total:
            lo_tau = mid
        else:
            hi_tau = mid
    return w_of(0.5 * (lo_tau + hi_tau))


def _cov_psd(cov: Matrix) -> Matrix:
    return linalg.make_psd(cov, 1e-8)


def closed_form_min_variance(cov: Matrix) -> Vector:
    """Unconstrained (sum=1) global minimum-variance weights: inv(S)1 / 1'inv(S)1."""
    n = len(cov)
    ones = [1.0] * n
    inv_one = linalg.solve(_cov_psd(cov), ones)
    denom = sum(inv_one)
    if denom == 0:
        return [1.0 / n] * n
    return [x / denom for x in inv_one]


def _projected_gradient(grad_fn, x0: Vector, lo: Vector, hi: Vector, total: float,
                        step: float = 0.1, iters: int = 2000, tol: float = 1e-10) -> Vector:
    w = project_capped_simplex(x0, lo, hi, total)
    prev = None
    for _ in range(iters):
        g = grad_fn(w)
        trial = [w[i] - step * g[i] for i in range(len(w))]
        w_new = project_capped_simplex(trial, lo, hi, total)
        if prev is not None:
            diff = sum((w_new[i] - w[i]) ** 2 for i in range(len(w)))
            if diff < tol:
                w = w_new
                break
        prev = w
        w = w_new
    return w


def min_variance(cov: Matrix, symbols: Sequence[str],
                 constraints: Optional[Constraints] = None) -> Dict[str, float]:
    """Minimum-variance portfolio. Honors box + budget constraints."""
    c = constraints or Constraints()
    n = len(cov)
    psd = _cov_psd(cov)
    if c.long_only or c.per_name_min or c.per_name_max or c.weight_max < 1.0:
        lo, hi = c.bounds(symbols)
        x0 = [c.total / n] * n

        def grad(w):  # d/dw (0.5 w'Sw) = Sw
            return linalg.matvec(psd, w)
        w = _projected_gradient(grad, x0, lo, hi, c.total)
    else:
        w = closed_form_min_variance(psd)
        if c.total != 1.0:
            w = [x * c.total for x in w]
    return dict(zip(symbols, w))


def mean_variance(mu: Sequence[float], cov: Matrix, symbols: Sequence[str],
                  risk_aversion: float = 5.0,
                  constraints: Optional[Constraints] = None) -> Dict[str, float]:
    """Maximize mu'w - 0.5*lambda*w'Sw  (equivalently minimize the negative)."""
    c = constraints or Constraints()
    n = len(cov)
    psd = _cov_psd(cov)
    lo, hi = c.bounds(symbols)
    x0 = [c.total / n] * n

    def grad(w):  # grad of f(w) = 0.5*lambda*w'Sw - mu'w  is  lambda*Sw - mu
        return [risk_aversion * v - mu[i] for i, v in enumerate(linalg.matvec(psd, w))]
    w = _projected_gradient(grad, x0, lo, hi, c.total)
    return dict(zip(symbols, w))


def max_sharpe(mu: Sequence[float], cov: Matrix, symbols: Sequence[str], rf: float = 0.0,
               constraints: Optional[Constraints] = None) -> Dict[str, float]:
    """Approximate max-Sharpe by scanning risk-aversion and keeping the best
    ex-ante Sharpe. Robust and good enough for personal-scale problems."""
    c = constraints or Constraints()
    excess = [m - rf for m in mu]
    best_w, best_sharpe = None, -float("inf")
    for la in (0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55):
        w_map = mean_variance(excess, cov, symbols, risk_aversion=la, constraints=c)
        w = [w_map[s] for s in symbols]
        ret = sum(w[i] * excess[i] for i in range(len(w)))
        var = linalg.quadratic_form(w, _cov_psd(cov))
        if var <= 0:
            continue
        sharpe = ret / (var ** 0.5)
        if sharpe > best_sharpe:
            best_sharpe, best_w = sharpe, w_map
    return best_w or {s: c.total / len(symbols) for s in symbols}


def risk_parity(cov: Matrix, symbols: Sequence[str],
                budgets: Optional[Sequence[float]] = None, total: float = 1.0,
                iters: int = 5000, tol: float = 1e-12) -> Dict[str, float]:
    """Equal (or budgeted) risk-contribution portfolio.

    Multiplicative update w_i <- w_i * sqrt(b_i / share_i), where share_i is the
    current fraction of total risk borne by asset i. Converges monotonically to
    the risk-budgeting solution for a PSD covariance, long-only.
    """
    n = len(cov)
    psd = _cov_psd(cov)
    b = list(budgets) if budgets else [1.0 / n] * n
    bs = sum(b)
    b = [x / bs for x in b]  # normalise budgets to sum 1
    w = [1.0 / n] * n
    for _ in range(iters):
        sw = linalg.matvec(psd, w)
        rc = [w[i] * sw[i] for i in range(n)]
        total_rc = sum(rc)
        if total_rc <= 0:
            break
        shares = [r / total_rc for r in rc]
        w_new = [w[i] * (b[i] / shares[i]) ** 0.5 if shares[i] > 1e-15 else w[i]
                 for i in range(n)]
        s = sum(w_new)
        w_new = [x / s for x in w_new]
        if sum((w_new[i] - w[i]) ** 2 for i in range(n)) < tol:
            w = w_new
            break
        w = w_new
    w = [x * total for x in w]
    return dict(zip(symbols, w))


# ---- Convenience wrappers from return series ------------------------------

def optimize_from_returns(rets: Dict[str, List[float]], symbols: Sequence[str],
                          method: str = "min_variance",
                          mu: Optional[Dict[str, float]] = None,
                          risk_aversion: float = 5.0, rf: float = 0.0,
                          constraints: Optional[Constraints] = None,
                          shrink: float = 0.1) -> Dict[str, float]:
    syms = [s for s in symbols if s in rets]
    n = min(len(rets[s]) for s in syms)
    series = [rets[s][-n:] for s in syms]
    cov = stats.shrink_cov(series, shrink)
    if method == "min_variance":
        return min_variance(cov, syms, constraints)
    if method == "risk_parity":
        return risk_parity(cov, syms, total=(constraints.total if constraints else 1.0))
    expected = [(mu or {}).get(s, stats.mean(rets[s][-n:])) for s in syms]
    if method == "mean_variance":
        return mean_variance(expected, cov, syms, risk_aversion, constraints)
    if method == "max_sharpe":
        return max_sharpe(expected, cov, syms, rf, constraints)
    raise ValueError(f"unknown method: {method}")
