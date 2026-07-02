"""Pure-Python statistics for quant analytics (stdlib only)."""
from __future__ import annotations

import math
from typing import List, Sequence

from . import linalg

Vector = List[float]
Matrix = List[List[float]]

TRADING_DAYS = 252


def mean(x: Sequence[float]) -> float:
    if not x:
        raise ValueError("mean of empty sequence")
    return sum(x) / len(x)


def variance(x: Sequence[float], ddof: int = 1) -> float:
    n = len(x)
    if n - ddof <= 0:
        raise ValueError("not enough data for variance")
    mu = mean(x)
    return sum((v - mu) ** 2 for v in x) / (n - ddof)


def std(x: Sequence[float], ddof: int = 1) -> float:
    return math.sqrt(variance(x, ddof))


def covariance(x: Sequence[float], y: Sequence[float], ddof: int = 1) -> float:
    if len(x) != len(y):
        raise ValueError("covariance length mismatch")
    n = len(x)
    if n - ddof <= 0:
        raise ValueError("not enough data for covariance")
    mx, my = mean(x), mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (n - ddof)


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    sx, sy = std(x), std(y)
    if sx == 0 or sy == 0:
        return 0.0
    return covariance(x, y) / (sx * sy)


def cov_matrix(series: Sequence[Sequence[float]], ddof: int = 1) -> Matrix:
    """Covariance matrix of a list of equal-length series (one per asset)."""
    n = len(series)
    return [[covariance(series[i], series[j], ddof) for j in range(n)] for i in range(n)]


def corr_matrix(series: Sequence[Sequence[float]]) -> Matrix:
    n = len(series)
    return [[correlation(series[i], series[j]) for j in range(n)] for i in range(n)]


def shrink_cov(series: Sequence[Sequence[float]], intensity: float = 0.1) -> Matrix:
    """Ledoit-Wolf-style shrinkage toward a constant-correlation target.

    A practical stabiliser when the number of observations is small relative to
    the number of assets. ``intensity`` in [0, 1]: 0 = sample cov, 1 = target.
    """
    cov = cov_matrix(series)
    n = len(cov)
    variances = [cov[i][i] for i in range(n)]
    stds = [math.sqrt(v) if v > 0 else 0.0 for v in variances]
    # Average pairwise correlation as the shrinkage target structure.
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            if stds[i] > 0 and stds[j] > 0:
                corrs.append(cov[i][j] / (stds[i] * stds[j]))
    avg_corr = mean(corrs) if corrs else 0.0
    target = [[(variances[i] if i == j else avg_corr * stds[i] * stds[j])
               for j in range(n)] for i in range(n)]
    return [[(1 - intensity) * cov[i][j] + intensity * target[i][j]
             for j in range(n)] for i in range(n)]


def zscore(x: Sequence[float]) -> Vector:
    mu = mean(x)
    sd = std(x)
    if sd == 0:
        return [0.0 for _ in x]
    return [(v - mu) / sd for v in x]


def rank(x: Sequence[float]) -> Vector:
    """Average ranks in [0, 1] (ties share the mean rank). Used for Rank-IC."""
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    if n > 1:
        return [r / (n - 1) for r in ranks]
    return [0.0]


def winsorize(x: Sequence[float], lower: float = 0.01, upper: float = 0.99) -> Vector:
    s = sorted(x)
    lo = quantile(s, lower)
    hi = quantile(s, upper)
    return [min(max(v, lo), hi) for v in x]


def quantile(sorted_or_unsorted: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile. Accepts unsorted input."""
    if not sorted_or_unsorted:
        raise ValueError("quantile of empty sequence")
    s = sorted(sorted_or_unsorted)
    if q <= 0:
        return s[0]
    if q >= 1:
        return s[-1]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ---- Normal distribution (no scipy) ---------------------------------------

def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse normal CDF via Acklam's rational approximation (|err| < 1e-9)."""
    if p <= 0.0 or p >= 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("normal_ppf domain is (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ---- Regression -----------------------------------------------------------

class OLSResult:
    def __init__(self, beta: Vector, intercept: float, r2: float, resid: Vector):
        self.beta = beta            # slope coefficients (one per regressor)
        self.intercept = intercept
        self.r2 = r2
        self.resid = resid

    def __repr__(self) -> str:
        return f"OLSResult(intercept={self.intercept:.4f}, beta={self.beta}, r2={self.r2:.4f})"


def ols(y: Sequence[float], x: Sequence[Sequence[float]], add_intercept: bool = True) -> OLSResult:
    """Ordinary least squares. ``x`` is a list of regressor columns.

    Solves the normal equations (X^T X) b = X^T y. Adequate for the small,
    well-conditioned designs used here (a handful of factors / one benchmark).
    """
    n = len(y)
    cols = [list(c) for c in x]
    if add_intercept:
        cols = [[1.0] * n] + cols
    k = len(cols)
    # Build X^T X and X^T y.
    xtx = [[sum(cols[i][t] * cols[j][t] for t in range(n)) for j in range(k)] for i in range(k)]
    xty = [sum(cols[i][t] * y[t] for t in range(n)) for i in range(k)]
    coef = linalg.solve(linalg.make_psd(xtx, 1e-12), xty)
    fitted = [sum(coef[i] * cols[i][t] for i in range(k)) for t in range(n)]
    resid = [y[t] - fitted[t] for t in range(n)]
    ybar = mean(y)
    ss_tot = sum((v - ybar) ** 2 for v in y)
    ss_res = sum(r * r for r in resid)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if add_intercept:
        return OLSResult(coef[1:], coef[0], r2, resid)
    return OLSResult(coef, 0.0, r2, resid)


def annualize_return(daily_mean: float, periods: int = TRADING_DAYS) -> float:
    return (1 + daily_mean) ** periods - 1


def annualize_vol(daily_std: float, periods: int = TRADING_DAYS) -> float:
    return daily_std * math.sqrt(periods)
