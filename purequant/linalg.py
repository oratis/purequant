"""Pure-Python linear algebra (stdlib only).

Vectors are ``list[float]``; matrices are ``list[list[float]]`` (row-major).
Sized for personal-scale portfolios (tens to low hundreds of assets) at daily
frequency. For large-scale backtests, swap these helpers for numpy — the engine
interfaces consume plain lists, so the substitution is local.
"""
from __future__ import annotations

from typing import List, Sequence

Vector = List[float]
Matrix = List[List[float]]


def zeros(n: int) -> Vector:
    return [0.0] * n


def zeros_matrix(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def shape(a: Matrix) -> tuple:
    return (len(a), len(a[0]) if a else 0)


def transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dot length mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def matvec(a: Matrix, x: Sequence[float]) -> Vector:
    return [dot(row, x) for row in a]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    bt = transpose(b)
    return [[dot(row, col) for col in bt] for row in a]


def add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x + y for x, y in zip(a, b)]


def sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x - y for x, y in zip(a, b)]


def scale(a: Sequence[float], s: float) -> Vector:
    return [x * s for x in a]


def outer(a: Sequence[float], b: Sequence[float]) -> Matrix:
    return [[x * y for y in b] for x in a]


def quadratic_form(x: Sequence[float], a: Matrix) -> float:
    """Compute x^T A x."""
    return dot(x, matvec(a, x))


def solve(a: Matrix, b: Sequence[float]) -> Vector:
    """Solve A x = b via Gaussian elimination with partial pivoting."""
    n = len(a)
    # Build augmented matrix (copy to avoid mutating inputs).
    m = [list(a[i]) + [float(b[i])] for i in range(n)]
    for col in range(n):
        # Partial pivot: pick the largest magnitude entry in this column.
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-15:
            raise ValueError("matrix is singular or near-singular")
        m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / piv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def inverse(a: Matrix) -> Matrix:
    """Invert a square matrix by solving against the identity, column by column."""
    n = len(a)
    cols = []
    eye = identity(n)
    for j in range(n):
        e_j = [eye[i][j] for i in range(n)]
        cols.append(solve(a, e_j))
    # cols[j] is the j-th column of the inverse.
    return transpose(cols)


def is_symmetric(a: Matrix, tol: float = 1e-9) -> bool:
    n = len(a)
    return all(abs(a[i][j] - a[j][i]) <= tol for i in range(n) for j in range(i + 1, n))


def make_psd(a: Matrix, ridge: float = 1e-8) -> Matrix:
    """Nudge a covariance-like matrix toward positive definiteness by adding a
    small ridge to the diagonal. Keeps ``solve``/``inverse`` numerically safe
    when the sample covariance is rank-deficient (T < N)."""
    n = len(a)
    return [[a[i][j] + (ridge if i == j else 0.0) for j in range(n)] for i in range(n)]
