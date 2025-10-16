from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def kendall_tau(u: NDArray[np.float64]) -> float:
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be (n,2)")
    x, y = u[:, 0], u[:, 1]
    n = x.shape[0]
    s = 0
    for i in range(n - 1):
        future = slice(i + 1, None)
        dx = x[future] - x[i]
        dy = y[future] - y[i]
        s += int(np.sum(np.sign(dx * dy)))
    denom = n * (n - 1) / 2.0
    return float(s / denom) if denom > 0 else 0.0


def spearman_rho(u: NDArray[np.float64]) -> float:
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be (n,2)")
    r1 = np.argsort(np.argsort(u[:, 0])) + 1
    r2 = np.argsort(np.argsort(u[:, 1])) + 1
    n = u.shape[0]
    num = 6.0 * np.sum((r1 - r2) ** 2)
    den = n * (n * n - 1.0)
    return float(1.0 - num / den) if den > 0 else 0.0


def tail_dep_lower(u: NDArray[np.float64], q: float = 0.05) -> float:
    u = np.asarray(u, dtype=float)
    th = float(q)
    sel = (u[:, 0] <= th) & (u[:, 1] <= th)
    return float(np.mean(sel) / th) if 0.0 < th < 1.0 else 0.0


def tail_dep_upper(u: NDArray[np.float64], q: float = 0.95) -> float:
    u = np.asarray(u, dtype=float)
    th = float(q)
    sel = (u[:, 0] > th) & (u[:, 1] > th)
    return float(np.mean(sel) / (1.0 - th)) if 0.0 < th < 1.0 else 0.0
