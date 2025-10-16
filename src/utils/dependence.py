"""Dependence summaries for pseudo-observations."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau  # type: ignore[import-untyped]

from .types import FloatArray

__all__ = [
    "kendall_tau",
    "spearman_rho",
    "tail_dep_lower",
    "tail_dep_upper",
    "kendall_tau_matrix",
    "average_kendall_tau",
    "average_tail_dep_upper",
]


def _validate_bivariate(u: FloatArray) -> NDArray[np.float64]:
    array = np.asarray(u, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("u must be a (n, 2) array of pseudo-observations")
    return array


def _iter_pairs(dim: int) -> Iterable[Tuple[int, int]]:
    return combinations(range(dim), 2)


def kendall_tau(u: FloatArray) -> float:
    """Return Kendall's tau for bivariate pseudo-observations."""

    array = _validate_bivariate(u)
    tau, _ = kendalltau(array[:, 0], array[:, 1])
    return float(tau)


def kendall_tau_matrix(u: FloatArray) -> NDArray[np.float64]:
    """Compute the symmetric Kendall's tau matrix for any dimension."""

    array = np.asarray(u, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("u must be a two-dimensional array")

    n_obs, dim = array.shape
    if n_obs < 2 or dim < 2:
        raise ValueError(
            "At least two observations and two dimensions required"
        )

    tau_matrix = np.eye(dim, dtype=np.float64)
    for i, j in _iter_pairs(dim):
        tau, _ = kendalltau(array[:, i], array[:, j])
        tau_matrix[i, j] = tau_matrix[j, i] = float(tau)
    return tau_matrix


def average_kendall_tau(u: FloatArray) -> float:
    """Return the average pairwise Kendall's tau across dimensions."""

    array = np.asarray(u, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("u must be (n, d) with d >= 2")

    taus = []
    for i, j in _iter_pairs(array.shape[1]):
        tau, _ = kendalltau(array[:, i], array[:, j])
        taus.append(float(tau))
    return float(np.mean(taus))


def spearman_rho(u: FloatArray) -> float:
    """Return Spearman's rho for bivariate pseudo-observations."""

    array = _validate_bivariate(u)
    r1 = np.argsort(np.argsort(array[:, 0])) + 1
    r2 = np.argsort(np.argsort(array[:, 1])) + 1
    n = array.shape[0]
    num = 6.0 * np.sum((r1 - r2) ** 2)
    den = n * (n * n - 1.0)
    return float(1.0 - num / den) if den > 0 else 0.0


def tail_dep_lower(u: FloatArray, q: float = 0.05) -> float:
    """Return the lower tail dependence for bivariate pseudo-observations."""

    array = _validate_bivariate(u)
    th = float(q)
    sel = (array[:, 0] <= th) & (array[:, 1] <= th)
    return float(np.mean(sel) / th) if 0.0 < th < 1.0 else 0.0


def tail_dep_upper(u: FloatArray, q: float = 0.95) -> float:
    """Return the upper tail dependence for bivariate pseudo-observations."""

    array = _validate_bivariate(u)
    th = float(q)
    sel = (array[:, 0] > th) & (array[:, 1] > th)
    return float(np.mean(sel) / (1.0 - th)) if 0.0 < th < 1.0 else 0.0


def average_tail_dep_upper(u: FloatArray, q: float = 0.95) -> float:
    """Average upper tail dependence across all dimension pairs."""

    array = np.asarray(u, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("u must be (n, d) with d >= 2")

    pairs = list(_iter_pairs(array.shape[1]))
    if not pairs:
        return 0.0

    tail_values = []
    for i, j in pairs:
        sel = (array[:, i] > q) & (array[:, j] > q)
        denom = 1.0 - float(q)
        if denom <= 0.0:
            denom = float(np.finfo(np.float64).tiny)
        tail_rate = float(np.mean(sel.astype(np.float64)))
        tail_values.append(tail_rate / denom)
    return float(np.mean(tail_values))
