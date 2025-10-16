from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kstest, norm  # type: ignore[import-untyped]


def cond_cdf_gaussian(
    u: NDArray[np.float64], rho: float
) -> NDArray[np.float64]:
    """Conditional CDF ``C_{2|1}(u₂|u₁)`` for a Gaussian copula."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be (n,2)")
    z1 = norm.ppf(np.clip(u[:, 0], 1e-12, 1 - 1e-12))
    z2 = norm.ppf(np.clip(u[:, 1], 1e-12, 1 - 1e-12))
    denom = float(np.sqrt(max(1e-16, 1.0 - rho * rho)))
    values = (z2 - rho * z1) / denom
    return np.asarray(norm.cdf(values), dtype=np.float64)


def rosenblatt_2d(
    u: NDArray[np.float64],
    cond_cdf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Compute the 2D Rosenblatt transform ``(u₁, C_{2|1}(u₂|u₁))``."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be (n,2)")
    z1 = np.clip(u[:, 0], 1e-12, 1 - 1e-12)
    z2 = np.clip(cond_cdf(u), 1e-12, 1 - 1e-12)
    return np.asarray(np.column_stack([z1, z2]), dtype=np.float64)


def gof_ks_uniform_2d(z: NDArray[np.float64]) -> float:
    """Kolmogorov–Smirnov GoF p-value for Rosenblatt components."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError("z must be (n,2)")
    p1 = kstest(z[:, 0], "uniform").pvalue
    p2 = kstest(z[:, 1], "uniform").pvalue
    return float(min(p1, p2))


def rosenblatt_gaussian(
    u: NDArray[np.float64], rho: float
) -> tuple[NDArray[np.float64], float]:
    """Rosenblatt transform and GoF p-value for a Gaussian copula."""
    z = rosenblatt_2d(u, lambda w: cond_cdf_gaussian(w, rho))
    p = gof_ks_uniform_2d(z)
    return z, p
