from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kstest, norm


def cond_cdf_gaussian(u: NDArray[np.float64], rho: float) -> NDArray[np.float64]:
    """
    C_{2|1}(u2|u1) for Gaussian copula with correlation rho:
    Let z1 = Φ^{-1}(u1), z2 = Φ^{-1}(u2).
    Then U2|U1=u1  ~  Φ( (z2 - ρ z1)/sqrt(1-ρ²) ).
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be (n,2)")
    z1 = norm.ppf(np.clip(u[:, 0], 1e-12, 1 - 1e-12))
    z2 = norm.ppf(np.clip(u[:, 1], 1e-12, 1 - 1e-12))
    denom = float(np.sqrt(max(1e-16, 1.0 - rho * rho)))
    return norm.cdf((z2 - rho * z1) / denom)


def rosenblatt_2d(u: NDArray[np.float64], cond_cdf) -> NDArray[np.float64]:
    """
    Rosenblatt transform (2D): z1 = u1, z2 = C_{2|1}(u2|u1).
    `cond_cdf(u)` must return vector of C_{2|1}(u2|u1) row-wise.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be (n,2)")
    z1 = np.clip(u[:, 0], 1e-12, 1 - 1e-12)
    z2 = np.clip(cond_cdf(u), 1e-12, 1 - 1e-12)
    return np.column_stack([z1, z2])


def gof_ks_uniform_2d(z: NDArray[np.float64]) -> float:
    """
    KS GoF on Rosenblatt components (both should be Uniform(0,1)).
    Returns min p-value across components (conservative).
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError("z must be (n,2)")
    p1 = kstest(z[:, 0], "uniform").pvalue
    p2 = kstest(z[:, 1], "uniform").pvalue
    return float(min(p1, p2))


def rosenblatt_gaussian(u: NDArray[np.float64], rho: float) -> tuple[NDArray[np.float64], float]:
    """
    Convenience: Rosenblatt for Gaussian copula + KS GoF p-value.
    Returns (z, pvalue).
    """
    z = rosenblatt_2d(u, lambda w: cond_cdf_gaussian(w, rho))
    p = gof_ks_uniform_2d(z)
    return z, p
