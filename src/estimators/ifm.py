"""Inference functions for margins (IFM) estimators."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

from src.utils.types import FloatArray


def gaussian_ifm(u: FloatArray) -> float:
    """Estimate the Gaussian copula correlation using IFM."""
    u_array = np.asarray(u, dtype=np.float64)
    if u_array.ndim != 2 or u_array.shape[1] != 2:
        raise ValueError("U must be a (n, 2) array of pseudo-observations.")
    if u_array.shape[0] < 2:
        raise ValueError("At least two observations are required.")
    if not np.isfinite(u_array).all():
        raise ValueError("U must not contain NaNs or infinite values.")
    if np.any((u_array <= 0.0) | (u_array >= 1.0)):
        raise ValueError("U entries must lie strictly between 0 and 1.")

    clipped = np.clip(u_array, 1e-12, 1.0 - 1e-12)
    z = norm.ppf(clipped)
    corr = np.corrcoef(z.T)
    return float(corr[0, 1])


def gaussian_ifm_corr(u: FloatArray) -> FloatArray:
    """Return the IFM correlation matrix for an arbitrary dimension."""

    u_array = np.asarray(u, dtype=np.float64)
    if u_array.ndim != 2:
        raise ValueError("U must be a two-dimensional array.")
    n_obs, dim = u_array.shape
    if n_obs < 2 or dim < 2:
        raise ValueError(
            "At least two observations and two dimensions are required."
        )
    if not np.isfinite(u_array).all():
        raise ValueError("U must contain only finite values.")
    if np.any((u_array <= 0.0) | (u_array >= 1.0)):
        raise ValueError("U entries must lie strictly between 0 and 1.")

    clipped = np.clip(u_array, 1e-12, 1.0 - 1e-12)
    z = norm.ppf(clipped)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    return corr
