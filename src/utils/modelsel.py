from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_t, norm  # type: ignore[import-untyped]
from scipy.stats import t as student_t

from .types import FloatArray

__all__ = [
    "gaussian_pseudo_loglik",
    "student_t_pseudo_loglik",
    "information_criteria",
]


def _validate_corr(matrix: FloatArray) -> NDArray[np.float64]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    if array.shape[0] < 2:
        raise ValueError("Correlation matrix must be at least 2x2.")
    if not np.allclose(array, array.T, atol=1e-9):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(array), 1.0, atol=1e-9):
        raise ValueError("Correlation matrix must have unit diagonal.")
    try:
        np.linalg.cholesky(array)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Correlation matrix must be positive definite."
        ) from exc
    return array


def _validate_u(u: FloatArray) -> NDArray[np.float64]:
    array = np.asarray(u, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("u must be a two-dimensional array")
    if array.shape[0] == 0:
        raise ValueError("u must contain at least one observation")
    if np.isnan(array).any():
        raise ValueError("u must not contain NaNs")
    if np.any((array <= 0.0) | (array >= 1.0)):
        raise ValueError("u must take values strictly inside (0, 1)")
    return array


def gaussian_pseudo_loglik(u: FloatArray, corr: FloatArray) -> float:
    """Compute the Gaussian copula pseudo log-likelihood in any dimension."""

    data = _validate_u(u)
    corr_matrix = _validate_corr(corr)
    if data.shape[1] != corr_matrix.shape[0]:
        raise ValueError(
            "Dimension mismatch between data and correlation matrix"
        )

    clipped = np.clip(data, 1e-12, 1.0 - 1e-12)
    z = norm.ppf(clipped)
    inv_corr = np.linalg.inv(corr_matrix)
    sign, logdet = np.linalg.slogdet(corr_matrix)
    if sign <= 0.0:
        raise ValueError("Correlation matrix must have positive determinant")

    quad = np.einsum("ij,jk,ik->i", z, inv_corr, z)
    marginal = np.sum(z**2, axis=1)
    log_density = -0.5 * logdet - 0.5 * quad + 0.5 * marginal
    return float(np.sum(log_density))


def student_t_pseudo_loglik(
    u: FloatArray, corr: FloatArray, nu: float
) -> float:
    """Compute the Student t copula pseudo log-likelihood."""

    data = _validate_u(u)
    if nu <= 2.0:
        raise ValueError("nu must be greater than 2")
    corr_matrix = _validate_corr(corr)
    if data.shape[1] != corr_matrix.shape[0]:
        raise ValueError(
            "Dimension mismatch between data and correlation matrix"
        )

    clipped = np.clip(data, 1e-12, 1.0 - 1e-12)
    x = student_t.ppf(clipped, nu)
    mv_t = multivariate_t(
        loc=np.zeros(corr_matrix.shape[0]), shape=corr_matrix, df=nu
    )
    num = np.asarray(mv_t.pdf(x), dtype=np.float64)
    den = np.prod(student_t.pdf(x, nu), axis=1)
    if np.any(num <= 0.0) or np.any(den <= 0.0):
        raise ValueError("Encountered non-positive density values")
    log_density = np.log(num) - np.log(den)
    return float(np.sum(log_density))


def information_criteria(
    loglik: float, k_params: int, n: int
) -> tuple[float, float]:
    """Return Akaike and Bayesian information criteria from log-likelihood."""

    if not np.isfinite(loglik):
        raise ValueError("loglik must be finite.")
    if k_params < 0:
        raise ValueError("k_params must be non-negative.")
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    loglik_val = float(loglik)
    k_val = float(k_params)
    n_val = float(n)

    aic = 2.0 * k_val - 2.0 * loglik_val
    bic = k_val * np.log(n_val) - 2.0 * loglik_val
    return float(aic), float(bic)
