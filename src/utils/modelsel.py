from __future__ import annotations

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

from .types import FloatArray

__all__ = [
    "gaussian_pseudo_loglik",
    "student_t_pseudo_loglik",
    "information_criteria",
]


def gaussian_pseudo_loglik(u: FloatArray, rho: float) -> float:
    """
    Compute the Gaussian copula pseudo log-likelihood for bivariate data.

    Parameters
    ----------
    u:
        Pseudo-observations shaped ``(n, 2)`` with values strictly inside
        ``(0, 1)``.
    rho:
        Correlation parameter of the Gaussian copula. Must lie in ``(-1, 1)``.
    """

    u_array = np.asarray(u, dtype=np.float64)
    if u_array.ndim != 2 or u_array.shape[1] != 2:
        raise ValueError("u must be a (n, 2) array of pseudo-observations.")

    n_obs = u_array.shape[0]
    if n_obs == 0:
        raise ValueError("u must contain at least one observation.")

    if not np.isfinite(rho):
        raise ValueError("rho must be finite.")
    if not (-0.999999 < rho < 0.999999):
        raise ValueError("rho must belong to (-1, 1) to form a correlation.")

    if np.isnan(u_array).any():
        raise ValueError("u must not contain NaNs.")
    if np.any((u_array <= 0.0) | (u_array >= 1.0)):
        raise ValueError("u must take values strictly inside (0, 1).")

    eps = np.finfo(np.float64).eps
    clipped = np.clip(u_array, eps, 1.0 - eps)
    z = norm.ppf(clipped)

    rho_sq = rho * rho
    one_minus_rho_sq = 1.0 - rho_sq
    log_det = -0.5 * np.log(one_minus_rho_sq)

    z1 = z[:, 0]
    z2 = z[:, 1]
    quad_form = (
        (z1 * z1) - (2.0 * rho * z1 * z2) + (z2 * z2)
    ) / one_minus_rho_sq
    marginal_term = 0.5 * (z1 * z1 + z2 * z2)

    log_density = log_det - 0.5 * quad_form + marginal_term
    return float(np.sum(log_density))


def student_t_pseudo_loglik(u: FloatArray, rho: float, nu: float) -> float:
    """Compute the Student t copula pseudo log-likelihood for
    bivariate data."""

    u_array = np.asarray(u, dtype=np.float64)
    if u_array.ndim != 2 or u_array.shape[1] != 2:
        raise ValueError("u must be a (n, 2) array of pseudo-observations.")
    if u_array.shape[0] == 0:
        raise ValueError("u must contain at least one observation.")
    if np.isnan(u_array).any():
        raise ValueError("u must not contain NaNs.")
    if np.any((u_array <= 0.0) | (u_array >= 1.0)):
        raise ValueError("u must take values strictly inside (0, 1).")
    if not np.isfinite(rho):
        raise ValueError("rho must be finite.")
    if not (-0.999999 < rho < 0.999999):
        raise ValueError("rho must belong to (-1, 1) to form a correlation.")
    if not np.isfinite(nu):
        raise ValueError("nu must be finite.")
    if nu <= 2.0:
        raise ValueError("nu must be greater than 2.")

    from src.models.copulas.student_t import StudentTCopula

    copula = StudentTCopula(rho=float(rho), nu=float(nu))
    density = copula.pdf(np.asarray(u_array, dtype=np.float64))
    if np.any(density <= 0.0):
        raise ValueError("Copula density returned non-positive values.")
    return float(np.sum(np.log(density)))


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
