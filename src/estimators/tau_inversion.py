"""Kendall's tau inversion helpers for common copula families."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq  # type: ignore[import-untyped]

from src.utils.types import FloatArray

_TAU_EPS = 1e-9


def _validate_tau(tau: float) -> float:
    value = float(tau)
    if not (-1.0 + _TAU_EPS < value < 1.0 - _TAU_EPS):
        raise ValueError("tau must lie in (-1, 1)")
    return value


def _ensure_correlation(matrix: FloatArray) -> NDArray[np.float64]:
    sym = np.asarray(matrix, dtype=np.float64)
    if sym.ndim != 2 or sym.shape[0] != sym.shape[1]:
        raise ValueError("Matrix must be square to form a correlation matrix.")

    sym = 0.5 * (sym + sym.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 1e-8, None)
    adjusted = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(adjusted), 1e-12, None))
    corr = adjusted / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.asarray(corr, dtype=np.float64)


def _debye1(theta: float) -> float:
    value = float(theta)
    if abs(value) < 1e-6:
        return 1.0 - value / 4.0 + (value * value) / 36.0
    grid = np.linspace(0.0, value, num=2048)
    integrand = np.empty_like(grid)
    mask = np.isclose(grid, 0.0)
    integrand[mask] = 1.0
    integrand[~mask] = grid[~mask] / np.expm1(grid[~mask])
    integral = float(np.trapezoid(integrand, grid))
    return integral / value


def rho_from_tau_gaussian(tau: float) -> float:
    """Map Kendall's tau to Pearson's rho for the Gaussian copula."""

    value = _validate_tau(tau)
    return float(math.sin(0.5 * math.pi * value))


def rho_matrix_from_tau_gaussian(
    tau_matrix: FloatArray,
) -> NDArray[np.float64]:
    """Return a correlation matrix from a Kendall's tau matrix."""

    array = np.asarray(tau_matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("tau_matrix must be square")
    dim = array.shape[0]
    off_diag = array[~np.eye(dim, dtype=bool)]
    if np.any(off_diag <= -1.0) or np.any(off_diag >= 1.0):
        raise ValueError("tau values must lie strictly inside (-1, 1)")

    rho = np.sin(0.5 * math.pi * array)
    np.fill_diagonal(rho, 1.0)
    return _ensure_correlation(rho)


def rho_from_tau_student_t(tau: float) -> float:
    """Use the same tau-to-rho mapping as the Gaussian copula."""

    return rho_from_tau_gaussian(tau)


def rho_matrix_from_tau_student_t(
    tau_matrix: FloatArray,
) -> NDArray[np.float64]:
    """Return a Student t correlation matrix derived from Kendall's tau."""

    return rho_matrix_from_tau_gaussian(tau_matrix)


_MIN_STUDENT_NU = 2.0


def choose_nu_from_tail(lambda_u: float | None) -> float:
    """Select degrees of freedom from an optional upper tail dependence."""

    if lambda_u is None:
        return 10.0

    tail = max(0.0, min(0.99, float(lambda_u)))
    # Monotone map: larger tail dependence yields smaller nu (heavier tails).
    nu = 10.0 - 6.0 * tail
    return float(max(_MIN_STUDENT_NU, nu))


def theta_from_tau_clayton(tau: float) -> float:
    """Map Kendall's tau to Clayton's ``theta`` (positive dependence only)."""

    value = _validate_tau(tau)
    if value <= 0.0:
        raise ValueError("Clayton copula requires tau in (0, 1)")
    theta = 2.0 * value / (1.0 - value)
    return float(max(theta, 1e-6))


def theta_from_tau_gumbel(tau: float) -> float:
    """Map Kendall's tau to Gumbel's ``theta`` parameter."""

    value = _validate_tau(tau)
    if value < 0.0:
        raise ValueError("Gumbel copula requires non-negative tau")
    theta = 1.0 / (1.0 - value)
    return float(max(theta, 1.0))


def _tau_frank(theta: float) -> float:
    if abs(theta) < 1e-8:
        return 0.0
    if abs(theta) < 1e-3:
        return theta / 9.0
    d1 = _debye1(theta)
    return 1.0 - 4.0 / theta + 4.0 * d1 / (theta * theta)


def theta_from_tau_frank(tau: float) -> float:
    """Numerically invert Kendall's tau for the Frank copula."""

    value = _validate_tau(tau)
    if abs(value) < 1e-8:
        return 0.0

    def objective(theta: float) -> float:
        return _tau_frank(theta) - value

    if value > 0.0:
        bracket = (1e-6, 60.0)
    else:
        bracket = (-60.0, -1e-6)

    return float(brentq(objective, *bracket, maxiter=1000))


def _tau_joe(theta: float) -> float:
    if theta <= 1.0:
        return 0.0
    total = 0.0
    for k in range(1, 10000):
        term = 1.0 / (k * (k + 1.0) * (theta * k + 2.0))
        total += term
        if term < 1e-12:
            break
    return 1.0 - 4.0 * total


def theta_from_tau_joe(tau: float) -> float:
    """Invert Kendall's tau for the Joe copula."""

    value = _validate_tau(tau)
    if value < 0.0:
        raise ValueError("Joe copula requires non-negative tau")
    if value < 1e-9:
        return 1.0

    def objective(theta: float) -> float:
        return _tau_joe(theta) - value

    return float(brentq(objective, 1.0 + 1e-6, 1e3, maxiter=1000))


def _tau_amh(theta: float) -> float:
    if abs(theta) < 1e-8:
        return 0.0
    one_minus = max(1.0 - theta, 1e-12)
    term_log = (2.0 * one_minus * one_minus) / (3.0 * theta * theta)
    term_linear = (4.0 * one_minus) / (3.0 * theta)
    return 1.0 - term_log * math.log(one_minus) - term_linear


def theta_from_tau_amh(tau: float) -> float:
    """Invert Kendall's tau for the AMH copula."""

    value = _validate_tau(tau)
    if abs(value) < 1e-9:
        return 0.0
    if value <= 0.0:
        raise ValueError("AMH copula currently supports positive tau only")

    def objective(theta: float) -> float:
        return _tau_amh(theta) - value

    return float(brentq(objective, 1e-6, 0.999999, maxiter=1000))


__all__ = [
    "rho_from_tau_gaussian",
    "rho_matrix_from_tau_gaussian",
    "rho_from_tau_student_t",
    "rho_matrix_from_tau_student_t",
    "choose_nu_from_tail",
    "theta_from_tau_clayton",
    "theta_from_tau_gumbel",
    "theta_from_tau_frank",
    "theta_from_tau_joe",
    "theta_from_tau_amh",
]
