"""Kendall's tau inversion helpers for elliptical copulas."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq  # type: ignore[import-untyped]

_TAU_EPS = 1e-9


def _validate_tau(tau: float) -> float:
    value = float(tau)
    if not (-1.0 + _TAU_EPS < value < 1.0 - _TAU_EPS):
        raise ValueError("tau must lie in (-1, 1)")
    return value


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


def rho_from_tau_student_t(tau: float) -> float:
    """Use the same tau-to-rho mapping as the Gaussian copula."""

    return rho_from_tau_gaussian(tau)


def choose_nu_from_tail(lambda_u: float | None) -> float:
    """Select degrees of freedom from an optional upper tail dependence."""

    if lambda_u is None:
        return 10.0

    tail = max(0.0, min(0.99, float(lambda_u)))
    # Monotone map: larger tail dependence yields smaller nu (heavier tails).
    nu = 10.0 - 6.0 * tail
    return float(max(2.1, nu))


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


__all__ = [
    "rho_from_tau_gaussian",
    "rho_from_tau_student_t",
    "choose_nu_from_tail",
    "theta_from_tau_clayton",
    "theta_from_tau_gumbel",
    "theta_from_tau_frank",
]
