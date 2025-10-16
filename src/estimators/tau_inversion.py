"""Kendall's tau inversion helpers for elliptical copulas."""

from __future__ import annotations

import math

_TAU_EPS = 1e-9


def _validate_tau(tau: float) -> float:
    value = float(tau)
    if not (-1.0 + _TAU_EPS < value < 1.0 - _TAU_EPS):
        raise ValueError("tau must lie in (-1, 1)")
    return value


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
