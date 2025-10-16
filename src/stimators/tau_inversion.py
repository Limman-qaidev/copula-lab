from __future__ import annotations

import numpy as np


def rho_from_tau_gaussian(tau: float) -> float:
    if not (-0.999 < tau < 0.999):
        raise ValueError("tau must be in (-1,1)")
    return float(np.sin(0.5 * np.pi * tau))


def rho_from_tau_student_t(tau: float) -> float:
    # Uses the same transformation as the Gaussian copula (elliptical margins).
    return rho_from_tau_gaussian(tau)


def choose_nu_from_tail(lambda_u: float | None) -> float:
    if lambda_u is None:
        return 10.0
    # Simple monotone mapping; can be refined once calibration docs evolve.
    return max(2.1, 2.0 / (1.0 - float(lambda_u)) - 1.0)
