"""Backward-compatible re-export of tau inversion helpers."""

from __future__ import annotations

from src.estimators.tau_inversion import (  # noqa: F401
    choose_nu_from_tail,
    rho_from_tau_gaussian,
    rho_from_tau_student_t,
    theta_from_tau_clayton,
    theta_from_tau_frank,
    theta_from_tau_gumbel,
)

__all__ = [
    "choose_nu_from_tail",
    "rho_from_tau_gaussian",
    "rho_from_tau_student_t",
    "theta_from_tau_clayton",
    "theta_from_tau_gumbel",
    "theta_from_tau_frank",
]
