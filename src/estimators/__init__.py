"""Estimator helper utilities."""

from __future__ import annotations

from .ifm import gaussian_ifm
from .student_t import student_t_ifm, student_t_pmle
from .tau_inversion import (
    choose_nu_from_tail,
    rho_from_tau_gaussian,
    rho_from_tau_student_t,
    theta_from_tau_clayton,
    theta_from_tau_frank,
    theta_from_tau_gumbel,
)

__all__ = [
    "gaussian_ifm",
    "student_t_ifm",
    "student_t_pmle",
    "choose_nu_from_tail",
    "rho_from_tau_gaussian",
    "rho_from_tau_student_t",
    "theta_from_tau_clayton",
    "theta_from_tau_gumbel",
    "theta_from_tau_frank",
]
