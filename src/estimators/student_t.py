"""Student t copula estimation helpers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.stats import t as student_t  # type: ignore[import-untyped]

from src.estimators.tau_inversion import (
    choose_nu_from_tail,
    rho_from_tau_student_t,
)
from src.utils.dependence import kendall_tau, tail_dep_upper
from src.utils.modelsel import student_t_pseudo_loglik
from src.utils.types import FloatArray

_CLIP = 1e-12


def _validate_u(u: FloatArray) -> np.ndarray:
    array = np.asarray(u, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("U must be a (n, 2) array of pseudo-observations.")
    if array.shape[0] < 2:
        raise ValueError("At least two observations are required.")
    if not np.isfinite(array).all():
        raise ValueError("U must contain only finite values.")
    if np.any((array <= 0.0) | (array >= 1.0)):
        raise ValueError("U entries must lie strictly inside (0, 1).")
    return array


def student_t_ifm(u: FloatArray) -> Tuple[float, float]:
    """Return IFM estimates ``(rho_hat, nu_hat)`` for a Student t copula."""

    data = _validate_u(u)
    lambda_upper = tail_dep_upper(data)
    nu_hat = float(max(2.1, choose_nu_from_tail(lambda_upper)))
    clipped = np.clip(data, _CLIP, 1.0 - _CLIP)
    quantiles = student_t.ppf(clipped, df=nu_hat)
    corr = np.corrcoef(quantiles.T)
    rho_hat = float(np.clip(corr[0, 1], -0.999, 0.999))
    return rho_hat, nu_hat


def _initial_guesses(data: np.ndarray) -> List[Tuple[float, float]]:
    tau = kendall_tau(data)
    rho_tau = rho_from_tau_student_t(tau)
    nu_tail = float(max(2.1, choose_nu_from_tail(tail_dep_upper(data))))
    rho_ifm, nu_ifm = student_t_ifm(data)

    guesses = {
        (rho_tau, nu_tail),
        (rho_ifm, nu_ifm),
        (rho_tau, max(2.1, nu_tail - 2.0)),
        (rho_tau, max(2.1, nu_tail + 2.0)),
    }
    clipped = [
        (float(np.clip(rho, -0.95, 0.95)), float(max(2.1, nu)))
        for rho, nu in guesses
    ]
    return clipped


def _objective(params: np.ndarray, data: np.ndarray) -> float:
    rho, nu = float(params[0]), float(params[1])
    if not (-0.999 < rho < 0.999) or nu <= 2.0:
        return np.inf
    try:
        loglik = student_t_pseudo_loglik(data, rho, nu)
    except ValueError:
        return np.inf
    return -loglik


def student_t_pmle(u: FloatArray) -> Tuple[float, float, float]:
    """Return PMLE estimates ``(rho_hat, nu_hat, loglik)`` for Student t."""

    data = _validate_u(u)
    best_loglik = -np.inf
    best_params: Tuple[float, float] | None = None

    for rho0, nu0 in _initial_guesses(data):
        result = minimize(
            _objective,
            x0=np.array([rho0, nu0], dtype=np.float64),
            args=(data,),
            method="L-BFGS-B",
            bounds=((-0.995, 0.995), (2.05, 60.0)),
            options={"maxiter": 500},
        )
        if not result.success:
            continue
        loglik = -float(result.fun)
        if loglik > best_loglik:
            best_loglik = loglik
            best_params = (float(result.x[0]), float(result.x[1]))

    if best_params is None:
        raise ValueError(
            "Student t PMLE failed to converge from initial guesses."
        )

    rho_hat, nu_hat = best_params
    return rho_hat, nu_hat, best_loglik
