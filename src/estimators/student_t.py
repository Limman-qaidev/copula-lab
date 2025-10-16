"""Student t copula estimation helpers for arbitrary dimensions."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.stats import t as student_t  # type: ignore[import-untyped]

from src.estimators.tau_inversion import choose_nu_from_tail
from src.utils.dependence import average_tail_dep_upper, kendall_tau_matrix
from src.utils.modelsel import student_t_pseudo_loglik
from src.utils.types import FloatArray

_CLIP = 1e-12


def _validate_u(u: FloatArray) -> NDArray[np.float64]:
    array = np.asarray(u, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("U must be a two-dimensional array.")
    if array.shape[1] < 2:
        raise ValueError("U must contain at least two dimensions.")
    if array.shape[0] < 2:
        raise ValueError("At least two observations are required.")
    if not np.isfinite(array).all():
        raise ValueError("U must contain only finite values.")
    if np.any((array <= 0.0) | (array >= 1.0)):
        raise ValueError("U entries must lie strictly inside (0, 1).")
    return array


def _project_to_correlation(
    matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 1e-8, None)
    adjusted = (eigvecs * eigvals) @ eigvecs.T
    diag = np.sqrt(np.clip(np.diag(adjusted), 1e-12, None))
    corr = adjusted / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)
    return np.asarray(corr, dtype=np.float64)


def _corr_from_params(
    params: NDArray[np.float64], dim: int
) -> NDArray[np.float64]:
    diag_params = params[:dim]
    off_params = params[dim:-1]
    chol = np.zeros((dim, dim), dtype=np.float64)
    chol[np.diag_indices(dim)] = np.exp(diag_params)
    idx = 0
    for i in range(1, dim):
        for j in range(i):
            chol[i, j] = off_params[idx]
            idx += 1
    cov = chol @ chol.T
    diag = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)
    return np.asarray(corr, dtype=np.float64)


def _nu_from_param(param: float) -> float:
    return float(2.0 + np.exp(param))


def _pack_initial(corr: NDArray[np.float64], nu: float) -> NDArray[np.float64]:
    chol = np.linalg.cholesky(corr)
    diag_params = np.log(np.diag(chol))
    off_params = chol[np.tril_indices(corr.shape[0], k=-1)]
    nu_param = np.log(max(nu - 2.0, 1e-6))
    return np.concatenate([diag_params, off_params, np.array([nu_param])])


def student_t_ifm(u: FloatArray) -> Tuple[NDArray[np.float64], float]:
    """Return IFM estimates (correlation matrix, nu) for a Student t copula."""

    data = _validate_u(u)
    lambda_upper = average_tail_dep_upper(data)
    nu_hat = float(max(2.1, choose_nu_from_tail(lambda_upper)))
    clipped = np.clip(data, _CLIP, 1.0 - _CLIP)
    quantiles = student_t.ppf(clipped, df=nu_hat)
    corr = np.corrcoef(quantiles, rowvar=False)
    corr = _project_to_correlation(np.asarray(corr, dtype=np.float64))
    return corr, nu_hat


def _initial_guesses(
    data: NDArray[np.float64],
) -> List[Tuple[NDArray[np.float64], float]]:
    tau_matrix = kendall_tau_matrix(data)
    rho_guess = np.sin(0.5 * np.pi * tau_matrix)
    np.fill_diagonal(rho_guess, 1.0)
    rho_guess = _project_to_correlation(rho_guess)
    corr_ifm, nu_ifm = student_t_ifm(data)
    lambda_upper = average_tail_dep_upper(data)
    nu_tail = float(max(2.1, choose_nu_from_tail(lambda_upper)))
    return [
        (corr_ifm, nu_ifm),
        (rho_guess, nu_tail),
        (corr_ifm, max(2.1, nu_ifm - 2.0)),
        (corr_ifm, min(60.0, nu_ifm + 2.0)),
    ]


def student_t_pmle(u: FloatArray) -> Tuple[NDArray[np.float64], float, float]:
    """Return PMLE estimates (correlation, nu, loglik) for Student t copula."""

    data = _validate_u(u)
    dim = data.shape[1]

    best_loglik = -np.inf
    best_corr: NDArray[np.float64] | None = None
    best_nu: float | None = None

    for corr0, nu0 in _initial_guesses(data):
        x0 = _pack_initial(corr0, nu0)

        def objective(theta: NDArray[np.float64]) -> float:
            corr = _corr_from_params(theta, dim)
            nu_param = _nu_from_param(float(theta[-1]))
            try:
                loglik = student_t_pseudo_loglik(data, corr, nu_param)
            except ValueError:
                return np.inf
            return -loglik

        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            options={"maxiter": 300},
        )

        if not result.success:
            continue

        corr_hat = _corr_from_params(result.x, dim)
        nu_hat = _nu_from_param(float(result.x[-1]))
        loglik = -float(result.fun)
        if loglik > best_loglik:
            best_loglik = loglik
            best_corr = corr_hat
            best_nu = nu_hat

    if best_corr is None or best_nu is None:
        raise ValueError(
            "Student t PMLE failed to converge from initial guesses."
        )

    return best_corr, best_nu, best_loglik


__all__ = ["student_t_ifm", "student_t_pmle"]
