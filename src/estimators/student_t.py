"""Student t copula estimation helpers for arbitrary dimensions."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t  # type: ignore[import-untyped]

from src.estimators.tau_inversion import choose_nu_from_tail
from src.utils.dependence import average_tail_dep_upper, kendall_tau_matrix
from src.utils.modelsel import student_t_pseudo_loglik
from src.utils.types import FloatArray

_CLIP = 1e-12
_MIN_NU = 2.0


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


def _empirical_tail_per_pair(
    data: NDArray[np.float64], q: float = 0.95
) -> NDArray[np.float64]:
    dim = data.shape[1]
    tails: list[float] = []
    denom = max(1.0 - q, float(np.finfo(np.float64).tiny))
    for i in range(dim - 1):
        for j in range(i + 1, dim):
            mask = (data[:, i] > q) & (data[:, j] > q)
            tail = float(np.mean(mask.astype(np.float64))) / denom
            tails.append(tail)
    return np.asarray(tails, dtype=np.float64)


def _theoretical_tail_per_pair(
    corr: NDArray[np.float64], nu: float
) -> NDArray[np.float64]:
    dim = corr.shape[0]
    values: list[float] = []
    df = nu + 1.0
    for i in range(dim - 1):
        for j in range(i + 1, dim):
            rho = float(np.clip(corr[i, j], -0.999, 0.999))
            ratio = (1.0 - rho) / max(1.0 + rho, 1e-12)
            arg = np.sqrt(max((nu + 1.0) * ratio, 0.0))
            tail = 2.0 * float(student_t.cdf(-arg, df=df))
            values.append(tail)
    return np.asarray(values, dtype=np.float64)


def _refine_nu_from_tail(
    corr: NDArray[np.float64],
    nu_initial: float,
    tail_empirical: NDArray[np.float64],
) -> float:
    if tail_empirical.size == 0:
        return nu_initial
    if not np.isfinite(tail_empirical).all():
        return nu_initial

    if minimize_scalar is None:
        grid = np.linspace(_MIN_NU, 40.0, num=120)
        best = nu_initial
        best_error = float("inf")
        for nu_candidate in grid:
            theory = _theoretical_tail_per_pair(corr, nu_candidate)
            if theory.size == 0:
                continue
            error = float(np.mean((theory - tail_empirical) ** 2))
            if error < best_error:
                best_error = error
                best = float(nu_candidate)
        return best

    def objective(nu_val: float) -> float:
        if nu_val < _MIN_NU:
            return float("inf")
        theory = _theoretical_tail_per_pair(corr, nu_val)
        if theory.size == 0:
            return float("inf")
        return float(np.mean((theory - tail_empirical) ** 2))

    result = minimize_scalar(
        objective, bounds=(_MIN_NU, 60.0), method="bounded"
    )
    if result.success:
        return float(result.x)
    return nu_initial


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
    tail_pairs = _empirical_tail_per_pair(data)
    lambda_upper = float(np.mean(tail_pairs)) if tail_pairs.size > 0 else 0.0
    clipped = np.clip(data, _CLIP, 1.0 - _CLIP)

    def corr_from_nu(nu_val: float) -> NDArray[np.float64]:
        quantiles = student_t.ppf(clipped, df=nu_val)
        sample_corr = np.corrcoef(quantiles, rowvar=False)
        corr_matrix = np.asarray(sample_corr, dtype=np.float64)
        return _project_to_correlation(corr_matrix)

    def objective(nu_val: float) -> float:
        if nu_val < _MIN_NU:
            return float("inf")
        corr_candidate = corr_from_nu(nu_val)
        return -student_t_pseudo_loglik(data, corr_candidate, nu_val)

    nu_initial = float(max(_MIN_NU, choose_nu_from_tail(lambda_upper)))
    refine_tail = lambda_upper >= 0.25

    if minimize_scalar is not None:
        result = minimize_scalar(
            objective, bounds=(_MIN_NU, 60.0), method="bounded"
        )
        if result.success:
            nu_hat = float(result.x)
            corr_hat = corr_from_nu(nu_hat)
            if refine_tail:
                nu_refined = _refine_nu_from_tail(corr_hat, nu_hat, tail_pairs)
                if abs(nu_refined - nu_hat) > 1e-6:
                    nu_hat = nu_refined
                    corr_hat = corr_from_nu(nu_hat)
            return corr_hat, nu_hat

    corr = corr_from_nu(nu_initial)
    if refine_tail:
        nu_refined = _refine_nu_from_tail(corr, nu_initial, tail_pairs)
        if abs(nu_refined - nu_initial) > 1e-6:
            return corr_from_nu(nu_refined), nu_refined
    return corr, nu_initial


def _initial_guesses(
    data: NDArray[np.float64],
) -> List[Tuple[NDArray[np.float64], float]]:
    tau_matrix = kendall_tau_matrix(data)
    rho_guess = np.sin(0.5 * np.pi * tau_matrix)
    np.fill_diagonal(rho_guess, 1.0)
    rho_guess = _project_to_correlation(rho_guess)
    corr_ifm, nu_ifm = student_t_ifm(data)
    lambda_upper = average_tail_dep_upper(data)
    nu_tail = float(max(_MIN_NU, choose_nu_from_tail(lambda_upper)))
    return [
        (corr_ifm, nu_ifm),
        (rho_guess, nu_tail),
        (corr_ifm, max(_MIN_NU, nu_ifm - 2.0)),
        (corr_ifm, min(60.0, nu_ifm + 2.0)),
    ]


def student_t_pmle(u: FloatArray) -> Tuple[NDArray[np.float64], float, float]:
    """Return log-likelihood (PMLE) estimates for the Student t copula."""

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
