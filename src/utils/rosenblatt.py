from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]

from src.models.copulas.archimedean import (
    AMHCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)

from .types import FloatArray

cramervonmises = stats.cramervonmises
kstest = stats.kstest
norm = stats.norm
student_t = stats.t

_CLIP = 1e-12


def _validate_u(u: NDArray[np.float64]) -> FloatArray:
    array = np.asarray(u, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("u must be a (n, d) array with d >= 2")
    if not np.isfinite(array).all():
        raise ValueError("u must contain only finite values")
    if np.any((array <= 0.0) | (array >= 1.0)):
        raise ValueError("u must have entries strictly inside (0, 1)")
    return np.asarray(array, dtype=np.float64)


def cond_cdf_gaussian(u: NDArray[np.float64], rho: float) -> FloatArray:
    """Return conditional CDFs for an equicorrelated Gaussian copula."""

    data = _validate_u(u)
    if not (-0.999999 < rho < 0.999999):
        raise ValueError("rho must lie inside (-1, 1)")

    _, d = data.shape
    if d >= 2 and rho <= -1.0 / (d - 1):
        raise ValueError("rho too negative for equicorrelation")

    corr = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))
    z = norm.ppf(np.clip(data, _CLIP, 1.0 - _CLIP))

    result = np.empty_like(data)
    result[:, 0] = np.clip(data[:, 0], _CLIP, 1.0 - _CLIP)

    for j in range(1, d):
        sigma11 = corr[:j, :j]
        sigma21 = corr[j, :j]
        weights = np.linalg.solve(sigma11, sigma21)
        cond_var = 1.0 - float(sigma21 @ weights)
        if cond_var <= 0.0:
            raise ValueError("conditional variance must be positive")
        cond_mean = z[:, :j] @ weights
        standardized = (z[:, j] - cond_mean) / np.sqrt(cond_var)
        result[:, j] = np.clip(norm.cdf(standardized), _CLIP, 1.0 - _CLIP)

    return np.asarray(result, dtype=np.float64)


def cond_cdf_student_t(
    u: NDArray[np.float64], rho: float, nu: float
) -> FloatArray:
    """Return conditional CDFs for an equicorrelated Student t copula."""

    data = _validate_u(u)
    if not (-0.999999 < rho < 0.999999):
        raise ValueError("rho must lie inside (-1, 1)")
    if nu <= 2.0:
        raise ValueError("nu must be greater than 2")

    _, d = data.shape
    if d >= 2 and rho <= -1.0 / (d - 1):
        raise ValueError("rho too negative for equicorrelation")

    corr = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))
    z = student_t.ppf(np.clip(data, _CLIP, 1.0 - _CLIP), df=nu)

    result = np.empty_like(data)
    result[:, 0] = np.clip(data[:, 0], _CLIP, 1.0 - _CLIP)

    for j in range(1, d):
        sigma11 = corr[:j, :j]
        sigma21 = corr[j, :j]
        sigma11_inv = np.linalg.inv(sigma11)
        weights = sigma11_inv @ sigma21
        base_scale = 1.0 - float(sigma21 @ weights)
        if base_scale <= 0.0:
            raise ValueError("conditional scale must be positive")

        cond_mean = z[:, :j] @ weights
        quad = np.einsum("ni,ij,nj->n", z[:, :j], sigma11_inv, z[:, :j])
        cond_df = nu + float(j)
        cond_scale = np.sqrt((nu + quad) / cond_df * base_scale)
        standardized = student_t.cdf(
            z[:, j], df=cond_df, loc=cond_mean, scale=cond_scale
        )
        result[:, j] = np.clip(standardized, _CLIP, 1.0 - _CLIP)

    return np.asarray(result, dtype=np.float64)


def cond_cdf_clayton(u: NDArray[np.float64], theta: float) -> FloatArray:
    data = _validate_u(u)
    copula = ClaytonCopula(theta=float(theta), dim=data.shape[1])
    return copula.cond_cdf(data)


def cond_cdf_gumbel(u: NDArray[np.float64], theta: float) -> FloatArray:
    data = _validate_u(u)
    copula = GumbelCopula(theta=float(theta), dim=data.shape[1])
    return copula.cond_cdf(data)


def cond_cdf_frank(u: NDArray[np.float64], theta: float) -> FloatArray:
    data = _validate_u(u)
    copula = FrankCopula(theta=float(theta), dim=data.shape[1])
    return copula.cond_cdf(data)


def cond_cdf_joe(u: NDArray[np.float64], theta: float) -> FloatArray:
    data = _validate_u(u)
    copula = JoeCopula(theta=float(theta), dim=data.shape[1])
    return copula.cond_cdf(data)


def cond_cdf_amh(u: NDArray[np.float64], theta: float) -> FloatArray:
    data = _validate_u(u)
    if data.shape[1] != 2:
        raise ValueError("AMH conditional transform currently supports d=2")
    copula = AMHCopula(theta=float(theta))
    return copula.cond_cdf(data)


def rosenblatt(
    u: NDArray[np.float64],
    cond_cdf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> FloatArray:
    """Compute the Rosenblatt transform using a conditional CDF oracle."""

    data = _validate_u(u)
    cond_values = np.asarray(cond_cdf(data), dtype=float)
    if cond_values.shape != data.shape:
        raise ValueError("conditional CDF output must match input shape")
    if not np.isfinite(cond_values).all():
        raise ValueError("conditional CDF output must be finite")

    transformed = np.empty_like(data)
    transformed[:, 0] = np.clip(data[:, 0], _CLIP, 1.0 - _CLIP)
    transformed[:, 1:] = np.clip(cond_values[:, 1:], _CLIP, 1.0 - _CLIP)
    return np.asarray(transformed, dtype=np.float64)


def gof_ks_uniform(z: NDArray[np.float64]) -> float:
    """Return the minimum Kolmogorov–Smirnov p-value across dimensions."""

    data = np.asarray(z, dtype=float)
    if data.ndim != 2:
        raise ValueError("z must be a 2D array")
    if data.shape[1] < 1:
        raise ValueError("z must have at least one dimension")

    pvals = [
        kstest(data[:, j], "uniform").pvalue for j in range(data.shape[1])
    ]
    return float(np.min(pvals))


def gof_cvm_uniform(z: NDArray[np.float64]) -> float:
    """Return the minimum Cramér–von Mises p-value across dimensions."""

    data = np.asarray(z, dtype=float)
    if data.ndim != 2:
        raise ValueError("z must be a 2D array")
    if data.shape[1] < 1:
        raise ValueError("z must have at least one dimension")

    pvals = [
        cramervonmises(data[:, j], "uniform").pvalue
        for j in range(data.shape[1])
    ]
    return float(np.min(pvals))


def rosenblatt_gaussian(
    u: NDArray[np.float64], rho: float
) -> tuple[FloatArray, float, float]:
    """Compute Gaussian Rosenblatt transform and uniformity p-values."""

    transformed = rosenblatt(u, lambda w: cond_cdf_gaussian(w, rho))
    return (
        transformed,
        gof_ks_uniform(transformed),
        gof_cvm_uniform(transformed),
    )


def rosenblatt_student_t(
    u: NDArray[np.float64], rho: float, nu: float
) -> tuple[FloatArray, float, float]:
    """Compute Student t Rosenblatt transform and uniformity p-values."""

    transformed = rosenblatt(
        u, lambda w: cond_cdf_student_t(w, rho=rho, nu=nu)
    )
    return (
        transformed,
        gof_ks_uniform(transformed),
        gof_cvm_uniform(transformed),
    )


def rosenblatt_clayton(
    u: NDArray[np.float64], theta: float
) -> tuple[FloatArray, float, float]:
    transformed = rosenblatt(u, lambda w: cond_cdf_clayton(w, theta=theta))
    return (
        transformed,
        gof_ks_uniform(transformed),
        gof_cvm_uniform(transformed),
    )


def rosenblatt_gumbel(
    u: NDArray[np.float64], theta: float
) -> tuple[FloatArray, float, float]:
    transformed = rosenblatt(u, lambda w: cond_cdf_gumbel(w, theta=theta))
    return (
        transformed,
        gof_ks_uniform(transformed),
        gof_cvm_uniform(transformed),
    )


def rosenblatt_frank(
    u: NDArray[np.float64], theta: float
) -> tuple[FloatArray, float, float]:
    transformed = rosenblatt(u, lambda w: cond_cdf_frank(w, theta=theta))
    return (
        transformed,
        gof_ks_uniform(transformed),
        gof_cvm_uniform(transformed),
    )


def rosenblatt_joe(
    u: NDArray[np.float64], theta: float
) -> tuple[FloatArray, float, float]:
    transformed = rosenblatt(u, lambda w: cond_cdf_joe(w, theta=theta))
    return (
        transformed,
        gof_ks_uniform(transformed),
        gof_cvm_uniform(transformed),
    )


def rosenblatt_amh(
    u: NDArray[np.float64], theta: float
) -> tuple[FloatArray, float, float]:
    transformed = rosenblatt(u, lambda w: cond_cdf_amh(w, theta=theta))
    return (
        transformed,
        gof_ks_uniform(transformed),
        gof_cvm_uniform(transformed),
    )
