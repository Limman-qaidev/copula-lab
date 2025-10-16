"""Archimedean copula families (Clayton, Gumbel, Frank, Joe, AMH)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.stats import levy_stable  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_CLIP = 1e-12


def _validate_samples(u: NDArray[np.float64], dim: int) -> NDArray[np.float64]:
    array = np.asarray(u, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(
            f"u must be a (n, {dim}) array of pseudo-observations"
        )
    if not np.isfinite(array).all():
        raise ValueError("u must contain only finite values")
    if np.any((array <= 0.0) | (array >= 1.0)):
        raise ValueError("u must have entries strictly inside (0, 1)")
    return array


def _complete_bell_polynomial(derivatives: list[float]) -> float:
    """Return the complete exponential Bell polynomial of the derivatives."""

    order = len(derivatives)
    bell: list[float] = [0.0] * (order + 1)
    bell[0] = 1.0
    for m in range(1, order + 1):
        total = 0.0
        for k in range(1, m + 1):
            factor = math.comb(m - 1, k - 1)
            total += float(factor) * derivatives[k - 1] * bell[m - k]
        bell[m] = total
    return bell[order]


def _falling_factorial(x: float, k: int) -> float:
    result = 1.0
    for i in range(k):
        result *= x - float(i)
    return result


def _log_derivative_sequence(values: list[float], order: int) -> list[float]:
    """Return derivatives of ``log`` of a smooth function up to ``order``."""

    log_derivs = [0.0] * (order + 1)
    for n in range(1, order + 1):
        term = values[n]
        for k in range(1, n):
            coeff = math.comb(n - 1, k - 1)
            term -= coeff * values[k] * log_derivs[n - k]
        log_derivs[n] = term / values[0]
    return [log_derivs[n] for n in range(1, order + 1)]


@lru_cache(maxsize=None)
def _eulerian_numbers(n: int) -> tuple[float, ...]:
    if n < 0:
        raise ValueError("n must be non-negative")
    table = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]
    table[0][0] = 1.0
    for m in range(1, n + 1):
        for k in range(m + 1):
            term1 = (k + 1) * table[m - 1][k] if k <= m - 1 else 0.0
            term2 = (m - k) * table[m - 1][k - 1] if k >= 1 else 0.0
            table[m][k] = term1 + term2
    return tuple(table[n][: n + 1])


def _polylog_negative_int(
    order: int, x: NDArray[np.float64]
) -> NDArray[np.float64]:
    if order < 0:
        raise ValueError("order must be non-negative")
    values = np.asarray(x, dtype=np.float64)
    values = np.minimum(values, 1.0 - 1e-12)
    if order == 0:
        return np.asarray(values / (1.0 - values), dtype=np.float64)

    coeffs = _eulerian_numbers(order)
    numerator = np.zeros_like(values, dtype=np.float64)
    for power, coeff in enumerate(coeffs):
        numerator += coeff * np.power(values, power)
    denominator = np.power(1.0 - values, order + 1)
    return np.asarray(values * numerator / denominator, dtype=np.float64)


def _psi_derivative_clayton(
    theta: float, order: int, t: NDArray[np.float64]
) -> NDArray[np.float64]:
    coeff = 1.0
    if order > 0:
        coeff = float(
            np.prod(
                [1.0 + theta * float(k) for k in range(order)],
                dtype=np.float64,
            )
        )
    exponent = -(1.0 / theta + float(order))
    values = np.asarray(t, dtype=np.float64)
    base = np.power(1.0 + values, exponent)
    return np.asarray(coeff * base, dtype=np.float64)


def _psi_derivative_gumbel(
    theta: float, order: int, t: NDArray[np.float64]
) -> NDArray[np.float64]:
    values = np.asarray(t, dtype=np.float64)
    if order == 0:
        base = np.exp(-np.power(values, 1.0 / theta))
        return np.asarray(base, dtype=np.float64)

    alpha = 1.0 / theta
    flat = values.reshape(-1)
    result = np.empty_like(flat, dtype=np.float64)
    for idx, val in enumerate(flat):
        derivatives = [
            -_falling_factorial(alpha, k) * float(val) ** (alpha - float(k))
            for k in range(1, order + 1)
        ]
        bell = _complete_bell_polynomial(derivatives)
        result[idx] = math.exp(-float(val) ** alpha) * bell
    return result.reshape(values.shape)


def _psi_derivative_frank(
    theta: float, order: int, t: NDArray[np.float64]
) -> NDArray[np.float64]:
    values = np.asarray(t, dtype=np.float64)
    if order == 0:
        a = 1.0 - math.exp(-theta)
        base = -np.log1p(-a * np.exp(-values)) / theta
        return np.asarray(base, dtype=np.float64)

    scalar = 1.0 - math.exp(-theta)
    argument = scalar * np.exp(-values)
    poly = _polylog_negative_int(order - 1, argument)
    result = ((-1.0) ** order / theta) * poly
    return np.asarray(result, dtype=np.float64)


def _psi_derivative_joe(
    theta: float, order: int, t: NDArray[np.float64]
) -> NDArray[np.float64]:
    values = np.asarray(t, dtype=np.float64)
    if order == 0:
        return np.asarray(
            1.0 - np.power(1.0 - np.exp(-values), 1.0 / theta),
            dtype=np.float64,
        )

    alpha = 1.0 / theta
    flat = values.reshape(-1)
    result = np.empty_like(flat, dtype=np.float64)
    for idx, val in enumerate(flat):
        total = 0.0
        binom = alpha
        for k in range(1, 200):
            if k > 1:
                binom *= (alpha - (k - 1)) / float(k)
            magnitude = float(k) ** order
            decay = math.exp(-float(k) * float(val))
            term = (-1.0) ** (k + 1) * binom * magnitude * decay
            total += term
            if abs(term) < 1e-12:
                break
        result[idx] = ((-1.0) ** order) * total
    return result.reshape(values.shape)


def _psi_derivative_amh(
    theta: float, order: int, t: NDArray[np.float64]
) -> NDArray[np.float64]:
    values = np.asarray(t, dtype=np.float64)
    if order == 0:
        return np.asarray(
            (1.0 - theta) / (np.exp(values) - theta), dtype=np.float64
        )

    flat = values.reshape(-1)
    result = np.empty_like(flat, dtype=np.float64)
    for idx, val in enumerate(flat):
        exp_val = math.exp(float(val))
        g0 = max(exp_val - theta, 1e-12)
        derivatives = [g0]
        for _ in range(1, order + 1):
            derivatives.append(exp_val)
        log_derivs = _log_derivative_sequence(derivatives, order)
        h_derivs = [-deriv for deriv in log_derivs]
        bell = _complete_bell_polynomial(h_derivs)
        result[idx] = (1.0 - theta) * (1.0 / g0) * bell
    return result.reshape(values.shape)


def _archimedean_conditional(
    u: NDArray[np.float64],
    theta: float,
    dim: int,
    psi_inv: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    psi_derivative: Callable[
        [float, int, NDArray[np.float64]], NDArray[np.float64]
    ],
) -> NDArray[np.float64]:
    data = _validate_samples(u, dim)
    clipped = np.clip(data, _CLIP, 1.0 - _CLIP)
    inv_values = np.asarray(psi_inv(clipped), dtype=np.float64)

    transformed = np.empty_like(clipped, dtype=np.float64)
    transformed[:, 0] = clipped[:, 0]

    for j in range(1, dim):
        prefix = np.sum(inv_values[:, :j], axis=1)
        total = prefix + inv_values[:, j]
        numerator = np.asarray(
            psi_derivative(theta, j, total), dtype=np.float64
        )
        denominator = np.asarray(
            psi_derivative(theta, j, prefix), dtype=np.float64
        )
        denominator = np.where(
            denominator == 0.0, np.finfo(np.float64).tiny, denominator
        )
        ratio = numerator / denominator
        transformed[:, j] = np.clip(ratio, _CLIP, 1.0 - _CLIP)

    return np.asarray(transformed, dtype=np.float64)


@dataclass(frozen=True)
class ClaytonCopula:
    """Clayton copula with parameter ``theta > 0``."""

    theta: float
    dim: int = 2

    def __post_init__(self) -> None:
        if not (self.dim >= 2):
            raise ValueError("dim must be at least 2")
        if self.theta <= 0.0:
            raise ValueError("theta must be strictly positive")

    def _psi_inv(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(u, dtype=np.float64)
        return np.asarray(
            np.power(values, -self.theta) - 1.0, dtype=np.float64
        )

    def _psi_inv_prime(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(u, dtype=np.float64)
        return np.asarray(
            -self.theta * np.power(values, -self.theta - 1.0), dtype=np.float64
        )

    def cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        data = _validate_samples(u, self.dim)
        inv = self._psi_inv(np.clip(data, _CLIP, 1.0 - _CLIP))
        inv_sum = np.sum(inv, axis=1)
        return np.asarray(
            np.power(inv_sum + 1.0, -1.0 / self.theta), dtype=np.float64
        )

    def pdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        data = _validate_samples(u, self.dim)
        clipped = np.clip(data, _CLIP, 1.0 - _CLIP)
        sum_term = np.sum(np.power(clipped, -self.theta), axis=1) - (
            self.dim - 1.0
        )
        base = np.power(sum_term, -(self.dim + 1.0 / self.theta))
        prod_term = np.prod(np.power(clipped, -self.theta - 1.0), axis=1)
        coeff = float(
            np.prod(
                [1.0 + self.theta * float(k) for k in range(self.dim)],
                dtype=np.float64,
            )
        )
        return np.asarray(coeff * prod_term * base, dtype=np.float64)

    def rvs(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        gamma_shape = 1.0 / self.theta
        w = rng.gamma(shape=gamma_shape, scale=1.0, size=n)
        e = rng.exponential(scale=1.0, size=(n, self.dim))
        base = np.power(1.0 + e / w[:, None], -1.0 / self.theta)
        return np.asarray(np.clip(base, _CLIP, 1.0 - _CLIP), dtype=np.float64)

    def cond_cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        return _archimedean_conditional(
            u,
            self.theta,
            self.dim,
            self._psi_inv,
            _psi_derivative_clayton,
        )


@dataclass(frozen=True)
class GumbelCopula:
    """Gumbel copula with ``theta >= 1``."""

    theta: float
    dim: int = 2

    def __post_init__(self) -> None:
        if not (self.dim >= 2):
            raise ValueError("dim must be at least 2")
        if self.theta < 1.0:
            raise ValueError("theta must be at least 1")

    def _psi_inv(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(u, _CLIP, 1.0 - _CLIP)
        return np.asarray(
            np.power(-np.log(clipped), self.theta), dtype=np.float64
        )

    def _psi_inv_prime(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(u, _CLIP, 1.0 - _CLIP)
        numerator = -self.theta * np.power(-np.log(clipped), self.theta - 1.0)
        return np.asarray(numerator / clipped, dtype=np.float64)

    def cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        data = _validate_samples(u, self.dim)
        inv_sum = np.sum(self._psi_inv(data), axis=1)
        return np.asarray(
            np.exp(-np.power(inv_sum, 1.0 / self.theta)), dtype=np.float64
        )

    def pdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        data = _validate_samples(u, self.dim)
        clipped = np.clip(data, _CLIP, 1.0 - _CLIP)
        inv_values = self._psi_inv(clipped)
        total = np.sum(inv_values, axis=1)
        derivative = _psi_derivative_gumbel(self.theta, self.dim, total)
        prod_term = np.prod(-self._psi_inv_prime(clipped), axis=1)
        sign = -1.0 if self.dim % 2 == 1 else 1.0
        return np.asarray(sign * derivative * prod_term, dtype=np.float64)

    def rvs(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        alpha = 1.0 / self.theta
        stable = np.asarray(
            levy_stable.rvs(alpha, 1.0, size=n, random_state=rng),
            dtype=np.float64,
        )
        stable = np.maximum(stable, _CLIP)
        e = rng.exponential(scale=1.0, size=(n, self.dim))
        base = np.exp(-e / np.power(stable[:, None], alpha))
        return np.asarray(np.clip(base, _CLIP, 1.0 - _CLIP), dtype=np.float64)

    def cond_cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        return _archimedean_conditional(
            u,
            self.theta,
            self.dim,
            self._psi_inv,
            _psi_derivative_gumbel,
        )


@dataclass(frozen=True)
class FrankCopula:
    """Frank copula with parameter ``theta != 0``."""

    theta: float
    dim: int = 2

    def __post_init__(self) -> None:
        if not (self.dim >= 2):
            raise ValueError("dim must be at least 2")
        if abs(self.theta) < 1e-9:
            raise ValueError("theta must be non-zero")

    def _psi_inv(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(u, _CLIP, 1.0 - _CLIP)
        num = np.expm1(-self.theta * clipped)
        den = np.expm1(-self.theta)
        ratio = num / den
        return np.asarray(-np.log(ratio), dtype=np.float64)

    def _psi_inv_prime(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(u, _CLIP, 1.0 - _CLIP)
        numerator = self.theta * np.exp(-self.theta * clipped)
        denominator = np.expm1(-self.theta * clipped)
        return np.asarray(numerator / denominator, dtype=np.float64)

    def cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        data = _validate_samples(u, self.dim)
        inv_sum = np.sum(self._psi_inv(data), axis=1)
        a = 1.0 - math.exp(-self.theta)
        return np.asarray(
            (-1.0 / self.theta) * np.log1p(-a * np.exp(-inv_sum)),
            dtype=np.float64,
        )

    def pdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        data = _validate_samples(u, self.dim)
        clipped = np.clip(data, _CLIP, 1.0 - _CLIP)
        inv_values = self._psi_inv(clipped)
        total = np.sum(inv_values, axis=1)
        derivative = _psi_derivative_frank(self.theta, self.dim, total)
        prod_term = np.prod(-self._psi_inv_prime(clipped), axis=1)
        sign = -1.0 if self.dim % 2 == 1 else 1.0
        return np.asarray(sign * derivative * prod_term, dtype=np.float64)

    def rvs(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        w = rng.exponential(scale=1.0, size=n)
        v = rng.uniform(size=(n, self.dim))
        inner = np.exp(-self.theta * v) * (
            np.exp(-self.theta * w)[:, None] - 1.0
        )
        samples = -np.log1p(inner) / self.theta
        return np.asarray(
            np.clip(samples, _CLIP, 1.0 - _CLIP), dtype=np.float64
        )

    def cond_cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        return _archimedean_conditional(
            u,
            self.theta,
            self.dim,
            self._psi_inv,
            _psi_derivative_frank,
        )


@dataclass(frozen=True)
class JoeCopula:
    """Joe copula with parameter ``theta >= 1``."""

    theta: float
    dim: int = 2

    def __post_init__(self) -> None:
        if self.dim < 2:
            raise ValueError("dim must be at least 2")
        if self.theta < 1.0:
            raise ValueError("theta must be at least 1")

    def _psi_inv(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(u, _CLIP, 1.0 - _CLIP)
        inner = np.power(1.0 - clipped, self.theta)
        return np.asarray(-np.log(1.0 - inner), dtype=np.float64)

    def _psi_inv_prime(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(u, _CLIP, 1.0 - _CLIP)
        numerator = -self.theta * np.power(1.0 - clipped, self.theta - 1.0)
        denominator = 1.0 - np.power(1.0 - clipped, self.theta)
        denominator = np.maximum(denominator, np.finfo(np.float64).tiny)
        return np.asarray(numerator / denominator, dtype=np.float64)

    def cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        data = _validate_samples(u, self.dim)
        total = np.sum(self._psi_inv(data), axis=1)
        inner = 1.0 - np.exp(-total)
        return np.asarray(
            1.0 - np.power(inner, 1.0 / self.theta), dtype=np.float64
        )

    def pdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        data = _validate_samples(u, self.dim)
        clipped = np.clip(data, _CLIP, 1.0 - _CLIP)
        total = np.sum(self._psi_inv(clipped), axis=1)
        derivative = _psi_derivative_joe(self.theta, self.dim, total)
        prod_term = np.prod(-self._psi_inv_prime(clipped), axis=1)
        sign = -1.0 if self.dim % 2 == 1 else 1.0
        return np.asarray(sign * derivative * prod_term, dtype=np.float64)

    def rvs(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        samples = np.empty((n, self.dim), dtype=np.float64)
        for i in range(n):
            row = np.empty(self.dim, dtype=np.float64)
            row[0] = rng.uniform(_CLIP, 1.0 - _CLIP)
            for j in range(1, self.dim):
                target = rng.uniform(_CLIP, 1.0 - _CLIP)

                def objective(x: float) -> float:
                    trial = np.full(self.dim, x, dtype=np.float64)
                    trial[:j] = row[:j]
                    cond = _archimedean_conditional(
                        trial[None, :],
                        self.theta,
                        self.dim,
                        self._psi_inv,
                        _psi_derivative_joe,
                    )
                    return float(cond[0, j] - target)

                row[j] = brentq(objective, _CLIP, 1.0 - _CLIP, maxiter=256)
            samples[i] = row
        return samples

    def cond_cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        return _archimedean_conditional(
            u,
            self.theta,
            self.dim,
            self._psi_inv,
            _psi_derivative_joe,
        )


@dataclass(frozen=True)
class AMHCopula:
    """Ali–Mikhail–Haq copula with ``theta in [0, 1)``."""

    theta: float
    dim: int = 2

    def __post_init__(self) -> None:
        if self.dim != 2:
            raise ValueError("AMH copula currently supports dim=2")
        if not (-1.0 < self.theta < 1.0):
            raise ValueError("theta must lie in (-1, 1)")

    def _psi_inv(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(u, _CLIP, 1.0 - _CLIP)
        numerator = 1.0 - self.theta + self.theta * clipped
        return np.asarray(np.log(numerator / clipped), dtype=np.float64)

    def _psi_inv_prime(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(u, _CLIP, 1.0 - _CLIP)
        numerator = self.theta - 1.0
        denominator = clipped * (1.0 - self.theta + self.theta * clipped)
        denominator = np.maximum(denominator, np.finfo(np.float64).tiny)
        return np.asarray(numerator / denominator, dtype=np.float64)

    def cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        data = _validate_samples(u, self.dim)
        total = np.sum(self._psi_inv(data), axis=1)
        return np.asarray(
            (1.0 - self.theta) / (np.exp(total) - self.theta), dtype=np.float64
        )

    def pdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        data = _validate_samples(u, self.dim)
        clipped = np.clip(data, _CLIP, 1.0 - _CLIP)
        total = np.sum(self._psi_inv(clipped), axis=1)
        derivative = _psi_derivative_amh(self.theta, self.dim, total)
        prod_term = np.prod(-self._psi_inv_prime(clipped), axis=1)
        sign = -1.0 if self.dim % 2 == 1 else 1.0
        return np.asarray(sign * derivative * prod_term, dtype=np.float64)

    def rvs(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        samples = np.empty((n, self.dim), dtype=np.float64)
        for i in range(n):
            row = np.empty(self.dim, dtype=np.float64)
            row[0] = rng.uniform(_CLIP, 1.0 - _CLIP)
            target = rng.uniform(_CLIP, 1.0 - _CLIP)

            def objective(x: float) -> float:
                trial = np.array([row[0], x], dtype=np.float64)
                cond = _archimedean_conditional(
                    trial[None, :],
                    self.theta,
                    self.dim,
                    self._psi_inv,
                    _psi_derivative_amh,
                )
                return float(cond[0, 1] - target)

            row[1] = brentq(objective, _CLIP, 1.0 - _CLIP, maxiter=256)
            samples[i] = row
        return samples

    def cond_cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        return _archimedean_conditional(
            u,
            self.theta,
            self.dim,
            self._psi_inv,
            _psi_derivative_amh,
        )


__all__ = [
    "ClaytonCopula",
    "GumbelCopula",
    "FrankCopula",
    "JoeCopula",
    "AMHCopula",
]
