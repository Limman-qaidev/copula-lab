"""Student t copula utilities supporting arbitrary dimension."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_t  # type: ignore[import-untyped]
from scipy.stats import t as student_t

from src.utils.types import FloatArray

logger = logging.getLogger(__name__)

_CLIP = 1e-12


def _validate_corr(matrix: FloatArray) -> NDArray[np.float64]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    if array.shape[0] < 2:
        raise ValueError("Correlation matrix must be at least 2x2.")
    if not np.allclose(array, array.T, atol=1e-9):
        raise ValueError("Correlation matrix must be symmetric.")
    diag = np.diag(array)
    if not np.allclose(diag, 1.0, atol=1e-9):
        raise ValueError("Correlation matrix must have unit diagonal.")
    try:
        np.linalg.cholesky(array)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Correlation matrix must be positive definite."
        ) from exc
    return array


@dataclass(frozen=True)
class StudentTCopula:
    """Student t copula with a correlation matrix and degrees of freedom."""

    corr: NDArray[np.float64]
    nu: float

    def __init__(
        self,
        *,
        corr: FloatArray | None = None,
        rho: float | None = None,
        nu: float,
    ) -> None:
        if corr is None and rho is None:
            raise ValueError("Provide corr or rho to parameterize the copula.")
        if corr is not None and rho is not None:
            raise ValueError("Specify only one of corr or rho.")
        if nu <= 2.0:
            raise ValueError("nu must be greater than 2")

        if corr is not None:
            matrix = _validate_corr(corr)
        else:
            rho_val = cast(float, rho)
            if not (-0.999999 < rho_val < 0.999999):
                raise ValueError("rho must lie strictly between -1 and 1")
            matrix = np.array(
                [[1.0, rho_val], [rho_val, 1.0]], dtype=np.float64
            )

        object.__setattr__(self, "corr", matrix)
        object.__setattr__(self, "nu", float(nu))

    @property
    def dim(self) -> int:
        return int(self.corr.shape[0])

    def _validate_u(self, u: FloatArray) -> NDArray[np.float64]:
        array = np.asarray(u, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.dim:
            raise ValueError(
                f"u must be an (n, {self.dim}) array of pseudo-observations"
            )
        if np.any((array <= 0.0) | (array >= 1.0)):
            raise ValueError("u must lie strictly inside (0, 1)")
        return array

    def pdf(self, u: FloatArray) -> NDArray[np.float64]:
        data = self._validate_u(u)
        x = student_t.ppf(np.clip(data, _CLIP, 1.0 - _CLIP), self.nu)
        mv_t = cast(
            Any,
            multivariate_t,
        )(
            loc=np.zeros(self.dim),
            shape=self.corr,
            df=float(self.nu),
        )
        num = np.asarray(mv_t.pdf(x), dtype=np.float64)
        den = np.prod(student_t.pdf(x, self.nu), axis=1)
        density = num / np.asarray(den, dtype=np.float64)
        return np.asarray(density, dtype=np.float64)

    def cdf(self, u: FloatArray) -> NDArray[np.float64]:
        data = self._validate_u(u)
        x = student_t.ppf(np.clip(data, _CLIP, 1.0 - _CLIP), self.nu)
        mv_t = cast(
            Any,
            multivariate_t,
        )(
            loc=np.zeros(self.dim),
            shape=self.corr,
            df=float(self.nu),
        )
        return np.asarray(mv_t.cdf(x), dtype=np.float64)

    def rvs(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        mv_t = cast(
            Any,
            multivariate_t,
        )(
            loc=np.zeros(self.dim),
            shape=self.corr,
            df=float(self.nu),
        )
        samples = np.asarray(
            mv_t.rvs(size=n, random_state=rng), dtype=np.float64
        )
        return np.asarray(student_t.cdf(samples, self.nu), dtype=np.float64)


__all__ = ["StudentTCopula"]
