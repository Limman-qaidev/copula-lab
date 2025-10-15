from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal, norm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GaussianCopula:
    """Bivariate Gaussian copula with correlation rho in (-1, 1)."""

    rho: float

    def __post_init__(self) -> None:
        if not (-0.999999 < self.rho < 0.999999):
            raise ValueError("rho must be in (-1, 1)")

    @property
    def corr(self) -> NDArray[np.float64]:
        return np.array([[1.0, self.rho], [self.rho, 1.0]], dtype=float)

    def pdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("u must be (n,2)")
        z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        mvn = multivariate_normal(mean=[0.0, 0.0], cov=self.corr)
        phi_r = np.asarray(mvn.pdf(z), dtype=np.float64)
        phi = np.asarray(norm.pdf(z), dtype=np.float64)
        denom = phi[:, 0] * phi[:, 1]
        return np.asarray(phi_r / denom, dtype=np.float64)

    def cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("u must be (n,2)")
        z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        mvn = multivariate_normal(mean=[0.0, 0.0], cov=self.corr)
        return np.asarray(mvn.cdf(z), dtype=np.float64)

    def rvs(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        z = rng.multivariate_normal(mean=[0.0, 0.0], cov=self.corr, size=n)
        return np.asarray(norm.cdf(z), dtype=np.float64)
