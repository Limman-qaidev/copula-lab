from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t
from scipy.stats import multivariate_t

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StudentTCopula:
    """Bivariate Student-t copula with rho in (-1,1) and nu>2."""

    rho: float
    nu: float

    def __post_init__(self) -> None:
        if not (-0.999999 < self.rho < 0.999999):
            raise ValueError("rho must be in (-1, 1)")
        if not (self.nu > 2.0):
            raise ValueError("nu must be > 2")

    @property
    def scale(self) -> NDArray[np.float64]:
        return np.array([[1.0, self.rho], [self.rho, 1.0]], dtype=float)

    def pdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("u must be (n,2)")
        x = student_t.ppf(np.clip(u, 1e-12, 1 - 1e-12), self.nu)
        num = multivariate_t(loc=np.zeros(2), shape=self.scale, df=self.nu).pdf(x)
        den = student_t.pdf(x[:, 0], self.nu) * student_t.pdf(x[:, 1], self.nu)
        return (num / den).astype(float)

    def cdf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("u must be (n,2)")
        x = student_t.ppf(np.clip(u, 1e-12, 1 - 1e-12), self.nu)
        mv_t = multivariate_t(loc=np.zeros(2), shape=self.scale, df=self.nu)
        return mv_t.cdf(x).astype(float)

    def rvs(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        x = multivariate_t(loc=np.zeros(2), shape=self.scale, df=self.nu).rvs(
            size=n, random_state=rng
        )
        return student_t.cdf(x, self.nu)
