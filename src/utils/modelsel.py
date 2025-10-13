from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, multivariate_normal


def gaussian_pseudo_loglik(u: NDArray[np.float64], rho: float) -> float:
    """
    Pseudo log-likelihood for Gaussian copula on pseudo-observations u.
    ℓ = sum [ log φ_R(z) - sum_i log φ(z_i) ], with z = Φ^{-1}(u)
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be (n,2)")
    z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    corr = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    ll_joint = multivariate_normal(mean=[0.0, 0.0], cov=corr).logpdf(z)
    ll_marg = np.sum(np.log(norm.pdf(z)), axis=1)
    return float(np.sum(ll_joint - ll_marg))


def information_criteria(
        loglik: float, k_params: int, n: int
) -> tuple[float, float]:
    """
    AIC = 2k - 2ℓ ;  BIC = k log n - 2ℓ
    """
    aic = 2.0 * k_params - 2.0 * float(loglik)
    bic = float(k_params) * np.log(float(n)) - 2.0 * float(loglik)
    return float(aic), float(bic)
