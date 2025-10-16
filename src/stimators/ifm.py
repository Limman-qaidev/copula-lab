from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore[import-untyped]


def gaussian_ifm(u: NDArray[np.float64]) -> float:
    """
    IFM for Gaussian copula on pseudo-observations u in (0,1)^2.
    Returns rho_hat.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be (n,2)")
    z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    return float(np.corrcoef(z.T)[0, 1])
