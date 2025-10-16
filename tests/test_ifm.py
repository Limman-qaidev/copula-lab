from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm  # type: ignore[import-untyped]

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.estimators.ifm import gaussian_ifm  # noqa: E402


def test_gaussian_ifm_matches_sample_correlation() -> None:
    rng = np.random.default_rng(1234)
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    z = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=5000)
    u = norm.cdf(z)

    corr_z = float(np.corrcoef(z.T)[0, 1])
    rho_hat = gaussian_ifm(u)

    assert rho_hat == pytest.approx(corr_z, abs=5e-3)


def test_gaussian_ifm_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError):
        gaussian_ifm(np.ones((10,), dtype=np.float64))


def test_gaussian_ifm_rejects_invalid_values() -> None:
    bad = np.array([[0.5, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        gaussian_ifm(bad)
