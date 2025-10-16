from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.copulas.archimedean import (  # noqa: E402
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
)


def test_clayton_pdf_positive() -> None:
    copula = ClaytonCopula(theta=1.5, dim=3)
    u = copula.rvs(500, seed=42)
    density = copula.pdf(u)
    assert density.shape == (500,)
    assert np.all(density > 0.0)
    cdf = copula.cdf(u)
    assert np.all((0.0 < cdf) & (cdf < 1.0))


def test_gumbel_pdf_positive() -> None:
    copula = GumbelCopula(theta=1.3, dim=3)
    u = copula.rvs(200, seed=123)
    density = copula.pdf(u)
    assert density.shape == (200,)
    assert np.all(np.isfinite(density))
    assert np.all(density > 0.0)


def test_frank_pdf_positive() -> None:
    copula = FrankCopula(theta=4.0, dim=3)
    u = copula.rvs(200, seed=987)
    density = copula.pdf(u)
    assert density.shape == (200,)
    assert np.all(np.isfinite(density))
    assert np.all(density > 0.0)
