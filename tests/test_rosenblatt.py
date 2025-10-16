from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.copulas.archimedean import (  # noqa: E402
    AMHCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from src.models.copulas.student_t import StudentTCopula  # noqa: E402
from src.utils.rosenblatt import (  # noqa: E402
    cond_cdf_amh,
    cond_cdf_clayton,
    cond_cdf_frank,
    cond_cdf_gumbel,
    cond_cdf_joe,
    cond_cdf_student_t,
    gof_cvm_uniform,
    gof_ks_uniform,
    rosenblatt_amh,
    rosenblatt_clayton,
    rosenblatt_frank,
    rosenblatt_gaussian,
    rosenblatt_gumbel,
    rosenblatt_joe,
    rosenblatt_student_t,
)


def test_cond_cdf_student_t_matches_dimensions() -> None:
    copula = StudentTCopula(rho=0.4, nu=5.0)
    U = copula.rvs(200, seed=123)
    result = cond_cdf_student_t(U, rho=0.4, nu=5.0)
    assert result.shape == U.shape
    assert np.all((0.0 < result) & (result < 1.0))


def test_cond_cdf_student_t_rejects_low_nu() -> None:
    U = np.full((10, 2), 0.5, dtype=np.float64)
    with pytest.raises(ValueError):
        cond_cdf_student_t(U, rho=0.1, nu=1.5)


def test_rosenblatt_student_t_returns_uniform_components() -> None:
    copula = StudentTCopula(rho=0.3, nu=7.0)
    U = copula.rvs(1000, seed=456)
    Z, ks_pvalue, cvm_pvalue = rosenblatt_student_t(U, rho=0.3, nu=7.0)
    assert Z.shape == U.shape
    assert np.all((0.0 < Z) & (Z < 1.0))
    assert 0.0 <= ks_pvalue <= 1.0
    assert 0.0 <= cvm_pvalue <= 1.0


def test_gof_statistics_align_on_gaussian() -> None:
    U = np.column_stack(
        (np.sort(np.random.rand(500)), np.sort(np.random.rand(500)))
    )
    Z_gauss, ks_gauss, cvm_gauss = rosenblatt_gaussian(U, rho=0.0)
    assert Z_gauss.shape == U.shape
    assert np.all((0.0 < Z_gauss) & (Z_gauss < 1.0))
    assert 0.0 <= ks_gauss <= 1.0
    assert 0.0 <= cvm_gauss <= 1.0
    assert gof_ks_uniform(Z_gauss) == pytest.approx(ks_gauss)
    assert gof_cvm_uniform(Z_gauss) == pytest.approx(cvm_gauss)


def test_cond_cdf_clayton_vectorized() -> None:
    copula = ClaytonCopula(theta=1.5, dim=3)
    U = copula.rvs(128, seed=321)
    result = cond_cdf_clayton(U, theta=1.5)
    assert result.shape == U.shape
    assert np.all((0.0 < result) & (result < 1.0))


def test_cond_cdf_gumbel_vectorized() -> None:
    copula = GumbelCopula(theta=1.2, dim=3)
    U = copula.rvs(64, seed=654)
    result = cond_cdf_gumbel(U, theta=1.2)
    assert result.shape == U.shape
    assert np.all((0.0 < result) & (result < 1.0))


def test_cond_cdf_frank_vectorized() -> None:
    copula = FrankCopula(theta=3.0, dim=3)
    U = copula.rvs(64, seed=111)
    result = cond_cdf_frank(U, theta=3.0)
    assert result.shape == U.shape
    assert np.all((0.0 < result) & (result < 1.0))


def test_cond_cdf_joe_vectorized() -> None:
    copula = JoeCopula(theta=1.5, dim=3)
    U = copula.rvs(64, seed=222)
    result = cond_cdf_joe(U, theta=1.5)
    assert result.shape == U.shape
    assert np.all((0.0 < result) & (result < 1.0))


def test_cond_cdf_amh_bivariate() -> None:
    copula = AMHCopula(theta=0.6)
    U = copula.rvs(128, seed=333)
    result = cond_cdf_amh(U, theta=0.6)
    assert result.shape == U.shape
    assert np.all((0.0 < result) & (result < 1.0))


def test_rosenblatt_archimedean_outputs_uniforms() -> None:
    copula = ClaytonCopula(theta=2.0, dim=3)
    U = copula.rvs(200, seed=77)
    Z_clayton, ks_clayton, cvm_clayton = rosenblatt_clayton(U, theta=2.0)
    assert Z_clayton.shape == U.shape
    assert np.all((0.0 < Z_clayton) & (Z_clayton < 1.0))
    assert 0.0 <= ks_clayton <= 1.0
    assert 0.0 <= cvm_clayton <= 1.0

    gumbel = GumbelCopula(theta=1.3, dim=3)
    U_g = gumbel.rvs(200, seed=1234)
    Z_gumbel, _, _ = rosenblatt_gumbel(U_g, theta=1.3)
    assert np.all((0.0 < Z_gumbel) & (Z_gumbel < 1.0))

    frank = FrankCopula(theta=4.0, dim=3)
    U_f = frank.rvs(200, seed=987)
    Z_frank, _, _ = rosenblatt_frank(U_f, theta=4.0)
    assert np.all((0.0 < Z_frank) & (Z_frank < 1.0))

    joe = JoeCopula(theta=1.7, dim=3)
    U_j = joe.rvs(200, seed=4321)
    Z_joe, _, _ = rosenblatt_joe(U_j, theta=1.7)
    assert np.all((0.0 < Z_joe) & (Z_joe < 1.0))

    amh = AMHCopula(theta=0.5)
    U_a = amh.rvs(200, seed=2468)
    Z_amh, _, _ = rosenblatt_amh(U_a, theta=0.5)
    assert np.all((0.0 < Z_amh) & (Z_amh < 1.0))
