from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.copulas.student_t import StudentTCopula  # noqa: E402
from src.utils.rosenblatt import (  # noqa: E402
    cond_cdf_student_t,
    gof_cvm_uniform,
    gof_ks_uniform,
    rosenblatt_gaussian,
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
