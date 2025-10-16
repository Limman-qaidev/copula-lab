from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm  # type: ignore[import-untyped]

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.estimators.ifm import gaussian_ifm, gaussian_ifm_corr  # noqa: E402
from src.estimators.student_t import (  # noqa: E402
    student_t_ifm,
    student_t_pmle,
)
from src.estimators.tau_inversion import (  # noqa: E402
    choose_nu_from_tail,
    rho_matrix_from_tau_student_t,
)
from src.models.copulas.student_t import StudentTCopula  # noqa: E402
from src.utils.dependence import (  # noqa: E402
    average_tail_dep_upper,
    kendall_tau_matrix,
)
from src.utils.modelsel import student_t_pseudo_loglik  # noqa: E402


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


def test_gaussian_ifm_corr_multivariate() -> None:
    rng = np.random.default_rng(2024)
    corr_true = np.array(
        [
            [1.0, 0.4, -0.2],
            [0.4, 1.0, 0.3],
            [-0.2, 0.3, 1.0],
        ],
        dtype=np.float64,
    )
    z = rng.multivariate_normal(mean=np.zeros(3), cov=corr_true, size=6000)
    u = norm.cdf(z)

    corr_hat = gaussian_ifm_corr(u)
    assert corr_hat.shape == (3, 3)
    assert np.allclose(corr_hat, corr_hat.T, atol=1e-8)
    assert corr_hat[0, 1] == pytest.approx(0.4, abs=0.05)
    assert corr_hat[1, 2] == pytest.approx(0.3, abs=0.05)


def test_student_t_ifm_estimates_parameters() -> None:
    copula = StudentTCopula(rho=0.5, nu=6.0)
    U = copula.rvs(5000, seed=321)

    corr_hat, nu_hat = student_t_ifm(U)

    assert corr_hat.shape == (2, 2)
    assert corr_hat[0, 1] == pytest.approx(0.5, abs=0.05)
    assert nu_hat == pytest.approx(6.0, abs=3.0)
    assert nu_hat > 2.0


def test_student_t_pmle_matches_true_parameters() -> None:
    copula = StudentTCopula(rho=0.35, nu=8.0)
    U = copula.rvs(1500, seed=111)

    corr_hat, nu_hat, loglik = student_t_pmle(U)

    assert corr_hat.shape == (2, 2)
    assert corr_hat[0, 1] == pytest.approx(0.35, abs=0.05)
    assert nu_hat == pytest.approx(8.0, abs=3.0)
    assert np.isfinite(loglik)

    tau_matrix = kendall_tau_matrix(U)
    baseline_corr = rho_matrix_from_tau_student_t(tau_matrix)
    baseline_nu = choose_nu_from_tail(average_tail_dep_upper(U))
    baseline_loglik = student_t_pseudo_loglik(U, baseline_corr, baseline_nu)
    assert loglik >= baseline_loglik
