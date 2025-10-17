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
from src.models.copulas.gaussian import GaussianCopula  # noqa: E402
from src.models.copulas.student_t import StudentTCopula  # noqa: E402
from src.workflows.calibration import (  # noqa: E402
    reconstruct_corr,
    run_calibration,
)


def _assert_corr_close(
    params: dict[str, float], expected: np.ndarray, atol: float
) -> None:
    corr_hat = reconstruct_corr(params, expected.shape[0])
    assert corr_hat is not None
    assert np.allclose(corr_hat, expected, atol=atol)


def test_gaussian_tau_inversion_recovers_corr() -> None:
    corr_true = np.array(
        [[1.0, 0.45, -0.3], [0.45, 1.0, 0.25], [-0.3, 0.25, 1.0]],
        dtype=np.float64,
    )
    samples = GaussianCopula(corr=corr_true).rvs(20000, seed=42)
    outcome = run_calibration("Gaussian", "Tau inversion", samples)
    _assert_corr_close(outcome.result.params, corr_true, atol=0.03)


def test_gaussian_ifm_recovers_corr() -> None:
    corr_true = np.array(
        [[1.0, 0.35, 0.2], [0.35, 1.0, -0.4], [0.2, -0.4, 1.0]],
        dtype=np.float64,
    )
    samples = GaussianCopula(corr=corr_true).rvs(20000, seed=17)
    outcome = run_calibration("Gaussian", "IFM", samples)
    _assert_corr_close(outcome.result.params, corr_true, atol=0.03)


def test_gaussian_loglik_recovers_corr() -> None:
    corr_true = np.array(
        [[1.0, 0.55, -0.25], [0.55, 1.0, 0.15], [-0.25, 0.15, 1.0]],
        dtype=np.float64,
    )
    samples = GaussianCopula(corr=corr_true).rvs(18000, seed=29)
    outcome = run_calibration("Gaussian", "Log-likelihood", samples)
    _assert_corr_close(outcome.result.params, corr_true, atol=0.03)


def test_student_tau_inversion_recovers_parameters() -> None:
    corr_true = np.array(
        [[1.0, 0.5, -0.2], [0.5, 1.0, 0.3], [-0.2, 0.3, 1.0]],
        dtype=np.float64,
    )
    nu_true = 6.5
    samples = StudentTCopula(corr=corr_true, nu=nu_true).rvs(18000, seed=7)
    outcome = run_calibration("Student t", "Tau inversion", samples)
    _assert_corr_close(outcome.result.params, corr_true, atol=0.05)
    assert outcome.result.params["nu"] == pytest.approx(nu_true, rel=0.05)


def test_student_ifm_recovers_parameters() -> None:
    corr_true = np.array(
        [[1.0, 0.4, -0.25], [0.4, 1.0, 0.35], [-0.25, 0.35, 1.0]],
        dtype=np.float64,
    )
    nu_true = 5.5
    samples = StudentTCopula(corr=corr_true, nu=nu_true).rvs(18000, seed=19)
    outcome = run_calibration("Student t", "IFM", samples)
    _assert_corr_close(outcome.result.params, corr_true, atol=0.05)
    assert outcome.result.params["nu"] == pytest.approx(nu_true, rel=0.05)


def test_student_low_nu_recovery() -> None:
    corr_true = np.array(
        [[1.0, 0.4, -0.2], [0.4, 1.0, 0.35], [-0.2, 0.35, 1.0]],
        dtype=np.float64,
    )
    nu_true = 2.0
    samples = StudentTCopula(corr=corr_true, nu=nu_true).rvs(22000, seed=13)

    outcome_tau = run_calibration("Student t", "Tau inversion", samples)
    assert outcome_tau.result.params["nu"] == pytest.approx(
        nu_true, abs=0.1
    )

    outcome_ifm = run_calibration("Student t", "IFM", samples)
    assert outcome_ifm.result.params["nu"] == pytest.approx(
        nu_true, abs=0.1
    )


def test_student_loglik_recovers_parameters() -> None:
    corr_true = np.array(
        [[1.0, 0.45, 0.2], [0.45, 1.0, -0.3], [0.2, -0.3, 1.0]],
        dtype=np.float64,
    )
    nu_true = 7.0
    samples = StudentTCopula(corr=corr_true, nu=nu_true).rvs(3000, seed=23)
    outcome = run_calibration("Student t", "Log-likelihood", samples)
    _assert_corr_close(outcome.result.params, corr_true, atol=0.07)
    assert outcome.result.params["nu"] == pytest.approx(nu_true, rel=0.05)


def test_clayton_tau_inversion_recovers_theta() -> None:
    theta_true = 2.5
    samples = ClaytonCopula(theta=theta_true, dim=3).rvs(6000, seed=29)
    outcome = run_calibration("Clayton", "Tau inversion", samples)
    assert outcome.result.params["theta"] == pytest.approx(
        theta_true, rel=0.05
    )


def test_gumbel_tau_inversion_recovers_theta() -> None:
    theta_true = 1.7
    samples = GumbelCopula(theta=theta_true, dim=3).rvs(6000, seed=31)
    outcome = run_calibration("Gumbel", "Tau inversion", samples)
    assert outcome.result.params["theta"] == pytest.approx(
        theta_true, rel=0.05
    )


def test_frank_tau_inversion_recovers_theta() -> None:
    theta_true = 4.0
    samples = FrankCopula(theta=theta_true, dim=3).rvs(6000, seed=37)
    outcome = run_calibration("Frank", "Tau inversion", samples)
    assert outcome.result.params["theta"] == pytest.approx(
        theta_true, rel=0.05
    )


def test_joe_tau_inversion_recovers_theta() -> None:
    theta_true = 1.9
    samples = JoeCopula(theta=theta_true, dim=2).rvs(2500, seed=43)
    outcome = run_calibration("Joe", "Tau inversion", samples)
    assert outcome.result.params["theta"] == pytest.approx(
        theta_true, rel=0.12
    )


def test_amh_tau_inversion_recovers_theta() -> None:
    theta_true = 0.5
    samples = AMHCopula(theta=theta_true).rvs(10000, seed=47)
    outcome = run_calibration("AMH", "Tau inversion", samples)
    assert outcome.result.params["theta"] == pytest.approx(
        theta_true, rel=0.05
    )
