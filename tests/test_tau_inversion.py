from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.estimators.tau_inversion import (  # noqa: E402
    _debye1,
    _tau_amh,
    _tau_joe,
    theta_from_tau_amh,
    theta_from_tau_clayton,
    theta_from_tau_frank,
    theta_from_tau_gumbel,
    theta_from_tau_joe,
)


def test_theta_from_tau_clayton_matches_formula() -> None:
    tau = 0.4
    theta = theta_from_tau_clayton(tau)
    assert theta == pytest.approx(2.0 * tau / (1.0 - tau))


def test_theta_from_tau_gumbel_matches_formula() -> None:
    tau = 0.35
    theta = theta_from_tau_gumbel(tau)
    assert theta == pytest.approx(1.0 / (1.0 - tau))


def test_theta_from_tau_frank_round_trip() -> None:
    theta_true = 5.0
    debye_term = 4.0 * _debye1(theta_true) / (theta_true * theta_true)
    tau_true = 1.0 - 4.0 / theta_true + debye_term
    theta_est = theta_from_tau_frank(tau_true)
    assert theta_est == pytest.approx(theta_true, rel=1e-4)


def test_theta_from_tau_joe_round_trip() -> None:
    theta_true = 2.5
    tau_true = _tau_joe(theta_true)
    theta_est = theta_from_tau_joe(tau_true)
    assert theta_est == pytest.approx(theta_true, rel=1e-4)


def test_theta_from_tau_amh_positive_round_trip() -> None:
    theta_true = 0.5
    tau_true = _tau_amh(theta_true)
    theta_est = theta_from_tau_amh(tau_true)
    assert theta_est == pytest.approx(theta_true, rel=1e-4)


@pytest.mark.parametrize("tau", [-0.1, -0.4])
def test_theta_from_tau_clayton_rejects_negative_tau(tau: float) -> None:
    with pytest.raises(ValueError):
        theta_from_tau_clayton(tau)


def test_theta_from_tau_gumbel_rejects_negative_tau() -> None:
    with pytest.raises(ValueError):
        theta_from_tau_gumbel(-0.2)


def test_theta_from_tau_amh_rejects_negative_tau() -> None:
    with pytest.raises(ValueError):
        theta_from_tau_amh(-0.1)
