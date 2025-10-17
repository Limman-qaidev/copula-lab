"""Regression tests for calibration helper utilities."""

from __future__ import annotations

import numpy as np

from src.workflows import calibration


def test_project_to_correlation_enforces_unit_diagonal() -> None:
    raw = np.array(
        [
            [1.2, 0.4, -0.3],
            [0.4, 1.5, 0.2],
            [-0.3, 0.2, 0.8],
        ]
    )
    corr = calibration._project_to_correlation(raw)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T, atol=1e-10)


def test_pack_and_rebuild_correlation_round_trip() -> None:
    corr = np.array(
        [
            [1.0, 0.35, -0.2],
            [0.35, 1.0, 0.25],
            [-0.2, 0.25, 1.0],
        ]
    )
    params = calibration._pack_corr_params(corr)
    rebuilt = calibration._corr_from_params(params, dim=3)
    assert rebuilt.shape == corr.shape
    assert np.allclose(rebuilt, corr, atol=1e-6)
