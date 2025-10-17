"""Unit tests for sandbox helper routines."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

sandbox = importlib.import_module("app.pages.5_Sandbox")


def test_compile_expression_supports_vector_inputs() -> None:
    evaluator = sandbox._compile_expression("u1 + 2 * u2", ("u1", "u2"))
    points = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    values = evaluator(points)
    assert values.shape == (2,)
    assert np.allclose(values, np.array([0.5, 1.1]))


def test_sample_custom_density_produces_expected_shape() -> None:
    def density_fn(pts: np.ndarray) -> np.ndarray:
        return np.ones(pts.shape[0], dtype=np.float64)

    samples = sandbox._sample_custom_density(density_fn, dim=3, n=50, seed=1)
    assert samples.shape == (50, 3)
    assert np.all((samples > 0.0) & (samples < 1.0))


def test_require_scalar_theta_rejects_matrix() -> None:
    params = {"theta": np.eye(2, dtype=np.float64)}
    with pytest.raises(ValueError):
        sandbox._require_scalar_theta(params, "Clayton")


def test_sample_preset_gaussian_returns_uniform_points() -> None:
    params = {"corr": np.eye(2, dtype=np.float64)}
    samples = sandbox._sample_preset(
        "Gaussian",
        dim=2,
        params=params,
        n=20,
        seed=0,
    )
    assert samples.shape == (20, 2)
    assert np.all((samples > 0.0) & (samples < 1.0))
