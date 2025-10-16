from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils.transforms import empirical_pit  # noqa: E402


def test_empirical_pit_rejects_nans() -> None:
    data = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64)

    with pytest.raises(ValueError):
        empirical_pit(data)


def test_empirical_pit_average_ranks_with_ties() -> None:
    data = np.array(
        [
            [2.0, 1.0, 3.0],
            [2.0, 2.0, 1.0],
            [5.0, 2.0, 2.0],
            [9.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )

    result = empirical_pit(data)
    n = data.shape[0] + 1.0

    expected_first = np.array([1.5, 1.5, 3.0, 4.0], dtype=np.float64) / n
    expected_second = np.array([1.0, 2.5, 2.5, 4.0], dtype=np.float64) / n
    expected_third = np.array([3.5, 1.0, 2.0, 3.5], dtype=np.float64) / n

    np.testing.assert_allclose(result[:, 0], expected_first)
    np.testing.assert_allclose(result[:, 1], expected_second)
    np.testing.assert_allclose(result[:, 2], expected_third)


def test_empirical_pit_outputs_in_open_unit_interval() -> None:
    rng = np.random.default_rng(123)
    data = rng.normal(size=(30, 3))

    result = empirical_pit(data)

    assert np.all(result > 0.0)
    assert np.all(result < 1.0)
