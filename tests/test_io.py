from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils.io import read_csv_2cols, read_csv_columns  # noqa: E402


def test_read_csv_columns_loads_numeric_data(tmp_path: Path) -> None:
    rows = ["x,y,z"]
    rows.extend(f"{i},{i + 1},{i + 2}" for i in range(25))

    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    data = read_csv_columns(str(csv_path), columns=["x", "z"])

    assert data.shape == (25, 2)
    assert data.dtype == np.float64
    np.testing.assert_allclose(data[:, 0], np.arange(25, dtype=np.float64))
    np.testing.assert_allclose(data[:, 1], np.arange(2, 27, dtype=np.float64))


def test_read_csv_columns_drop_nan(tmp_path: Path) -> None:
    rows = ["a,b"]
    rows.extend(f"{i},{i * 2}" for i in range(10))
    rows.append("10,")
    rows.extend(f"{i},{i * 2}" for i in range(11, 35))

    csv_path = tmp_path / "with_nan.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    filtered = read_csv_columns(
        str(csv_path), columns=["a", "b"], drop_nan=True
    )

    assert filtered.shape == (34, 2)
    assert np.all(np.isfinite(filtered))


def test_read_csv_2cols_wrapper(tmp_path: Path) -> None:
    rows = ["x,y,z"]
    rows.extend(f"{i},{i + 1},{i + 2}" for i in range(25))

    csv_path = tmp_path / "two_cols.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    data = read_csv_2cols(str(csv_path), "x", "y")

    assert data.shape == (25, 2)
    np.testing.assert_allclose(data[:, 0], np.arange(25, dtype=np.float64))
