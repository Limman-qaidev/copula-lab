from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils.io import read_csv_columns  # noqa: E402


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
