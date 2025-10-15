from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def read_csv_2cols(
    path: str,
    col_x: str,
    col_y: str,
    encoding: str = "utf-8",
) -> NDArray[np.float64]:
    """Load two numeric columns from a CSV file into a ``(n, 2)`` array.

    The function attempts to use :mod:`pandas` if it is available in the
    environment. When :mod:`pandas` cannot be imported, it falls back to
    :func:`numpy.genfromtxt`.

    Args:
        path:
            Path to the CSV file on disk.
        col_x:
            Name of the first column.
        col_y:
            Name of the second column.
        encoding:
            Text encoding used to decode the file. Defaults to ``"utf-8"``.

    Returns:
        ``NDArray[np.float64]`` containing the selected columns as
        floating-point numbers.

    Raises:
        ValueError: If the CSV cannot be read, the requested columns are
            missing, the data contain NaNs or non-finite values, or fewer
            than twenty observations are available.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"El archivo no existe: {file_path}")

    dataset: Optional[NDArray[np.float64]] = None

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - optional dependency
        pass
    else:
        try:
            frame = pd.read_csv(
                file_path,
                usecols=[col_x, col_y],
                encoding=encoding,
            )
        except ValueError as exc:  # missing columns or parse issue
            raise ValueError(
                "No se pudieron leer las columnas solicitadas del CSV."
            ) from exc

        numeric = frame.apply(pd.to_numeric, errors="coerce")
        dataset = np.asarray(numeric.to_numpy(dtype=np.float64))

    if dataset is None:
        with file_path.open("r", encoding=encoding, newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError("El CSV está vacío.") from exc

        try:
            idx_x = header.index(col_x)
            idx_y = header.index(col_y)
        except ValueError as exc:
            raise ValueError(
                "Columnas solicitadas ausentes en el encabezado del CSV."
            ) from exc

        raw = np.genfromtxt(
            file_path,
            delimiter=",",
            skip_header=1,
            usecols=(idx_x, idx_y),
            dtype=np.float64,
            encoding=encoding,
        )

        if raw.size == 0:
            raise ValueError("El CSV no contiene datos numéricos.")

        array = np.asarray(raw, dtype=np.float64)
        if array.ndim == 1:
            array = np.reshape(array, (1, array.size))
        dataset = array

    if dataset.ndim != 2 or dataset.shape[1] != 2:
        raise ValueError("El CSV debe aportar exactamente dos columnas.")

    if not np.isfinite(dataset).all():
        raise ValueError("Se encontraron valores no numéricos o NaN.")

    n_obs = dataset.shape[0]
    if n_obs < 20:
        raise ValueError("Se requieren al menos 20 observaciones.")

    return np.asarray(dataset, dtype=np.float64)
