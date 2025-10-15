from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray


def read_csv_columns(
    path: str,
    columns: Sequence[str],
    encoding: str = "utf-8",
) -> NDArray[np.float64]:
    """Load numeric columns from a CSV file into a dense array."""

    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"El archivo no existe: {file_path}")

    if not columns:
        raise ValueError("Debes seleccionar al menos una columna.")

    dataset: Optional[NDArray[np.float64]] = None

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - optional dependency
        pass
    else:
        try:
            frame = pd.read_csv(
                file_path,
                usecols=list(columns),
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
            indices = [header.index(col) for col in columns]
        except ValueError as exc:
            raise ValueError(
                "Columnas solicitadas ausentes en el encabezado del CSV."
            ) from exc

        raw = np.genfromtxt(
            file_path,
            delimiter=",",
            skip_header=1,
            usecols=tuple(indices),
            dtype=np.float64,
            encoding=encoding,
        )

        if raw.size == 0:
            raise ValueError("El CSV no contiene datos numéricos.")

        array = np.asarray(raw, dtype=np.float64)
        if array.ndim == 1:
            array = np.reshape(array, (array.size, 1))
        dataset = array

    if dataset.ndim != 2:
        raise ValueError("El CSV debe producir una matriz bidimensional.")

    if dataset.shape[1] != len(columns):
        raise ValueError(
            "Las columnas seleccionadas no se pudieron procesar correctamente."
        )

    if not np.isfinite(dataset).all():
        raise ValueError("Se encontraron valores no numéricos o NaN.")

    n_obs = dataset.shape[0]
    if n_obs < 20:
        raise ValueError("Se requieren al menos 20 observaciones.")

    return np.asarray(dataset, dtype=np.float64)
