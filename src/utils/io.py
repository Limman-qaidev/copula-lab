from __future__ import annotations

import csv
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


def read_csv_columns(
    path: str,
    columns: Sequence[str],
    encoding: str = "utf-8",
    *,
    drop_nan: bool = False,
) -> NDArray[np.float64]:
    """Load numeric columns from a CSV file into a dense array.

    Args:
        path: Filesystem path to the CSV file.
        columns: Column names to extract. Order is preserved.
        encoding: Text encoding used to decode the CSV.
        drop_nan: When ``True`` rows containing ``NaN`` values are removed
            silently. Otherwise a ``ValueError`` is raised if non-finite
            values are found. Defaults to ``False``.

    Returns:
        A ``float64`` dense matrix with shape ``(n_obs, len(columns))``.

    Raises:
        ValueError: If the file does not exist, the columns are missing,
            non-numeric entries are found, or fewer than 20 observations are
            available after validation.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"El archivo no existe: {file_path}")

    if not columns:
        raise ValueError("Debes seleccionar al menos una columna.")

    dataset: Optional[NDArray[np.float64]] = None

    pandas_spec = importlib.util.find_spec("pandas")
    if pandas_spec is not None:
        pandas_module: Any = importlib.import_module("pandas")
        try:
            frame = pandas_module.read_csv(
                file_path,
                usecols=list(columns),
                encoding=encoding,
            )
        except ValueError as exc:  # missing columns or parse issue
            raise ValueError(
                "No se pudieron leer las columnas solicitadas del CSV."
            ) from exc

        numeric = frame.apply(pandas_module.to_numeric, errors="coerce")
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

    finite_mask = np.isfinite(dataset)
    if drop_nan:
        keep_rows = np.all(finite_mask, axis=1)
        dataset = dataset[keep_rows, :]
        if dataset.size == 0:
            raise ValueError(
                "No hay observaciones válidas tras eliminar filas con NaN."
            )
    elif not np.all(finite_mask):
        raise ValueError("Se encontraron valores no numéricos o NaN.")

    n_obs = dataset.shape[0]
    if n_obs < 20:
        raise ValueError("Se requieren al menos 20 observaciones.")

    return np.asarray(dataset, dtype=np.float64)


def read_csv_2cols(
    path: str,
    col_x: str,
    col_y: str,
    encoding: str = "utf-8",
    *,
    drop_nan: bool = False,
) -> NDArray[np.float64]:
    """Backward compatible helper that reads exactly two columns.

    Args:
        path: Filesystem path to the CSV file.
        col_x: Name of the first numeric column.
        col_y: Name of the second numeric column.
        encoding: Text encoding used to decode the CSV.
        drop_nan: Forwarded to :func:`read_csv_columns`.

    Returns:
        A ``(n, 2)`` array containing the requested columns as ``float64``.
    """

    return read_csv_columns(
        path,
        columns=(col_x, col_y),
        encoding=encoding,
        drop_nan=drop_nan,
    )
