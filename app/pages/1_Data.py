from __future__ import annotations

import csv
import importlib
import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils import session as session_utils  # noqa: E402
from src.utils.io import read_csv_columns  # noqa: E402
from src.utils.transforms import empirical_pit  # noqa: E402

logger = logging.getLogger(__name__)


def _safe_decode(raw: bytes, encoding: str) -> io.TextIOWrapper:
    try:
        return io.TextIOWrapper(io.BytesIO(raw), encoding=encoding)
    except LookupError as exc:  # invalid encoding name
        raise ValueError(f"Codificación inválida: {encoding}") from exc


def _extract_header(raw: bytes, encoding: str) -> List[str]:
    with _safe_decode(raw, encoding) as buffer:
        reader = csv.reader(buffer)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("El archivo está vacío.") from exc

    if len(header) < 2:
        raise ValueError("Se requieren al menos dos columnas en el CSV.")
    return header


def _display_preview(
    raw: bytes,
    columns: Iterable[str],
    encoding: str,
) -> None:
    pandas_spec = importlib.util.find_spec("pandas")
    if pandas_spec is None:
        return

    pandas_module = importlib.import_module("pandas")
    with io.BytesIO(raw) as buffer:
        try:
            frame = pandas_module.read_csv(buffer, encoding=encoding)
        except Exception:  # pragma: no cover - fallback handled elsewhere
            return

    preview_columns = list(columns)
    preview_frame = frame[preview_columns].head(25)
    st.dataframe(preview_frame, use_container_width=True, hide_index=True)


st.title("Data")
st.caption("Carga datos, selecciona columnas y genera pseudo-observaciones.")

with st.container():
    config_col, action_col = st.columns((2, 1))

    with config_col:
        encoding = st.text_input("Codificación", value="utf-8")
        uploaded = st.file_uploader("Archivo CSV", type=["csv"])
        drop_nan = st.checkbox(
            "Eliminar filas con valores faltantes antes de construir U",
            value=True,
        )

    if uploaded is None:
        st.info("Carga un archivo CSV para comenzar.")
        st.stop()

    raw_bytes = uploaded.getvalue()

    try:
        header_columns = _extract_header(raw_bytes, encoding)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    default_selection = header_columns[: min(3, len(header_columns))]

    with action_col:
        selected_columns = st.multiselect(
            "Columnas (elige al menos dos)",
            options=header_columns,
            default=default_selection,
        )
        build = st.button("Construir U (PIT empírico)", type="primary")

    if not selected_columns:
        st.warning("Selecciona columnas para continuar.")
        st.stop()

    if len(selected_columns) < 2:
        st.warning("Elige al menos dos columnas para construir una cópula.")
        st.stop()

_display_preview(raw_bytes, selected_columns, encoding)

if not build:
    st.info("Pulsa el botón para generar pseudo-observaciones.")
    st.stop()

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
    tmp_file.write(raw_bytes)
    tmp_path = Path(tmp_file.name)

try:
    data = read_csv_columns(
        str(tmp_path),
        columns=selected_columns,
        encoding=encoding,
        drop_nan=drop_nan,
    )
except ValueError as exc:
    tmp_path.unlink(missing_ok=True)
    st.error(str(exc))
    st.stop()

logger.info(
    "Datos cargados correctamente: archivo=%s columnas=%s n=%d",
    uploaded.name,
    ", ".join(selected_columns),
    data.shape[0],
)

try:
    U = empirical_pit(data)
except ValueError as exc:
    tmp_path.unlink(missing_ok=True)
    st.error(str(exc))
    st.stop()

tmp_path.unlink(missing_ok=True)

logger.info("Pseudo-observaciones almacenadas: n=%d d=%d", *U.shape)

session_utils.set_U(U)
st.session_state["U_columns"] = tuple(selected_columns)

pandas_spec = importlib.util.find_spec("pandas")
if pandas_spec is not None:
    pandas_module = importlib.import_module("pandas")
    st.session_state["data_df"] = pandas_module.DataFrame(
        data,
        columns=list(selected_columns),
    )
else:
    st.session_state["data_df"] = data

st.success(
    "Se generaron pseudo-observaciones en sesión: "
    f"n={U.shape[0]}, d={U.shape[1]}"
)

if U.shape[1] >= 2:
    st.caption("Visualización de las dos primeras dimensiones de U.")
    chart_data = {f"U{i + 1}": U[:, i] for i in range(2)}
    st.scatter_chart(chart_data)
else:
    st.info("U es univariado; no se muestra gráfico de dispersión.")

preview_rows = min(10, U.shape[0])
preview = U[:preview_rows, :]
column_labels = [
    f"U{i + 1} ({selected_columns[i]})" for i in range(U.shape[1])
]

if importlib.util.find_spec("pandas") is not None:
    pandas_module = importlib.import_module("pandas")
    preview_frame = pandas_module.DataFrame(preview, columns=column_labels)
    st.dataframe(
        preview_frame,
        hide_index=True,
        use_container_width=True,
    )
else:
    preview_dict = {
        label: preview[:, idx] for idx, label in enumerate(column_labels)
    }
    st.dataframe(
        preview_dict,
        hide_index=True,
        use_container_width=True,
    )

st.info(
    "Puedes continuar con la pestaña **Calibrate** para ajustar modelos con"
    " las pseudo-observaciones almacenadas."
)
