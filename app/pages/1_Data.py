from __future__ import annotations

import csv
import importlib
import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils import session as session_utils  # noqa: E402
from src.utils.io import read_csv_columns  # noqa: E402
from src.utils.transforms import empirical_pit  # noqa: E402

logger = logging.getLogger(__name__)


def _read_header(raw: bytes, encoding: str) -> List[str]:
    """Extract the header row from a CSV byte buffer."""

    try:
        text_stream = io.TextIOWrapper(
            io.BytesIO(raw), encoding=encoding, newline=""
        )
    except LookupError as exc:  # invalid encoding name
        raise ValueError(f"Codificación inválida: {encoding}") from exc

    with text_stream:
        line = text_stream.readline()

    if not line:
        raise ValueError("El archivo está vacío.")

    reader = csv.reader([line])
    try:
        header = next(reader)
    except StopIteration as exc:  # pragma: no cover - defensive
        raise ValueError("El archivo está vacío.") from exc

    if len(header) < 2:
        raise ValueError("Se requieren al menos dos columnas.")
    return header


st.title("Data workspace")

st.markdown(
    """
Carga un archivo **CSV**, selecciona las columnas numéricas que quieras
utilizar y transforma los datos en pseudo-observaciones empíricas.
"""
)

with st.container():
    col_config, col_actions = st.columns((2, 1))

    with col_config:
        encoding = st.text_input("Codificación", value="utf-8")
        uploaded = st.file_uploader("Archivo CSV", type=["csv"])

    if uploaded is None:
        st.info("Carga un archivo CSV para continuar.")
        st.stop()

    raw_bytes = uploaded.getvalue()

    try:
        header_columns = _read_header(raw_bytes, encoding)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with col_actions:
        st.caption("Selecciona al menos dos columnas numéricas")
        default_selection = header_columns[: min(3, len(header_columns))]
        selected_columns = st.multiselect(
            "Columnas", options=header_columns, default=default_selection
        )

        build = st.button("Construir U (PIT empírico)", type="primary")

    if not selected_columns:
        st.warning("Selecciona al menos dos columnas para continuar.")
        st.stop()

    if len(selected_columns) < 2:
        st.warning("Las copulas requieren como mínimo dos columnas.")

if build:
    if len(selected_columns) < 2:
        st.error("Selecciona como mínimo dos columnas distintas.")
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)

    try:
        data = read_csv_columns(
            str(tmp_path), columns=selected_columns, encoding=encoding
        )
    except ValueError as exc:
        st.error(str(exc))
        tmp_path.unlink(missing_ok=True)
        st.stop()

    logger.info(
        "Datos cargados: archivo=%s, columnas=%s, n=%d",
        uploaded.name,
        ", ".join(selected_columns),
        data.shape[0],
    )

    try:
        U = empirical_pit(data)
    except ValueError as exc:
        st.error(str(exc))
        tmp_path.unlink(missing_ok=True)
        st.stop()

    logger.info("Pseudo-observaciones construidas para n=%d", U.shape[0])
    session_utils.set_U(U)
    st.session_state["U_columns"] = tuple(selected_columns)

    pandas_spec = importlib.util.find_spec("pandas")
    if pandas_spec is not None:
        pandas_module = importlib.import_module("pandas")
        st.session_state["data_df"] = pandas_module.DataFrame(
            data,
            columns=selected_columns,
        )
    else:
        st.session_state["data_df"] = data

    tmp_path.unlink(missing_ok=True)

    st.success(f"Se almacenaron U en sesión (n={U.shape[0]}, d={U.shape[1]}).")

    chart_data = {f"U{i + 1}": U[:, i] for i in range(min(U.shape[1], 2))}

    if U.shape[1] >= 2:
        st.caption("Visualización de las dos primeras dimensiones de U.")
        st.scatter_chart(chart_data)
    else:  # pragma: no cover - condición defensiva
        st.info("U es univariado; no hay gráfico de dispersión disponible.")

    preview_rows = min(10, U.shape[0])
    preview = U[:preview_rows, :]
    column_labels = [
        f"U{i + 1} ({selected_columns[i]})" for i in range(U.shape[1])
    ]
    preview_data = {
        label: preview[:, idx] for idx, label in enumerate(column_labels)
    }
    st.dataframe(
        preview_data,
        use_container_width=True,
        hide_index=True,
    )
