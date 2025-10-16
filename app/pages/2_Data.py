from __future__ import annotations

import csv
import io
import logging
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

from src.utils import session as session_utils
from src.utils.io import read_csv_2cols
from src.utils.transforms import empirical_pit

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


st.title("Data")

st.markdown(
    """
Sube un archivo **CSV**, elige dos columnas numéricas y construye las
pseudo-observaciones empíricas que se usarán en el resto de la aplicación.
"""
)

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

default_y = 1 if len(header_columns) > 1 else 0
col_x = st.selectbox("Columna X", options=header_columns, index=0)
col_y = st.selectbox("Columna Y", options=header_columns, index=default_y)

if col_x == col_y:
    st.warning("Selecciona dos columnas distintas para construir U.")

build = st.button("Construir U (PIT empírico)")

if build:
    if col_x == col_y:
        st.error("Las columnas deben ser diferentes.")
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)

    try:
        data = read_csv_2cols(
            str(tmp_path), col_x=col_x, col_y=col_y, encoding=encoding
        )
    except ValueError as exc:
        st.error(str(exc))
        tmp_path.unlink(missing_ok=True)
        st.stop()

    logger.info(
        "Datos cargados: archivo=%s, columnas=(%s, %s), n=%d",
        uploaded.name,
        col_x,
        col_y,
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

    tmp_path.unlink(missing_ok=True)

    st.success(f"Se almacenaron U en sesión (n={U.shape[0]}).")
    st.scatter_chart({"U1": U[:, 0], "U2": U[:, 1]})
