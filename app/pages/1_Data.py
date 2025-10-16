from __future__ import annotations

import csv
import importlib
import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
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
        raise ValueError(f"Invalid encoding provided: {encoding}") from exc


def _extract_header(raw: bytes, encoding: str) -> List[str]:
    with _safe_decode(raw, encoding) as buffer:
        reader = csv.reader(buffer)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("The uploaded file is empty.") from exc

    if len(header) < 2:
        raise ValueError("The CSV must contain at least two columns.")
    return header


def _display_preview(
    raw: bytes,
    columns: Iterable[str],
    encoding: str,
    header: List[str],
) -> np.ndarray:
    selected = list(columns)
    if not selected:
        raise ValueError("Select at least one column for the preview.")

    pandas_spec = importlib.util.find_spec("pandas")
    if pandas_spec is not None:
        pandas_module = importlib.import_module("pandas")
        with io.BytesIO(raw) as buffer:
            frame = pandas_module.read_csv(buffer, encoding=encoding)
        preview: np.ndarray = frame[selected].to_numpy(dtype=np.float64)
        preview_frame = frame[selected].head(200)
        st.subheader("Preview (first 200 rows)")
        st.dataframe(preview_frame, use_container_width=True, hide_index=True)
        return preview

    indices: List[int] = []
    for column in selected:
        try:
            indices.append(header.index(column))
        except ValueError as exc:  # pragma: no cover - guarded by header usage
            raise ValueError(
                f"Column '{column}' is not present in the header."
            ) from exc

    with io.BytesIO(raw) as buffer:
        array = np.genfromtxt(
            buffer,
            delimiter=",",
            skip_header=1,
            usecols=tuple(indices),
            dtype=np.float64,
            encoding=encoding,
        )

    if array.size == 0:
        raise ValueError("The CSV preview could not be constructed.")

    preview_array = np.asarray(array, dtype=np.float64)
    if preview_array.ndim == 1:
        preview_array = np.reshape(preview_array, (preview_array.size, 1))

    subset = preview_array[: min(200, preview_array.shape[0]), :]
    preview_dict = {
        column: subset[:, idx] for idx, column in enumerate(selected)
    }
    st.subheader("Preview (first 200 rows)")
    st.dataframe(
        preview_dict,
        use_container_width=True,
        hide_index=True,
    )
    return preview_array


def _display_histograms(data: np.ndarray, columns: Iterable[str]) -> None:
    selected = list(columns)
    if data.size == 0 or not selected:
        return

    st.subheader("Column histograms (preview slice)")
    for idx, column in enumerate(selected):
        values = data[: min(200, data.shape[0]), idx]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            st.warning(f"Column '{column}' has no finite values in preview.")
            continue
        hist, bin_edges = np.histogram(finite, bins=20)
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            st.bar_chart(hist.astype(np.int64))
            st.caption(
                "Matplotlib not installed; displaying counts without bin "
                "labels."
            )
            continue

        fig, ax = plt.subplots()
        ax.hist(
            finite, bins=bin_edges.tolist(), color="#2563eb", edgecolor="white"
        )
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)


st.title("Data")
st.caption("Upload data, choose columns, and build pseudo-observations.")

with st.container():
    config_col, action_col = st.columns((2, 1))

    with config_col:
        encoding = st.text_input("Encoding", value="utf-8")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        drop_nan = st.checkbox(
            "Drop rows with missing values before building U",
            value=True,
        )

    if uploaded is None:
        st.info("Upload a CSV file to get started.")
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
            "Columns (pick at least two)",
            options=header_columns,
            default=default_selection,
        )
        build = st.button("Build U (empirical PIT)", type="primary")

    if not selected_columns:
        st.warning("Select one or more columns to continue.")
        st.stop()

    if len(selected_columns) < 2:
        st.warning("Choose at least two columns to construct a copula.")
        st.stop()

preview_data: Optional[np.ndarray]

try:
    preview_data = _display_preview(
        raw_bytes, selected_columns, encoding, header_columns
    )
except ValueError as exc:
    st.warning(str(exc))
    preview_data = None
else:
    _display_histograms(preview_data, selected_columns)

if not build:
    st.info("Press the button to generate pseudo-observations.")
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
    "Loaded dataset successfully: file=%s columns=%s n=%d",
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

logger.info("Stored pseudo-observations: n=%d d=%d", *U.shape)

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
    "Pseudo-observations saved to the session: "
    f"n={U.shape[0]}, d={U.shape[1]}"
)

if U.shape[1] >= 2:
    st.caption("Scatter of the first two dimensions of U.")
    chart_data = {f"U{i + 1}": U[:, i] for i in range(2)}
    st.scatter_chart(chart_data)
else:
    st.info("U is univariate; the scatter plot is skipped.")

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
    "Continue with the **Calibrate** tab to fit models using "
    "the stored pseudo-observations."
)
