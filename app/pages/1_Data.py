from __future__ import annotations

import csv
import importlib
import inspect
import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils import session as session_utils  # noqa: E402
from src.utils.io import read_csv_columns  # noqa: E402
from src.utils.transforms import empirical_pit  # noqa: E402

logger = logging.getLogger(__name__)


def _supports_width_kwarg(renderer: Any) -> bool:
    try:
        signature = inspect.signature(renderer)
    except (TypeError, ValueError):
        return False
    return "width" in signature.parameters


def _show_altair_chart(chart: Any) -> None:
    """Render an Altair chart with responsive width when available."""

    altair_renderer = getattr(st, "altair_chart")
    if _supports_width_kwarg(altair_renderer):
        try:
            altair_renderer(chart, width="stretch")
            return
        except TypeError:
            pass
    altair_renderer(chart, use_container_width=True)

def _load_preview(
    raw: bytes,
    columns: Iterable[str],
    encoding: str,
    header: List[str],
) -> np.ndarray:
    selected = list(columns)
    if not selected:
        raise ValueError("Select at least one column for the preview.")

def _show_dataframe(*, data: Any, hide_index: bool = True) -> None:
    dataframe_renderer = getattr(st, "dataframe")
    if _supports_width_kwarg(dataframe_renderer):
        try:
            dataframe_renderer(data, width="stretch", hide_index=hide_index)
            return
        except TypeError:
            pass
    dataframe_renderer(data, use_container_width=True, hide_index=hide_index)


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


def _load_preview(
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

    return preview_array


def _render_preview_table(data: np.ndarray, columns: List[str]) -> None:
    rows = min(200, data.shape[0])
    subset = data[:rows, :]
    pandas_spec = importlib.util.find_spec("pandas")
    if pandas_spec is not None:
        pandas_module = importlib.import_module("pandas")
        frame = pandas_module.DataFrame(subset, columns=columns)
        _show_dataframe(data=frame)
    else:
        _show_dataframe(
            data={column: subset[:, idx] for idx, column in enumerate(columns)}
        )


def _render_histograms(data: np.ndarray, columns: List[str]) -> None:
    if data.size == 0 or not columns:
        st.info("Upload data to preview histograms.")
        return

    rows = min(200, data.shape[0])
    subset = data[:rows, :]
    finite_mask = np.isfinite(subset)

    pandas_spec = importlib.util.find_spec("pandas")
    altair_spec = importlib.util.find_spec("altair")

    if pandas_spec is not None and altair_spec is not None:
        pandas_module = importlib.import_module("pandas")
        altair_module = importlib.import_module("altair")
        frame = pandas_module.DataFrame(subset, columns=columns)
        clean_frame = frame.replace([np.inf, -np.inf], np.nan)
        for column in columns:
            series = clean_frame[column].dropna()
            if series.empty:
                st.warning(
                    f"Column '{column}' has no finite values in the preview."
                )
                continue
            frame_series = series.to_frame(name=column)
            chart = (
                altair_module.Chart(frame_series)
                .mark_bar(color="#2563eb")
                .encode(
                    x=altair_module.X(
                        column, bin=altair_module.Bin(maxbins=20)
                    ),
                    y="count()",
                    tooltip=[column, "count()"],
                )
                .properties(height=220)
            )
            _show_altair_chart(chart)
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        for idx, column in enumerate(columns):
            finite = subset[:, idx][finite_mask[:, idx]]
            hist, _ = np.histogram(finite, bins=20)
            st.bar_chart(hist.astype(np.int64))
        st.caption("Matplotlib not installed; displayed raw counts instead.")
        return

    for idx, column in enumerate(columns):
        finite = subset[:, idx][finite_mask[:, idx]]
        if finite.size == 0:
            st.warning(f"Column '{column}' has no finite values in preview.")
            continue
        fig, ax = plt.subplots()
        ax.hist(finite, bins=20, color="#2563eb", edgecolor="white")
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
    preview_data = _load_preview(
        raw_bytes, selected_columns, encoding, header_columns
    )
except ValueError as exc:
    st.warning(str(exc))
    preview_data = None
else:
    preview_tab, histogram_tab = st.tabs(
        ["Preview (first 200 rows)", "Distribution snapshot"]
    )
    with preview_tab:
        _render_preview_table(preview_data, list(selected_columns))
    with histogram_tab:
        _render_histograms(preview_data, list(selected_columns))

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
session_utils.set_dataset(data, selected_columns)
st.session_state["U_columns"] = tuple(selected_columns)

st.success(
    "Pseudo-observations saved to the session: "
    f"n={U.shape[0]}, d={U.shape[1]}"
)

summary_tab, scatter_tab, u_hist_tab = st.tabs(
    ["Summary", "Scatter (first two dims)", "U histograms"]
)

with summary_tab:
    preview_rows = min(10, U.shape[0])
    preview = U[:preview_rows, :]
    column_labels = [
        f"U{i + 1} ({selected_columns[i]})" for i in range(U.shape[1])
    ]
    if importlib.util.find_spec("pandas") is not None:
        pandas_module = importlib.import_module("pandas")
        preview_frame = pandas_module.DataFrame(preview, columns=column_labels)
        _show_dataframe(data=preview_frame)
    else:
        preview_dict = {
            label: preview[:, idx] for idx, label in enumerate(column_labels)
        }
        _show_dataframe(data=preview_dict)

with scatter_tab:
    if U.shape[1] < 2:
        st.info("U is univariate; the scatter plot is skipped.")
    else:
        pandas_spec = importlib.util.find_spec("pandas")
        altair_spec = importlib.util.find_spec("altair")
        if pandas_spec is not None and altair_spec is not None:
            pandas_module = importlib.import_module("pandas")
            altair_module = importlib.import_module("altair")
            scatter_frame = pandas_module.DataFrame(
                {
                    "U1": U[:, 0],
                    "U2": U[:, 1],
                    "index": np.arange(U.shape[0]),
                }
            )
            chart = (
                altair_module.Chart(scatter_frame)
                .mark_circle(opacity=0.7, size=60, color="#2563eb")
                .encode(
                    x="U1",
                    y="U2",
                    tooltip=["index", "U1", "U2"],
                )
                .properties(height=400)
            )
            chart = chart.properties(width="container")
            _show_altair_chart(chart)
        else:
            chart_data = {
                f"U{i + 1}": U[:, i] for i in range(min(2, U.shape[1]))
            }
            st.scatter_chart(chart_data)

with u_hist_tab:
    _render_histograms(U, [f"U{i + 1}" for i in range(U.shape[1])])

st.info(
    "Continue with the **Calibrate** tab to fit models using "
    "the stored pseudo-observations."
)
