from __future__ import annotations

import io
import logging
from typing import List, BinaryIO

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

st.title("Data")

st.markdown(
    "Upload a **CSV** or **Parquet** file, pick numeric columns (2+), "
    "and store it in the session."
)

uploaded = st.file_uploader(
    "Upload CSV/Parquet",
    type=["csv", "parquet"],
)


@st.cache_data(show_spinner=False)
def _read_file(buf: BinaryIO, name: str) -> pd.DataFrame:
    """
    Read a CSV or Parquet file-like object into a DataFrame.

    Args:
        buf (BinaryIO): Binary file-like buffer positioned at start.
        name (str): Original filename (used to infer format).

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        ValueError: If extension is unsupported.
    """
    lower = name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(buf)
    if lower.endswith(".parquet"):
        return pd.read_parquet(buf)
    raise ValueError(
        "Unsupported file extension. Use .csv or .parquet."
    )


def _ensure_min_columns(cols: List[str], k: int = 2) -> None:
    """
    Validate a minimum number of selected columns.

    Args:
        cols (List[str]): Selected column names.
        k (int): Minimum required columns (default 2).

    Raises:
        ValueError: If fewer than k columns are provided.
    """
    if len(cols) < k:
        raise ValueError(
            f"Select at least {k} numeric columns (got {len(cols)})."
        )


if uploaded is not None:
    # Convert UploadedFile to a BytesIO to satisfy strict typing.
    data = uploaded.getvalue()
    df = _read_file(io.BytesIO(data), uploaded.name)

    st.write(
        f"Loaded **{uploaded.name}**: "
        f"{df.shape[0]} rows × {df.shape[1]} cols"
    )
    st.dataframe(df.head(25), use_container_width=True)

    numeric_cols: List[str] = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    st.divider()
    st.subheader("Column selection")

    default_cols: List[str] = (
        numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
    )
    picked = st.multiselect(
        "Numeric columns (choose at least 2)",
        options=numeric_cols,
        default=default_cols,
    )

    try:
        _ensure_min_columns(picked, k=2)
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()

    na_policy = st.selectbox(
        "Missing values policy",
        ["drop rows with NA", "keep (as-is)"],
    )

    df_selected = df[picked].copy()
    if na_policy == "drop rows with NA":
        before = len(df_selected)
        df_selected = df_selected.dropna()
        after = len(df_selected)
        if after < before:
            logger.info(
                "Dropped %d rows with NA (from %d to %d).",
                before - after,
                before,
                after,
            )

    st.session_state["data_df"] = df_selected
    st.success(
        f"Saved to session: "
        f"{df_selected.shape[0]} rows × {df_selected.shape[1]} cols"
    )

    st.info(
        "Next: go to **Calibrate** to fit a copula "
        "(placeholder in this step)."
    )
else:
    st.info("Upload a file to proceed.")
