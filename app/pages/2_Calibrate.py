from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, cast

import numpy as np
import streamlit as st

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

DataFrameLike = Union[np.ndarray, "pd.DataFrame"]

st.title("Calibrate (placeholder)")

raw_data = st.session_state.get("data_df")
if raw_data is None:
    st.info("Go to **Data** to upload and select columns first.")
    st.stop()

data = cast(DataFrameLike, raw_data)
rows, cols = data.shape
st.write(f"Dataset in memory: {rows} rows × {cols} cols")

family = st.selectbox(
    "Copula family",
    ["Gaussian", "StudentT", "Clayton", "Gumbel", "Frank"],
)

method = st.selectbox(
    "Calibration method",
    ["Pseudo-MLE", "Tau inversion", "Distance (CvM)"],
)

# Streamlit returns an int here (value and step are ints).
seed_val: int = int(st.number_input("Seed", value=42, step=1))

st.caption(
    "This page currently simulates a live calibration loop. "
    "In the next step we will plug in real fitting."
)

placeholder_stats = st.empty()
placeholder_plot = st.empty()
progress = st.progress(0)


@dataclass(frozen=True)
class CalibState:
    """Container for live calibration metrics (placeholder values).

    Attributes
    ----------
    neg_loglik:
        Current negative log-likelihood estimate.
    aic:
        Akaike Information Criterion.
    bic:
        Bayesian Information Criterion.
    tau:
        Kendall's tau estimate on pseudo-observations.
    rho:
        Spearman's rho estimate on pseudo-observations.
    trace:
        A 1D series to visualize as a running trace.
    """

    neg_loglik: float
    aic: float
    bic: float
    tau: float
    rho: float
    trace: np.ndarray


def _fake_state(step: int, seed: int) -> CalibState:
    """Generate placeholder metrics for the live loop.

    Args
    ----
    step
        Current iteration number (1..N).
    seed
        Integer seed for reproducible pseudo-randomness.

    Returns
    -------
    CalibState
        Structured placeholder metrics and a random trace.
    """
    rng = np.random.default_rng(seed + step)
    trace = rng.standard_normal(100).cumsum()
    return CalibState(
        neg_loglik=float(np.log1p(step) + 1.23),
        aic=100.0 + 0.5 * float(step),
        bic=120.0 + 0.7 * float(step),
        tau=0.2 + 0.001 * float(step),
        rho=0.3 + 0.001 * float(step),
        trace=trace,
    )


if st.button("Start calibration"):
    for i in range(1, 101):
        state = _fake_state(i, seed_val)
        progress.progress(i)

        with placeholder_stats.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("-loglik", f"{state.neg_loglik:.3f}")
            c2.metric("AIC", f"{state.aic:.2f}")
            c3.metric("tau (Kendall)", f"{state.tau:.3f}")
            c4.metric("rho (Spearman)", f"{state.rho:.3f}")

        # A simple running trace; will be replaced by real optimizer signals.
        placeholder_plot.line_chart(state.trace.tolist())

        time.sleep(0.03)

    st.success("Calibration finished.")

st.info(
    "Next step: wire real copulas (Gaussian/StudentT) and a streaming "
    "optimizer that updates these panels."
)
