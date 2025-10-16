from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils import session as session_utils  # noqa: E402
from src.utils.modelsel import (  # noqa: E402
    gaussian_pseudo_loglik,
    information_criteria,
    student_t_pseudo_loglik,
)

logger = logging.getLogger(__name__)

st.title("Compare")
st.caption("Rank calibrated copula models on shared pseudo-observations.")

if not session_utils.has_U():
    st.error("Pseudo-observations are required before running comparisons.")
    st.page_link("pages/1_Data.py", label="Open Data page", icon="📄")
    st.stop()

U_raw = session_utils.get_U()
if U_raw is None:
    st.error("Failed to read pseudo-observations from the session state.")
    st.stop()

U = np.asarray(U_raw, dtype=np.float64)
if U.ndim != 2:
    st.error("Pseudo-observations must be a 2D array to compute metrics.")
    st.stop()

n_obs, dim = U.shape
logger.info(
    "Compare page loaded with pseudo-observations: n=%d d=%d", n_obs, dim
)

fit_results = list(session_utils.get_fit_results())
if not fit_results:
    st.info("Run at least one calibration before comparing models.")
    st.page_link(
        "pages/2_Calibrate.py",
        label="Open Calibrate page",
        icon="🛠️",
    )
    st.stop()

st.write(f"Models available for comparison: {len(fit_results)}")


def _load_pandas() -> Any:
    if importlib.util.find_spec("pandas") is None:
        return None
    import pandas as pd  # type: ignore

    return pd


def _sort_key(criterion: str, row: dict[str, Any]) -> tuple[int, float]:
    value = row.get(criterion)
    if value is None:
        return (1, 0.0)
    if criterion == "LogLik":
        return (0, -float(value))
    return (0, float(value))


def _format_metrics(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


rows: list[dict[str, Any]] = []
for idx, result in enumerate(fit_results):
    loglik = result.loglik
    aic = result.aic
    bic = result.bic
    if result.family == "Gaussian":
        rho = result.params.get("rho")
        if not isinstance(rho, float):
            st.warning(
                "Gaussian model without a numeric rho parameter cannot be "
                "ranked."
            )
        elif dim != 2:
            st.warning(
                "Gaussian model comparison is currently limited to "
                "bivariate data."
            )
        else:
            try:
                loglik = gaussian_pseudo_loglik(U, rho)
                aic, bic = information_criteria(loglik, k_params=1, n=n_obs)
            except ValueError as exc:
                st.warning(f"Failed to evaluate Gaussian metrics: {exc}")
            else:
                updated = result.with_metrics(loglik=loglik, aic=aic, bic=bic)
                session_utils.update_fit_result(idx, updated)
                fit_results[idx] = updated
    elif result.family == "Student t":
        rho = result.params.get("rho")
        nu = result.params.get("nu")
        if not (isinstance(rho, float) and isinstance(nu, float)):
            st.warning(
                "Student t model requires numeric rho and nu to compute "
                "metrics."
            )
        elif dim != 2:
            st.warning(
                "Student t comparison is currently limited to bivariate data."
            )
        else:
            try:
                loglik = student_t_pseudo_loglik(U, rho, nu)
                aic, bic = information_criteria(loglik, k_params=2, n=n_obs)
            except ValueError as exc:
                st.warning(f"Failed to evaluate Student t metrics: {exc}")
            else:
                updated = result.with_metrics(loglik=loglik, aic=aic, bic=bic)
                session_utils.update_fit_result(idx, updated)
                fit_results[idx] = updated
    rows.append(
        {
            "Index": idx,
            "Family": result.family,
            "Method": result.method,
            "LogLik": loglik,
            "AIC": aic,
            "BIC": bic,
            "Params": ", ".join(
                f"{key}={value:.4f}" for key, value in result.params.items()
            ),
        }
    )

criterion_label = st.selectbox(
    "Ranking criterion",
    ("LogLik", "AIC", "BIC"),
    format_func=lambda key: {
        "LogLik": "Log-likelihood (higher is better)",
        "AIC": "Akaike Information Criterion (lower is better)",
        "BIC": "Bayesian Information Criterion (lower is better)",
    }[key],
)

sorted_rows = sorted(rows, key=lambda row: _sort_key(criterion_label, row))
chart_rows = [dict(row) for row in sorted_rows]
for rank, row in enumerate(sorted_rows, start=1):
    row["Rank"] = rank
    row["LogLik"] = _format_metrics(row["LogLik"])
    row["AIC"] = _format_metrics(row["AIC"])
    row["BIC"] = _format_metrics(row["BIC"])

pd = _load_pandas()
display_columns: Iterable[str] = (
    "Rank",
    "Family",
    "Method",
    "Params",
    "LogLik",
    "AIC",
    "BIC",
)
if pd is not None:
    frame = pd.DataFrame(sorted_rows)
    st.dataframe(frame.loc[:, display_columns], use_container_width=True)
else:
    st.table(
        [{col: row[col] for col in display_columns} for row in sorted_rows]
    )

altair_spec = importlib.util.find_spec("altair")
if pd is not None and altair_spec is not None:
    altair_module = importlib.import_module("altair")
    chart_source = pd.DataFrame(chart_rows).dropna(subset=[criterion_label])
    if not chart_source.empty:
        chart = (
            altair_module.Chart(chart_source)
            .mark_bar(color="#2563eb")
            .encode(
                x=altair_module.X(
                    "Params",
                    sort=list(chart_source["Params"]),
                    title="Model parameters",
                ),
                y=altair_module.Y(criterion_label, title=criterion_label),
                tooltip=["Family", "Method", criterion_label],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

options = [row["Index"] for row in sorted_rows]
labels = [f"{row['Family']} ({row['Method']})" for row in sorted_rows]
best_index = session_utils.get_best_model_index()
if best_index in options:
    selected_idx = options.index(best_index)
else:
    selected_idx = 0

choice = st.radio(
    "Mark the best model",
    options=list(range(len(labels))),
    format_func=lambda idx: labels[idx],
    index=selected_idx,
)
session_utils.set_best_model_index(options[choice])
st.success(
    f"Current best model: {labels[choice]} (criterion: {criterion_label})."
)
