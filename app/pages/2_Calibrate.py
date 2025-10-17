from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils import session as session_utils  # noqa: E402
from src.utils.dependence import kendall_tau_matrix  # noqa: E402
from src.utils.types import FloatArray  # noqa: E402
from src.workflows.calibration import (  # noqa: E402
    CalibrationOutcome,
    CalibrationSpec,
    get_specs_for_family,
    list_family_names,
    run_spec,
)

logger = logging.getLogger(__name__)


def _require_pseudo_obs() -> FloatArray:
    if not session_utils.has_U():
        st.error("Pseudo-observations are required before calibration.")
        st.page_link("pages/1_Data.py", label="Open Data page", icon="📄")
        st.stop()

    U_raw = session_utils.get_U()
    if U_raw is None:
        st.error(
            "Failed to retrieve pseudo-observations from the session state."
        )
        st.stop()

    data = np.asarray(U_raw, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 2:
        st.error("Pseudo-observations must be a (n, d) array with n >= 2.")
        st.stop()
    if data.shape[1] < 2:
        st.error("At least two dimensions are required for calibration.")
        st.stop()
    return data


def _render_fit_summary(outcome: CalibrationOutcome) -> None:
    result = outcome.result
    display_params = outcome.display
    if display_params:
        summary = ", ".join(display_params)
    else:
        summary = ", ".join(
            f"{key}={value:.4f}" for key, value in result.params.items()
        )
    st.success(f"{result.family} via {result.method}: {summary}")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric(
        "Log-likelihood",
        f"{result.loglik:.3f}" if result.loglik is not None else "—",
    )
    metrics_col2.metric(
        "AIC",
        f"{result.aic:.3f}" if result.aic is not None else "—",
    )
    metrics_col3.metric(
        "BIC",
        f"{result.bic:.3f}" if result.bic is not None else "—",
    )
    st.json(
        {
            "family": result.family,
            "method": result.method,
            "params": result.params,
        }
    )


def _calibrate_many(
    specs: Iterable[CalibrationSpec],
    U: FloatArray,
    labels: Tuple[str, ...] | None,
) -> None:
    for spec in specs:
        try:
            outcome = run_spec(spec, U, labels)
        except ValueError as exc:
            st.warning(
                "Failed to calibrate {family} ({method}): {msg}".format(
                    family=spec.family,
                    method=spec.method,
                    msg=exc,
                )
            )
            continue
        logger.info(
            "Calibration completed: family=%s method=%s params=%s",
            outcome.result.family,
            outcome.result.method,
            outcome.result.params,
        )
        session_utils.append_fit_result(outcome.result)
        st.session_state["fit_result"] = outcome.result
        _render_fit_summary(outcome)


def main() -> None:
    st.title("Calibrate")
    st.caption("Estimate copula parameters from stored pseudo-observations.")

    U = _require_pseudo_obs()
    n_obs, dim = U.shape
    labels = st.session_state.get("U_columns")
    label_tuple = tuple(labels) if isinstance(labels, (list, tuple)) else None

    dataset_entry = session_utils.get_dataset()
    if dataset_entry is not None:
        dataset_values, dataset_columns = dataset_entry
        if dataset_values.shape[1] == dim:
            joined = ", ".join(dataset_columns)
            st.caption(
                f"Using pseudo-observations derived from columns: {joined}"
            )
        else:
            st.caption("Dataset dimensions do not match pseudo-observations.")

    logger.info("Calibration page loaded: n_obs=%d dim=%d", n_obs, dim)
    st.write(f"Pseudo-observations in memory: n={n_obs}, d={dim}")

    tau_matrix = kendall_tau_matrix(U)
    avg_tau = float(np.mean(tau_matrix[np.triu_indices(dim, 1)]))
    st.subheader("Dependence overview")
    st.write("Average Kendall's tau: ", avg_tau)

    families = list_family_names()
    family = st.selectbox("Copula family", families)

    specs_for_family = get_specs_for_family(family, dim)
    method_labels = [spec.method for spec in specs_for_family]
    if not method_labels:
        st.warning("No calibration methods are available for this dimension.")
        st.stop()

    default_methods = [method_labels[0]]
    selected_methods = st.multiselect(
        "Estimation methods",
        method_labels,
        default=default_methods,
        help=(
            "Select one or more estimators to store their calibrated copulas."
        ),
    )

    if st.button("Estimate selected methods", type="primary"):
        if not selected_methods:
            st.warning(
                "Choose at least one estimation method before calibrating."
            )
        else:
            chosen_specs = tuple(
                spec
                for spec in specs_for_family
                if spec.method in selected_methods
            )
            _calibrate_many(chosen_specs, U, label_tuple)


main()
