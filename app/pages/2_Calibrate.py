from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.estimators.ifm import gaussian_ifm  # noqa: E402
from src.estimators.tau_inversion import (  # noqa: E402
    choose_nu_from_tail,
    rho_from_tau_gaussian,
    rho_from_tau_student_t,
)
from src.utils import session as session_utils  # noqa: E402
from src.utils.dependence import kendall_tau, tail_dep_upper  # noqa: E402
from src.utils.types import FloatArray  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitResult:
    """Minimal structure to persist calibration results."""

    family: str
    params: dict[str, float]
    method: str
    loglik: float | None = None
    aic: float | None = None
    bic: float | None = None


def _run_tau_inversion(
    family: str, tau_value: float, lambda_upper: float
) -> FitResult:
    if family == "Gaussian":
        rho_hat = rho_from_tau_gaussian(tau_value)
        params: dict[str, float] = {"rho": rho_hat}
    else:
        rho_hat = rho_from_tau_student_t(tau_value)
        nu_hat = choose_nu_from_tail(lambda_upper)
        params = {"rho": rho_hat, "nu": nu_hat}
    return FitResult(family=family, params=params, method="Tau inversion")


def _run_gaussian_ifm(U: FloatArray) -> FitResult:
    rho_hat = gaussian_ifm(U)
    return FitResult(
        family="Gaussian",
        params={"rho": rho_hat},
        method="IFM (Gaussian)",
    )


st.title("Calibrate")
st.caption(
    "Estimate simple copula parameters from the session pseudo-observations."
)

if not session_utils.has_U():
    st.error("Pseudo-observations are required before calibration.")
    st.page_link("pages/1_Data.py", label="Open Data page", icon="📄")
    st.stop()

U_raw = session_utils.get_U()
if U_raw is None:
    st.error("Failed to retrieve pseudo-observations from the session state.")
    st.stop()

U = np.asarray(U_raw, dtype=np.float64)
if U.ndim != 2:
    st.error("Pseudo-observations must be a 2D array.")
    st.stop()

n_obs, dim = U.shape
logger.info(
    "Calibrate page loaded with pseudo-observations: n=%d d=%d", n_obs, dim
)
st.write(f"Pseudo-observations in memory: n={n_obs}, d={dim}")

if dim != 2:
    st.warning(
        "Calibration currently supports bivariate copulas. "
        "Select exactly two columns on the Data page."
    )
    st.page_link(
        "pages/1_Data.py", label="Adjust selection in Data", icon="📄"
    )
    st.stop()

try:
    sample_tau = kendall_tau(U)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

lambda_upper = tail_dep_upper(U)

metric_cols = st.columns(2)
metric_cols[0].metric("Sample Kendall's tau", f"{sample_tau:.3f}")
metric_cols[1].metric("Empirical upper tail dep.", f"{lambda_upper:.3f}")

family = st.selectbox("Copula family", ("Gaussian", "Student t"))
method = st.selectbox("Estimation method", ("Tau inversion", "IFM (Gaussian)"))

if family == "Student t":
    st.info(
        "Student t calibration currently relies on tau inversion with a "
        "heuristic degrees-of-freedom mapping."
    )

if method == "IFM (Gaussian)" and family != "Gaussian":
    st.warning("IFM is currently implemented for the Gaussian copula only.")

if st.button("Estimate parameters", type="primary"):
    try:
        if method == "Tau inversion":
            fit_result = _run_tau_inversion(family, sample_tau, lambda_upper)
        else:
            fit_result = _run_gaussian_ifm(U)
    except ValueError as exc:
        st.error(str(exc))
    else:
        logger.info(
            "Calibration completed: family=%s method=%s params=%s",
            fit_result.family,
            fit_result.method,
            fit_result.params,
        )
        st.session_state["fit_result"] = fit_result
        summary = ", ".join(
            f"{key}={value:.4f}" for key, value in fit_result.params.items()
        )
        st.success(f"{fit_result.family} via {fit_result.method}: {summary}")
        st.json(
            {
                "family": fit_result.family,
                "method": fit_result.method,
                "params": fit_result.params,
                "loglik": fit_result.loglik,
                "aic": fit_result.aic,
                "bic": fit_result.bic,
            }
        )
