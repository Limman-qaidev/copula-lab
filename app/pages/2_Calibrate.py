from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.estimators.ifm import gaussian_ifm  # noqa: E402
from src.estimators.student_t import (  # noqa: E402
    student_t_ifm,
    student_t_pmle,
)
from src.estimators.tau_inversion import (  # noqa: E402
    choose_nu_from_tail,
    rho_from_tau_gaussian,
    rho_from_tau_student_t,
)
from src.utils import session as session_utils  # noqa: E402
from src.utils.dependence import kendall_tau, tail_dep_upper  # noqa: E402
from src.utils.modelsel import (  # noqa: E402
    gaussian_pseudo_loglik,
    information_criteria,
    student_t_pseudo_loglik,
)
from src.utils.results import FitResult  # noqa: E402
from src.utils.types import FloatArray  # noqa: E402

logger = logging.getLogger(__name__)


def _run_gaussian_tau(U: FloatArray, tau_value: float) -> FitResult:
    rho_hat = rho_from_tau_gaussian(tau_value)
    loglik = gaussian_pseudo_loglik(U, rho_hat)
    aic, bic = information_criteria(loglik, k_params=1, n=U.shape[0])
    return FitResult(
        family="Gaussian",
        params={"rho": rho_hat},
        method="Tau inversion",
        loglik=loglik,
        aic=aic,
        bic=bic,
    )


def _run_student_tau(
    U: FloatArray, tau_value: float, lambda_upper: float
) -> FitResult:
    rho_hat = rho_from_tau_student_t(tau_value)
    nu_hat = choose_nu_from_tail(lambda_upper)
    loglik = student_t_pseudo_loglik(U, rho_hat, nu_hat)
    aic, bic = information_criteria(loglik, k_params=2, n=U.shape[0])
    return FitResult(
        family="Student t",
        params={"rho": rho_hat, "nu": nu_hat},
        method="Tau inversion",
        loglik=loglik,
        aic=aic,
        bic=bic,
    )


def _run_gaussian_ifm(U: FloatArray) -> FitResult:
    rho_hat = gaussian_ifm(U)
    loglik = gaussian_pseudo_loglik(U, rho_hat)
    aic, bic = information_criteria(loglik, k_params=1, n=U.shape[0])
    return FitResult(
        family="Gaussian",
        params={"rho": rho_hat},
        method="IFM (Gaussian)",
        loglik=loglik,
        aic=aic,
        bic=bic,
    )

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

def _run_student_ifm(U: FloatArray) -> FitResult:
    rho_hat, nu_hat = student_t_ifm(U)
    loglik = student_t_pseudo_loglik(U, rho_hat, nu_hat)
    aic, bic = information_criteria(loglik, k_params=2, n=U.shape[0])
    return FitResult(
        family="Student t",
        params={"rho": rho_hat, "nu": nu_hat},
        method="IFM (Student t)",
        loglik=loglik,
        aic=aic,
        bic=bic,
    )


def _run_student_pmle(U: FloatArray) -> FitResult:
    rho_hat, nu_hat, loglik = student_t_pmle(U)
    aic, bic = information_criteria(loglik, k_params=2, n=U.shape[0])
    return FitResult(
        family="Student t",
        params={"rho": rho_hat, "nu": nu_hat},
        method="PMLE (Student t)",
        loglik=loglik,
        aic=aic,
        bic=bic,
    )


def _load_session_pseudo_obs() -> FloatArray:
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
    if data.ndim != 2:
        st.error("Pseudo-observations must be a 2D array.")
        st.stop()

    return np.asarray(data, dtype=np.float64)


def main() -> None:
    st.title("Calibrate")
    st.caption(
        "Estimate simple copula parameters from the session "
        "pseudo-observations."
    )

    U = _load_session_pseudo_obs()

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

    metric_cols = st.columns(3)
    metric_cols[0].metric("Sample Kendall's tau", f"{sample_tau:.3f}")
    metric_cols[1].metric("Upper tail dependence", f"{lambda_upper:.3f}")
    metric_cols[2].metric("Observations", f"{n_obs}")

    family = st.selectbox("Copula family", ("Gaussian", "Student t"))

    METHOD_CHOICES: dict[str, tuple[str, ...]] = {
        "Gaussian": ("Tau inversion", "IFM (Gaussian)"),
        "Student t": (
            "Tau inversion",
            "IFM (Student t)",
            "PMLE (Student t)",
        ),
    }
    method = st.selectbox("Estimation method", METHOD_CHOICES[family])

    with st.expander("Methodology notes", expanded=False):
        if family == "Gaussian" and method == "Tau inversion":
            st.markdown(
                "Kendall's tau is mapped to Pearson's correlation using the "
                "closed-form relation."
            )
        elif family == "Gaussian":
            st.markdown(
                "IFM applies a probit transform to pseudo-observations and "
                "uses the sample correlation of the latent Gaussian variates."
            )
        elif method == "Tau inversion":
            st.markdown(
                "Tau inversion infers the correlation from Kendall's tau and "
                "sets the degrees of freedom from the empirical tail "
                "dependence."
            )
        elif method == "IFM (Student t)":
            st.markdown(
                "IFM leverages a Student t quantile transform with a "
                "tail-informed degrees-of-freedom guess before computing the "
                "latent correlation."
            )
        else:
            st.markdown(
                "PMLE maximizes the Student t copula log-likelihood using "
                "quasi-Newton optimization."
            )

    if st.button("Estimate parameters", type="primary"):
        try:
            if family == "Gaussian" and method == "Tau inversion":
                fit_result = _run_gaussian_tau(U, sample_tau)
            elif family == "Gaussian":
                fit_result = _run_gaussian_ifm(U)
            elif method == "Tau inversion":
                fit_result = _run_student_tau(U, sample_tau, lambda_upper)
            elif method == "IFM (Student t)":
                fit_result = _run_student_ifm(U)
            else:
                fit_result = _run_student_pmle(U)
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
            session_utils.append_fit_result(fit_result)
            summary = ", ".join(
                f"{key}={value:.4f}"
                for key, value in fit_result.params.items()
            )
            st.success(
                f"{fit_result.family} via {fit_result.method}: {summary}"
            )

            metrics_tab, params_tab = st.tabs(
                ["Model metrics", "Parameter table"]
            )

            with metrics_tab:
                metric_cols = st.columns(3)
                metric_cols[0].metric(
                    "Log-likelihood",
                    f"{fit_result.loglik:.3f}" if fit_result.loglik else "—",
                )
                metric_cols[1].metric(
                    "AIC", f"{fit_result.aic:.3f}" if fit_result.aic else "—"
                )
                metric_cols[2].metric(
                    "BIC", f"{fit_result.bic:.3f}" if fit_result.bic else "—"
                )

            with params_tab:
                st.json(
                    {
                        "family": fit_result.family,
                        "method": fit_result.method,
                        "params": fit_result.params,
                    }
                )


main()
