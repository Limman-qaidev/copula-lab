from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.estimators.ifm import gaussian_ifm_corr  # noqa: E402
from src.estimators.student_t import (  # noqa: E402
    student_t_ifm,
    student_t_pmle,
)
from src.estimators.tau_inversion import (  # noqa: E402
    choose_nu_from_tail,
    rho_matrix_from_tau_gaussian,
    rho_matrix_from_tau_student_t,
    theta_from_tau_amh,
    theta_from_tau_clayton,
    theta_from_tau_frank,
    theta_from_tau_gumbel,
    theta_from_tau_joe,
)
from src.models.copulas.archimedean import (  # noqa: E402
    AMHCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from src.utils import session as session_utils  # noqa: E402
from src.utils.dependence import (  # noqa: E402
    average_kendall_tau,
    average_tail_dep_upper,
    kendall_tau_matrix,
)
from src.utils.modelsel import (  # noqa: E402
    gaussian_pseudo_loglik,
    information_criteria,
    student_t_pseudo_loglik,
)
from src.utils.results import FitResult  # noqa: E402
from src.utils.types import FloatArray  # noqa: E402

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


def _flatten_corr(
    corr: FloatArray, labels: Tuple[str, ...] | None
) -> Tuple[Dict[str, float], Tuple[str, ...]]:
    matrix = np.asarray(corr, dtype=np.float64)
    dim = matrix.shape[0]
    params: Dict[str, float] = {}
    display: list[str] = []
    for i in range(dim):
        for j in range(i + 1, dim):
            key = f"rho_{i + 1}_{j + 1}"
            params[key] = float(matrix[i, j])
            if labels is None:
                lhs = f"rho({i + 1},{j + 1})"
            else:
                lhs = f"rho({labels[i]}, {labels[j]})"
            display.append(f"{lhs}={matrix[i, j]:.4f}")
    return params, tuple(display)


def _fit_gaussian_tau(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> Tuple[FitResult, Tuple[str, ...]]:
    tau_matrix = kendall_tau_matrix(U)
    corr = rho_matrix_from_tau_gaussian(tau_matrix)
    loglik = gaussian_pseudo_loglik(U, corr)
    k_params = U.shape[1] * (U.shape[1] - 1) // 2
    aic, bic = information_criteria(loglik, k_params=k_params, n=U.shape[0])
    params, display = _flatten_corr(corr, labels)
    return (
        FitResult(
            family="Gaussian",
            params=params,
            method="Tau inversion",
            loglik=loglik,
            aic=aic,
            bic=bic,
        ),
        display,
    )

    if not session_utils.has_U():
        st.error("Pseudo-observations are required before calibration.")
        st.page_link("pages/1_Data.py", label="Open Data page", icon="📄")
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

def _fit_gaussian_ifm(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> Tuple[FitResult, Tuple[str, ...]]:
    corr = gaussian_ifm_corr(U)
    loglik = gaussian_pseudo_loglik(U, corr)
    k_params = U.shape[1] * (U.shape[1] - 1) // 2
    aic, bic = information_criteria(loglik, k_params=k_params, n=U.shape[0])
    params, display = _flatten_corr(corr, labels)
    return (
        FitResult(
            family="Gaussian",
            params=params,
            method="IFM",
            loglik=loglik,
            aic=aic,
            bic=bic,
        ),
        display,
    )


def _fit_student_tau(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> Tuple[FitResult, Tuple[str, ...]]:
    tau_matrix = kendall_tau_matrix(U)
    corr = rho_matrix_from_tau_student_t(tau_matrix)
    lambda_upper = average_tail_dep_upper(U)
    nu = choose_nu_from_tail(lambda_upper)
    loglik = student_t_pseudo_loglik(U, corr, nu)
    k_params = U.shape[1] * (U.shape[1] - 1) // 2 + 1
    aic, bic = information_criteria(loglik, k_params=k_params, n=U.shape[0])
    params, display = _flatten_corr(corr, labels)
    params["nu"] = float(nu)
    return FitResult(
        family="Student t",
        params=params,
        method="Tau inversion",
        loglik=loglik,
        aic=aic,
        bic=bic,
    ), display + (f"nu={nu:.4f}",)


def _fit_student_ifm(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> Tuple[FitResult, Tuple[str, ...]]:
    corr, nu = student_t_ifm(U)
    loglik = student_t_pseudo_loglik(U, corr, nu)
    k_params = U.shape[1] * (U.shape[1] - 1) // 2 + 1
    aic, bic = information_criteria(loglik, k_params=k_params, n=U.shape[0])
    params, display = _flatten_corr(corr, labels)
    params["nu"] = float(nu)
    return FitResult(
        family="Student t",
        params=params,
        method="IFM",
        loglik=loglik,
        aic=aic,
        bic=bic,
    ), display + (f"nu={nu:.4f}",)


def _fit_student_pmle(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> Tuple[FitResult, Tuple[str, ...]]:
    corr, nu, loglik = student_t_pmle(U)
    k_params = U.shape[1] * (U.shape[1] - 1) // 2 + 1
    aic, bic = information_criteria(loglik, k_params=k_params, n=U.shape[0])
    params, display = _flatten_corr(corr, labels)
    params["nu"] = float(nu)
    return FitResult(
        family="Student t",
        params=params,
        method="PMLE",
        loglik=loglik,
        aic=aic,
        bic=bic,
    ), display + (f"nu={nu:.4f}",)


CopulaType = ClaytonCopula | GumbelCopula | FrankCopula | JoeCopula | AMHCopula

CopulaBuilder = Callable[[float, int], CopulaType]


def _fit_archimedean(
    U: FloatArray,
    family: str,
    builder: CopulaBuilder,
    theta_from_tau: Callable[[float], float],
) -> Tuple[FitResult, Tuple[str, ...]]:
    avg_tau = average_kendall_tau(U)
    theta = theta_from_tau(avg_tau)
    copula = builder(theta, U.shape[1])
    density = copula.pdf(U)
    if np.any(density <= 0.0):
        raise ValueError("Copula density returned non-positive values.")
    loglik = float(np.sum(np.log(density)))
    aic, bic = information_criteria(loglik, k_params=1, n=U.shape[0])
    return FitResult(
        family=family,
        params={"theta": float(theta)},
        method="Tau inversion",
        loglik=loglik,
        aic=aic,
        bic=bic,
    ), (f"theta={theta:.4f}",)


def _build_archimedean_fit(
    U: FloatArray, family: str
) -> Tuple[FitResult, Tuple[str, ...]]:
    if family == "Clayton":

        def builder(theta: float, dim: int) -> CopulaType:
            return ClaytonCopula(theta=theta, dim=dim)

        estimator = theta_from_tau_clayton
    elif family == "Gumbel":

        def builder(theta: float, dim: int) -> CopulaType:
            return GumbelCopula(theta=theta, dim=dim)

        estimator = theta_from_tau_gumbel
    elif family == "Frank":

        def builder(theta: float, dim: int) -> CopulaType:
            return FrankCopula(theta=theta, dim=dim)

        estimator = theta_from_tau_frank
    elif family == "Joe":

        def builder(theta: float, dim: int) -> CopulaType:
            return JoeCopula(theta=theta, dim=dim)

        estimator = theta_from_tau_joe
    else:
        if U.shape[1] != 2:
            raise ValueError(
                "The AMH copula is currently supported for two dimensions."
            )

        def builder(theta: float, dim: int) -> CopulaType:
            _ = dim
            return AMHCopula(theta=theta)

        estimator = theta_from_tau_amh
    return _fit_archimedean(U, family, builder, estimator)


def _render_fit_summary(
    result: FitResult, display_params: Tuple[str, ...]
) -> None:
    if display_params:
        summary = ", ".join(display_params)
    else:
        summary = ", ".join(f"{k}={v:.4f}" for k, v in result.params.items())
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


def main() -> None:
    st.title("Calibrate")
    st.caption("Estimate copula parameters from stored pseudo-observations.")

    U = _require_pseudo_obs()
    n_obs, dim = U.shape
    labels = st.session_state.get("U_columns")
    label_tuple = tuple(labels) if isinstance(labels, (list, tuple)) else None

    logger.info("Calibration page loaded: n_obs=%d dim=%d", n_obs, dim)
    st.write(f"Pseudo-observations in memory: n={n_obs}, d={dim}")

    tau_matrix = kendall_tau_matrix(U)
    st.subheader("Dependence overview")
    st.write(
        "Average Kendall's tau: ",
        float(np.mean(tau_matrix[np.triu_indices(dim, 1)])),
    )

    families = [
        "Gaussian",
        "Student t",
        "Clayton",
        "Gumbel",
        "Frank",
        "Joe",
        "AMH",
    ]
    family = st.selectbox("Copula family", families)

    methods: Dict[str, Tuple[str, ...]] = {
        "Gaussian": ("Tau inversion", "IFM"),
        "Student t": ("Tau inversion", "IFM", "PMLE"),
        "Clayton": ("Tau inversion",),
        "Gumbel": ("Tau inversion",),
        "Frank": ("Tau inversion",),
        "Joe": ("Tau inversion",),
        "AMH": ("Tau inversion",),
    }
    method = st.selectbox("Estimation method", methods[family])

    if st.button("Estimate parameters", type="primary"):
        try:
            if family == "Gaussian" and method == "Tau inversion":
                fit_result, display = _fit_gaussian_tau(U, label_tuple)
            elif family == "Gaussian":
                fit_result, display = _fit_gaussian_ifm(U, label_tuple)
            elif family == "Student t" and method == "Tau inversion":
                fit_result, display = _fit_student_tau(U, label_tuple)
            elif family == "Student t" and method == "IFM":
                fit_result, display = _fit_student_ifm(U, label_tuple)
            elif family == "Student t":
                fit_result, display = _fit_student_pmle(U, label_tuple)
            else:
                fit_result, display = _build_archimedean_fit(U, family)
        except ValueError as exc:
            st.error(str(exc))
            return

        logger.info(
            "Calibration completed: family=%s method=%s params=%s",
            fit_result.family,
            fit_result.method,
            fit_result.params,
        )
        session_utils.append_fit_result(fit_result)
        st.session_state["fit_result"] = fit_result
        _render_fit_summary(fit_result, display)


main()
