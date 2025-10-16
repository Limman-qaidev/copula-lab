from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.estimators.student_t import student_t_pmle  # noqa: E402
from src.estimators.tau_inversion import rho_from_tau_gaussian  # noqa: E402
from src.utils import session as session_utils  # noqa: E402
from src.utils.dependence import kendall_tau  # noqa: E402
from src.utils.modelsel import (  # noqa: E402
    gaussian_pseudo_loglik,
    information_criteria,
)
from src.utils.results import FitResult  # noqa: E402
from src.utils.rosenblatt import (  # noqa: E402
    rosenblatt_gaussian,
    rosenblatt_student_t,
)
from src.utils.transforms import empirical_pit  # noqa: E402
from src.utils.types import FloatArray  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitDiagnostics:
    fit: FitResult
    rosenblatt: FloatArray
    ks_pvalue: float
    cvm_pvalue: float


@dataclass(frozen=True)
class StudyArtifacts:
    data: FloatArray
    columns: List[str]
    U: FloatArray
    gaussian: FitDiagnostics
    student_t: FitDiagnostics
    figures: Dict[str, Path]


FIGURE_DIR = ROOT_DIR / "docs" / "assets" / "figures" / "06_practice_notes"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _load_example_dataset(
    n: int = 400, rho: float = 0.65
) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(session_utils.get_seed())
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    sample = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=n)
    columns = ["Spread", "Loss"]
    logger.info("Loaded study dataset: n=%d rho=%.3f", n, rho)
    return sample.astype(np.float64), columns


def _save_figures(
    data: FloatArray,
    columns: List[str],
    U: FloatArray,
    gaussian: FitDiagnostics,
    student_t_diag: FitDiagnostics,
) -> Dict[str, Path]:
    figure_map: Dict[str, Path] = {}

    fig1, ax1 = plt.subplots()
    ax1.scatter(
        data[:, 0], data[:, 1], alpha=0.75, color="#2563eb", edgecolor="white"
    )
    ax1.set_title("Sample data cloud")
    ax1.set_xlabel(columns[0])
    ax1.set_ylabel(columns[1])
    path1 = FIGURE_DIR / "practice_pipeline_raw.png"
    fig1.savefig(path1, bbox_inches="tight", dpi=220)
    plt.close(fig1)
    figure_map["raw"] = path1

    fig2, ax2 = plt.subplots()
    ax2.scatter(
        U[:, 0], U[:, 1], alpha=0.75, color="#16a34a", edgecolor="white"
    )
    ax2.set_title("Pseudo-observations (empirical PIT)")
    ax2.set_xlabel("U1")
    ax2.set_ylabel("U2")
    path2 = FIGURE_DIR / "practice_pipeline_pit.png"
    fig2.savefig(path2, bbox_inches="tight", dpi=220)
    plt.close(fig2)
    figure_map["pit"] = path2

    bins = np.linspace(0.0, 1.0, 21)
    fig3, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(
        gaussian.rosenblatt[:, 0],
        bins=bins,
        color="#f97316",
        edgecolor="white",
    )
    axes[0].set_title("Gaussian Z₁")
    axes[0].set_xlabel("Z1")
    axes[0].set_ylabel("Frequency")
    axes[1].hist(
        gaussian.rosenblatt[:, 1],
        bins=bins,
        color="#facc15",
        edgecolor="white",
    )
    axes[1].set_title("Gaussian Z₂")
    axes[1].set_xlabel("Z2")
    axes[1].set_ylabel("Frequency")
    fig3.suptitle(
        "Gaussian Rosenblatt diagnostics"
        f" (KS={gaussian.ks_pvalue:.3f}, CvM={gaussian.cvm_pvalue:.3f})"
    )
    path3 = FIGURE_DIR / "practice_pipeline_rosenblatt_gaussian.png"
    fig3.tight_layout()
    fig3.savefig(path3, bbox_inches="tight", dpi=220)
    plt.close(fig3)
    figure_map["rosenblatt_gaussian"] = path3

    fig4, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(
        student_t_diag.rosenblatt[:, 0],
        bins=bins,
        color="#38bdf8",
        edgecolor="white",
    )
    axes[0].set_title("Student t Z₁")
    axes[0].set_xlabel("Z1")
    axes[0].set_ylabel("Frequency")
    axes[1].hist(
        student_t_diag.rosenblatt[:, 1],
        bins=bins,
        color="#6366f1",
        edgecolor="white",
    )
    axes[1].set_title("Student t Z₂")
    axes[1].set_xlabel("Z2")
    axes[1].set_ylabel("Frequency")
    fig4.suptitle(
        "Student t Rosenblatt diagnostics "
        f"(KS={student_t_diag.ks_pvalue:.3f}, "
        f"CvM={student_t_diag.cvm_pvalue:.3f})"
    )
    path4 = FIGURE_DIR / "practice_pipeline_rosenblatt_student_t.png"
    fig4.tight_layout()
    fig4.savefig(path4, bbox_inches="tight", dpi=220)
    plt.close(fig4)
    figure_map["rosenblatt_student_t"] = path4

    paths = ", ".join(f"{key}:{value}" for key, value in figure_map.items())
    logger.info("Saved study figures: %s", paths)
    return figure_map


def _run_pipeline() -> StudyArtifacts:
    data, columns = _load_example_dataset()
    U = empirical_pit(data)
    logger.info(
        "Constructed pseudo-observations for study pipeline: shape=%s", U.shape
    )

    tau = kendall_tau(U)
    rho_gauss = rho_from_tau_gaussian(tau)
    loglik_gauss = gaussian_pseudo_loglik(U, rho_gauss)
    aic_gauss, bic_gauss = information_criteria(
        loglik_gauss, k_params=1, n=U.shape[0]
    )
    gaussian_fit = FitResult(
        family="Gaussian",
        params={"rho": rho_gauss},
        method="Tau inversion",
        loglik=loglik_gauss,
        aic=aic_gauss,
        bic=bic_gauss,
    )
    Z_gauss, ks_gauss, cvm_gauss = rosenblatt_gaussian(U, rho_gauss)
    gaussian_diag = FitDiagnostics(
        fit=gaussian_fit,
        rosenblatt=Z_gauss,
        ks_pvalue=ks_gauss,
        cvm_pvalue=cvm_gauss,
    )

    rho_t, nu_t, loglik_t = student_t_pmle(U)
    aic_t, bic_t = information_criteria(loglik_t, k_params=2, n=U.shape[0])
    student_fit = FitResult(
        family="Student t",
        params={"rho": rho_t, "nu": nu_t},
        method="PMLE (Student t)",
        loglik=loglik_t,
        aic=aic_t,
        bic=bic_t,
    )
    Z_t, ks_t, cvm_t = rosenblatt_student_t(U, rho_t, nu_t)
    student_diag = FitDiagnostics(
        fit=student_fit,
        rosenblatt=Z_t,
        ks_pvalue=ks_t,
        cvm_pvalue=cvm_t,
    )

    figures = _save_figures(data, columns, U, gaussian_diag, student_diag)
    return StudyArtifacts(
        data=data,
        columns=columns,
        U=U,
        gaussian=gaussian_diag,
        student_t=student_diag,
        figures=figures,
    )


st.title("Study")
st.caption("Replay the Copula Lab workflow on a curated dataset with exports.")

st.markdown(
    """
Use this guided notebook to recreate the full pipeline: sample generation, PIT
construction, copula calibration, and Rosenblatt diagnostics. The exported
figures live under `docs/assets/figures/06_practice_notes/` for quick reuse in
presentations.
"""
)

if st.button("Run end-to-end study", type="primary"):
    artifacts = _run_pipeline()
    st.success("Study pipeline completed successfully.", icon="✅")

    summary_cols = st.columns(2)
    for column, diagnostics in zip(
        summary_cols, (artifacts.gaussian, artifacts.student_t)
    ):
        with column:
            fit = diagnostics.fit
            st.subheader(fit.family)
            st.metric("Method", fit.method)
            st.metric("Log-likelihood", f"{fit.loglik:.3f}")
            st.metric("AIC / BIC", f"{fit.aic:.2f} / {fit.bic:.2f}")
            st.metric(
                "Uniformity (KS / CvM)",
                f"{diagnostics.ks_pvalue:.3f} / {diagnostics.cvm_pvalue:.3f}",
            )

    data_tab, pit_tab, diag_tab = st.tabs(
        ["Data", "Pseudo-observations", "Diagnostics"]
    )

    with data_tab:
        st.image(
            str(artifacts.figures["raw"]),
            caption="Figure 1. Sample credit spread vs. loss cloud.",
            use_column_width=True,
        )
        st.caption(
            "The synthetic dataset contains 400 observations " "with ρ=0.65."
        )

    with pit_tab:
        st.image(
            str(artifacts.figures["pit"]),
            caption="Figure 2. Empirical PIT pseudo-observations.",
            use_column_width=True,
        )
        preview = artifacts.U[:10, :]
        st.dataframe(
            preview,
            hide_index=True,
            use_container_width=True,
        )

    with diag_tab:
        gauss_tab, student_tab = st.tabs(["Gaussian", "Student t"])
        with gauss_tab:
            st.image(
                str(artifacts.figures["rosenblatt_gaussian"]),
                caption=(
                    "Figure 3. Gaussian Rosenblatt histograms with KS "
                    "and CvM p-values."
                ),
                use_column_width=True,
            )
        with student_tab:
            st.image(
                str(artifacts.figures["rosenblatt_student_t"]),
                caption=(
                    "Figure 4. Student t Rosenblatt histograms with KS "
                    "and CvM p-values."
                ),
                use_column_width=True,
            )
else:
    st.info(
        "Press the button to generate data, calibrate both copulas, and "
        "export the study figures."
    )
