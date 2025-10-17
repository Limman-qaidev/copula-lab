from __future__ import annotations

import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils import session as session_utils  # noqa: E402
from src.utils.results import FitResult  # noqa: E402
from src.utils.rosenblatt import (  # noqa: E402
    rosenblatt_amh,
    rosenblatt_clayton,
    rosenblatt_frank,
    rosenblatt_gaussian,
    rosenblatt_gumbel,
    rosenblatt_joe,
    rosenblatt_student_t,
)
from src.utils.transforms import empirical_pit  # noqa: E402
from src.utils.types import FloatArray  # noqa: E402
from src.workflows.calibration import (  # noqa: E402
    CalibrationOutcome,
    get_specs_for_dimension,
    list_family_names,
    reconstruct_corr,
    run_spec,
)

logger = logging.getLogger(__name__)


def _supports_width_kwarg(renderer: Any) -> bool:
    try:
        signature = inspect.signature(renderer)
    except (TypeError, ValueError):
        return False
    return "width" in signature.parameters


def _show_image(path: str, caption: str) -> None:
    image_renderer = getattr(st, "image")
    if _supports_width_kwarg(image_renderer):
        try:
            image_renderer(path, caption=caption, width=640)
            return
        except TypeError:
            pass
    image_renderer(path, caption=caption, use_column_width=True)


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
    calibrations: Dict[Tuple[str, str], CalibrationOutcome]
    gaussian: FitDiagnostics
    student_t: FitDiagnostics
    figures: Dict[str, Path]


FIGURE_DIR = ROOT_DIR / "docs" / "assets" / "figures" / "06_practice_notes"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _format_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def _evaluate_uniformity(
    result: FitResult, U: FloatArray
) -> Tuple[float | None, float | None]:
    dim = U.shape[1]
    try:
        if result.family == "Gaussian":
            corr = reconstruct_corr(result.params, dim)
            if corr is None:
                return (None, None)
            _, ks_val, cvm_val = rosenblatt_gaussian(U, float(corr[0, 1]))
            return float(ks_val), float(cvm_val)
        if result.family == "Student t":
            corr = reconstruct_corr(result.params, dim)
            nu_value = result.params.get("nu")
            if corr is None or nu_value is None:
                return (None, None)
            _, ks_val, cvm_val = rosenblatt_student_t(
                U, float(corr[0, 1]), float(nu_value)
            )
            return float(ks_val), float(cvm_val)
        theta_value = result.params.get("theta")
        if theta_value is None:
            return (None, None)
        theta = float(theta_value)
        if result.family == "Clayton":
            _, ks_val, cvm_val = rosenblatt_clayton(U, theta)
            return float(ks_val), float(cvm_val)
        if result.family == "Gumbel":
            _, ks_val, cvm_val = rosenblatt_gumbel(U, theta)
            return float(ks_val), float(cvm_val)
        if result.family == "Frank":
            _, ks_val, cvm_val = rosenblatt_frank(U, theta)
            return float(ks_val), float(cvm_val)
        if result.family == "Joe":
            _, ks_val, cvm_val = rosenblatt_joe(U, theta)
            return float(ks_val), float(cvm_val)
        if result.family == "AMH":
            _, ks_val, cvm_val = rosenblatt_amh(U, theta)
            return float(ks_val), float(cvm_val)
    except ValueError as exc:
        logger.warning(
            "Uniformity computation failed for %s (%s): %s",
            result.family,
            result.method,
            exc,
        )
    return (None, None)


def _select_best_outcome(
    calibrations: Dict[Tuple[str, str], CalibrationOutcome], family: str
) -> CalibrationOutcome:
    best: CalibrationOutcome | None = None
    best_score = float("inf")
    for (fam, _), outcome in calibrations.items():
        if fam != family:
            continue
        bic_value = outcome.result.bic
        score = float("inf") if bic_value is None else float(bic_value)
        if best is None or score < best_score:
            best = outcome
            best_score = score
    if best is None:
        raise ValueError(f"No calibration available for family {family}")
    return best


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

    fig1, ax1 = plt.subplots(figsize=(4.0, 4.0))
    ax1.scatter(
        data[:, 0], data[:, 1], alpha=0.75, color="#2563eb", edgecolor="white"
    )
    ax1.set_title("Sample data cloud")
    ax1.set_xlabel(columns[0])
    ax1.set_ylabel(columns[1])
    path1 = FIGURE_DIR / "practice_pipeline_raw.png"
    fig1.tight_layout()
    fig1.savefig(path1, bbox_inches="tight", dpi=220)
    plt.close(fig1)
    figure_map["raw"] = path1

    fig2, ax2 = plt.subplots(figsize=(4.0, 4.0))
    ax2.scatter(
        U[:, 0], U[:, 1], alpha=0.75, color="#16a34a", edgecolor="white"
    )
    ax2.set_title("Pseudo-observations (empirical PIT)")
    ax2.set_xlabel("U1")
    ax2.set_ylabel("U2")
    path2 = FIGURE_DIR / "practice_pipeline_pit.png"
    fig2.tight_layout()
    fig2.savefig(path2, bbox_inches="tight", dpi=220)
    plt.close(fig2)
    figure_map["pit"] = path2

    bins = np.linspace(0.0, 1.0, 21)
    fig3, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
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

    fig4, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
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

    labels = tuple(columns)
    calibrations: Dict[Tuple[str, str], CalibrationOutcome] = {}
    specs = get_specs_for_dimension(U.shape[1])
    for spec in specs:
        try:
            outcome = run_spec(spec, U, labels)
        except ValueError as exc:
            logger.warning(
                "Skipping %s (%s) during study calibration: %s",
                spec.family,
                spec.method,
                exc,
            )
            continue
        calibrations[(spec.family, spec.method)] = outcome

    logger.info(
        "Study calibrations: %s",
        ", ".join(
            f"{family} ({method})" for family, method in calibrations.keys()
        ),
    )

    gaussian_outcome = _select_best_outcome(calibrations, "Gaussian")
    student_outcome = _select_best_outcome(calibrations, "Student t")

    dim = U.shape[1]
    corr_gauss = reconstruct_corr(gaussian_outcome.result.params, dim)
    if corr_gauss is None:
        raise ValueError("Gaussian calibration lacks correlation parameters.")
    rho_gauss = float(corr_gauss[0, 1])
    Z_gauss, ks_gauss, cvm_gauss = rosenblatt_gaussian(U, rho_gauss)
    gaussian_diag = FitDiagnostics(
        fit=gaussian_outcome.result,
        rosenblatt=Z_gauss,
        ks_pvalue=ks_gauss,
        cvm_pvalue=cvm_gauss,
    )

    corr_t = reconstruct_corr(student_outcome.result.params, dim)
    nu_value = student_outcome.result.params.get("nu")
    if corr_t is None or nu_value is None:
        raise ValueError("Student t calibration requires correlation and nu.")
    rho_t = float(corr_t[0, 1])
    nu_t = float(nu_value)
    Z_t, ks_t, cvm_t = rosenblatt_student_t(U, rho_t, nu_t)
    student_diag = FitDiagnostics(
        fit=student_outcome.result,
        rosenblatt=Z_t,
        ks_pvalue=ks_t,
        cvm_pvalue=cvm_t,
    )

    figures = _save_figures(data, columns, U, gaussian_diag, student_diag)
    return StudyArtifacts(
        data=data,
        columns=columns,
        U=U,
        calibrations=calibrations,
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

    family_order = list_family_names()
    ordered_items = sorted(
        artifacts.calibrations.items(),
        key=lambda item: (
            (
                family_order.index(item[0][0])
                if item[0][0] in family_order
                else len(family_order)
            ),
            item[0][1],
        ),
    )
    catalog_rows: List[Dict[str, str]] = []
    for (family, method), outcome in ordered_items:
        ks_val, cvm_val = _evaluate_uniformity(outcome.result, artifacts.U)
        if outcome.display:
            params_display = ", ".join(outcome.display)
        else:
            params_display = ", ".join(
                f"{key}={value:.4f}"
                for key, value in outcome.result.params.items()
            )
        catalog_rows.append(
            {
                "Family": family,
                "Method": method,
                "LogLik": _format_metric(outcome.result.loglik),
                "AIC": _format_metric(outcome.result.aic),
                "BIC": _format_metric(outcome.result.bic),
                "KS": _format_metric(ks_val),
                "CvM": _format_metric(cvm_val),
                "Parameters": params_display,
            }
        )

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
        _show_image(
            str(artifacts.figures["raw"]),
            caption="Figure 1. Sample credit spread vs. loss cloud.",
        )
        st.caption(
            "The synthetic dataset contains 400 observations " "with ρ=0.65."
        )

    with pit_tab:
        _show_image(
            str(artifacts.figures["pit"]),
            caption="Figure 2. Empirical PIT pseudo-observations.",
        )
        preview = artifacts.U[:10, :]
        st.dataframe(
            preview,
            hide_index=True,
            width="stretch",
        )

    with diag_tab:
        gauss_tab, student_tab, catalog_tab = st.tabs(
            ["Gaussian", "Student t", "All copulas"]
        )
        with gauss_tab:
            _show_image(
                str(artifacts.figures["rosenblatt_gaussian"]),
                caption=(
                    "Figure 3. Gaussian Rosenblatt histograms with KS "
                    "and CvM p-values."
                ),
            )
        with student_tab:
            _show_image(
                str(artifacts.figures["rosenblatt_student_t"]),
                caption=(
                    "Figure 4. Student t Rosenblatt histograms with KS "
                    "and CvM p-values."
                ),
            )
        with catalog_tab:
            if catalog_rows:
                st.caption(
                    "Uniformity metrics and parameter summaries for all "
                    "calibrated copulas."
                )
                st.table(catalog_rows)
            else:
                st.info("No copula calibrations were generated in this run.")
else:
    st.info(
        "Press the button to generate data, calibrate both copulas, and "
        "export the study figures."
    )
