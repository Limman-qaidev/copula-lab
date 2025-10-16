from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.estimators.tau_inversion import rho_from_tau_gaussian  # noqa: E402
from src.utils import session as session_utils  # noqa: E402
from src.utils.dependence import kendall_tau  # noqa: E402
from src.utils.modelsel import (  # noqa: E402
    gaussian_pseudo_loglik,
    information_criteria,
)
from src.utils.results import FitResult  # noqa: E402
from src.utils.rosenblatt import rosenblatt_gaussian  # noqa: E402
from src.utils.transforms import empirical_pit  # noqa: E402
from src.utils.types import FloatArray  # noqa: E402


@dataclass(frozen=True)
class StudyArtifacts:
    data: FloatArray
    U: FloatArray
    rosenblatt: FloatArray
    fit: FitResult
    ks_pvalue: float
    figures: List[Path]


FIGURE_DIR = ROOT_DIR / "docs" / "assets" / "figures" / "06_practice_notes"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def _load_example_dataset(
    n: int = 400, rho: float = 0.65
) -> Tuple[np.ndarray, List[str]]:
    """Generate a reproducible Gaussian sample with correlation ``rho``."""

    rng = np.random.default_rng(session_utils.get_seed())
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    sample = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=n)
    columns = ["Spread", "Loss"]
    logger.info("Loaded study dataset: n=%d rho=%.3f", n, rho)
    return sample.astype(np.float64), columns


def _run_pipeline() -> StudyArtifacts:
    data, columns = _load_example_dataset()

    U = empirical_pit(data)
    logger.info(
        "Constructed pseudo-observations for study pipeline: shape=%s",
        U.shape,
    )

    tau = kendall_tau(U)
    rho_hat = rho_from_tau_gaussian(tau)
    fit = FitResult(
        family="Gaussian", params={"rho": rho_hat}, method="Tau inversion"
    )
    logger.info(
        "Calibrated Gaussian copula via tau inversion: rho_hat=%.3f", rho_hat
    )

    loglik = gaussian_pseudo_loglik(U, rho_hat)
    aic, bic = information_criteria(loglik, k_params=1, n=U.shape[0])
    fit = fit.with_metrics(loglik=loglik, aic=aic, bic=bic)

    Z, pval = rosenblatt_gaussian(U, rho_hat)
    logger.info("Computed Rosenblatt transform: ks_pvalue=%.3f", pval)

    figures = _save_figures(data, columns, U, Z, fit, pval)

    return StudyArtifacts(
        data=data,
        U=U,
        rosenblatt=Z,
        fit=fit,
        ks_pvalue=float(pval),
        figures=figures,
    )


def _save_figures(
    data: FloatArray,
    columns: List[str],
    U: FloatArray,
    Z: FloatArray,
    fit: FitResult,
    pval: float,
) -> List[Path]:
    figure_paths: List[Path] = []

    # Raw data scatter plot
    fig1, ax1 = plt.subplots()
    ax1.scatter(
        data[:, 0], data[:, 1], alpha=0.7, color="#2563eb", edgecolor="white"
    )
    ax1.set_title("Example credit portfolio sample")
    ax1.set_xlabel(columns[0])
    ax1.set_ylabel(columns[1])
    path1 = FIGURE_DIR / "practice_pipeline_raw.png"
    fig1.savefig(path1, bbox_inches="tight", dpi=200)
    plt.close(fig1)
    figure_paths.append(path1)

    # Pseudo-observations scatter plot
    fig2, ax2 = plt.subplots()
    ax2.scatter(
        U[:, 0], U[:, 1], alpha=0.7, color="#16a34a", edgecolor="white"
    )
    ax2.set_title("Empirical PIT pseudo-observations")
    ax2.set_xlabel("U1")
    ax2.set_ylabel("U2")
    path2 = FIGURE_DIR / "practice_pipeline_pit.png"
    fig2.savefig(path2, bbox_inches="tight", dpi=200)
    plt.close(fig2)
    figure_paths.append(path2)

    # Rosenblatt diagnostics
    fig3, axes = plt.subplots(1, 2, figsize=(8, 3))
    bins = np.linspace(0.0, 1.0, 21)
    axes[0].hist(Z[:, 0], bins=bins, color="#f97316", edgecolor="white")
    axes[0].set_title("Rosenblatt dim 1")
    axes[0].set_xlabel("Z1")
    axes[0].set_ylabel("Frequency")
    axes[1].hist(Z[:, 1], bins=bins, color="#facc15", edgecolor="white")
    axes[1].set_title("Rosenblatt dim 2")
    axes[1].set_xlabel("Z2")
    axes[1].set_ylabel("Frequency")
    fig3.suptitle(
        f"Gaussian fit (rho={fit.params['rho']:.2f}), KS p-value={pval:.3f}"
    )
    path3 = FIGURE_DIR / "practice_pipeline_rosenblatt.png"
    fig3.tight_layout()
    fig3.savefig(path3, bbox_inches="tight", dpi=200)
    plt.close(fig3)
    figure_paths.append(path3)

    logger.info(
        "Saved study figures: %s", ", ".join(str(p) for p in figure_paths)
    )
    return figure_paths


st.title("Study")
st.caption(
    "Replay the full Copula Lab workflow on a built-in dataset "
    "and export study figures."
)

st.write(
    "Use this page to generate a small dataset, build pseudo-observations, "
    "fit a Gaussian copula, and inspect basic diagnostics. The resulting "
    "figures are stored under `docs/assets/figures/06_practice_notes/` so "
    "they can be reused in practice notes or presentations."
)

if st.button("Run study pipeline", type="primary"):
    results = _run_pipeline()
    fit = results.fit
    st.success(
        "Pipeline complete: Gaussian copula calibrated via tau inversion.",
        icon="✅",
    )
    st.json(
        {
            "rho_hat": f"{fit.params['rho']:.4f}",
            "loglik": f"{fit.loglik:.3f}" if fit.loglik is not None else None,
            "AIC": f"{fit.aic:.3f}" if fit.aic is not None else None,
            "BIC": f"{fit.bic:.3f}" if fit.bic is not None else None,
            "KS p-value": f"{results.ks_pvalue:.3f}",
        }
    )

    captions = {
        "practice_pipeline_raw.png": (
            "Figure 1. Simulated credit spread vs. loss sample (data stage)."
        ),
        "practice_pipeline_pit.png": (
            "Figure 2. Empirical PIT pseudo-observations (calibration input)."
        ),
        "practice_pipeline_rosenblatt.png": (
            "Figure 3. Rosenblatt diagnostics with uniform histograms and "
            "KS p-value."
        ),
    }

    for path in results.figures:
        st.image(str(path), caption=captions[path.name])
else:
    st.info(
        "Press the button to generate the sample, run calibration, and export "
        "diagnostics figures."
    )
