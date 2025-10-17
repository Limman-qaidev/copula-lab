from __future__ import annotations

import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:  # pragma: no cover - optional dependency for type hints
    from copulas.base import BaseCopula  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - fallback for local models

    class BaseCopula(Protocol):
        """Protocol describing the minimal PDF interface for copulas."""

        def pdf(self, U: np.ndarray) -> np.ndarray:
            """Evaluate the copula density on points inside (0, 1)^d."""

from src.models.copulas.archimedean import (  # noqa: E402
    AMHCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from src.models.copulas.gaussian import GaussianCopula  # noqa: E402
from src.models.copulas.student_t import StudentTCopula  # noqa: E402
from src.utils import session as session_utils  # noqa: E402
from src.utils.modelsel import (  # noqa: E402
    gaussian_pseudo_loglik,
    information_criteria,
    student_t_pseudo_loglik,
)
from src.utils.results import FitResult  # noqa: E402

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

dataset_entry = session_utils.get_dataset()
if dataset_entry is not None:
    dataset_values, dataset_columns = dataset_entry
    if dataset_values.shape[1] == dim:
        st.caption(
            "Evaluating models on columns: " + ", ".join(dataset_columns)
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


def _supports_width_kwarg(renderer: Any) -> bool:
    try:
        signature = inspect.signature(renderer)
    except (TypeError, ValueError):
        return False
    return "width" in signature.parameters


def _load_pandas() -> Any:
    if importlib.util.find_spec("pandas") is None:
        return None
    import pandas as pd  # type: ignore

    return pd


def _show_altair_chart(chart: Any) -> None:
    """Render an Altair chart using a width-aware fallback."""

    altair_renderer = getattr(st, "altair_chart")
    if _supports_width_kwarg(altair_renderer):
        try:
            altair_renderer(chart, width="stretch")
            return
        except TypeError:
            pass
    altair_renderer(chart, use_container_width=True)


def _show_dataframe(data: Any, *, hide_index: bool = True) -> None:
    dataframe_renderer = getattr(st, "dataframe")
    if _supports_width_kwarg(dataframe_renderer):
        try:
            dataframe_renderer(data, width="stretch", hide_index=hide_index)
            return
        except TypeError:
            pass
    dataframe_renderer(data, use_container_width=True, hide_index=hide_index)


def _sort_key(criterion: str, row: dict[str, Any]) -> tuple[int, float]:
    value = row.get(criterion)
    if value is None:
        return (1, 0.0)
    if criterion == "LogLik":
        return (0, -float(value))
    return (0, float(value))


def _format_metrics(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def _rebuild_corr(params: Mapping[str, float], dim: int) -> np.ndarray | None:
    matrix = np.eye(dim, dtype=np.float64)
    found = False
    for key, value in params.items():
        if not key.startswith("rho_"):
            continue
        parts = key.split("_")
        if len(parts) != 3:
            continue
        try:
            i = int(parts[1]) - 1
            j = int(parts[2]) - 1
        except ValueError:
            continue
        if not (0 <= i < dim and 0 <= j < dim):
            continue
        matrix[i, j] = matrix[j, i] = float(value)
        found = True
    return matrix if found else None


def _build_copula_model(result: FitResult, dim: int) -> BaseCopula | None:
    """Instantiate a copula model from stored calibration parameters."""

    if result.family == "Gaussian":
        corr = _rebuild_corr(result.params, dim)
        if corr is None:
            return None
        return GaussianCopula(corr=corr)

    if result.family == "Student t":
        corr = _rebuild_corr(result.params, dim)
        nu_value = result.params.get("nu")
        if corr is None or nu_value is None:
            return None
        return StudentTCopula(corr=corr, nu=float(nu_value))

    theta_value = result.params.get("theta")
    if theta_value is None:
        return None

    theta = float(theta_value)
    if result.family == "Clayton":
        return ClaytonCopula(theta=theta, dim=dim)
    if result.family == "Gumbel":
        return GumbelCopula(theta=theta, dim=dim)
    if result.family == "Frank":
        return FrankCopula(theta=theta, dim=dim)
    if result.family == "Joe":
        return JoeCopula(theta=theta, dim=dim)
    if result.family == "AMH":
        if dim != 2:
            logger.warning(
                "AMH copula visualization requires two dimensions; "
                "received %d.",
                dim,
            )
            return None
        return AMHCopula(theta=theta)

    logger.warning(
        "Unsupported copula family for visualization: %s",
        result.family,
    )
    return None


def plot_copula_density(
    U_emp: np.ndarray, copula_model: BaseCopula, title: str
) -> None:
    """
    Plot empirical versus model copula density on the unit square.

    Assumptions
    ----------
    - ``U_emp`` contains pseudo-observations strictly inside ``(0, 1)``.
    - The copula model exposes a ``pdf`` method compatible with ``U_emp``.

    Limitations
    -----------
    - The comparison is restricted to bivariate copulas (``d = 2``).
    - Kernel density estimates may oversmooth for multimodal structures.
    """

    data = np.asarray(U_emp, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            "U_emp must be a (n, 2) array of pseudo-observations."
        )
    if np.any((data <= 0.0) | (data >= 1.0)):
        raise ValueError(
            "Pseudo-observations must lie strictly within (0, 1)."
        )

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_facecolor("white")

    if data.shape[0] > 5000:
        hexbin = ax.hexbin(
            data[:, 0],
            data[:, 1],
            gridsize=60,
            cmap="magma",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        colorbar = fig.colorbar(hexbin, ax=ax)
        colorbar.set_label("Empirical density")
    else:
        sns.kdeplot(
            x=data[:, 0],
            y=data[:, 1],
            fill=True,
            cmap="magma",
            bw_adjust=0.7,
            levels=100,
            thresh=0.0,
            ax=ax,
            cbar=True,
            cbar_kws={"label": "Empirical density"},
        )

    grid = np.linspace(0.001, 0.999, 100)
    U1, U2 = np.meshgrid(grid, grid)
    evaluation_points = np.column_stack([U1.ravel(), U2.ravel()])
    pdf_values = copula_model.pdf(evaluation_points)
    pdf_grid = np.asarray(pdf_values, dtype=np.float64).reshape(100, 100)
    if not np.all(np.isfinite(pdf_grid)):
        raise ValueError("Model density produced non-finite values.")

    ax.contour(
        U1,
        U2,
        pdf_grid,
        levels=10,
        colors="cyan",
        linewidths=1.0,
    )
    legend_handle = Line2D(
        [0],
        [0],
        color="cyan",
        linewidth=1.0,
        label="Model density",
    )
    ax.legend(handles=[legend_handle], loc="upper right")

    ax.set_xlabel("u₁")
    ax.set_ylabel("u₂")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.tight_layout()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


rows: list[dict[str, Any]] = []
for idx, result in enumerate(fit_results):
    loglik = result.loglik
    aic = result.aic
    bic = result.bic
    if result.family == "Gaussian":
        corr = _rebuild_corr(result.params, dim)
        if corr is None:
            st.warning(
                "Gaussian model is missing correlation entries and cannot be "
                "ranked."
            )
        else:
            try:
                loglik = gaussian_pseudo_loglik(U, corr)
                k_params = dim * (dim - 1) // 2
                aic, bic = information_criteria(
                    loglik, k_params=k_params, n=n_obs
                )
            except ValueError as exc:
                st.warning(f"Failed to evaluate Gaussian metrics: {exc}")
            else:
                updated = result.with_metrics(loglik=loglik, aic=aic, bic=bic)
                session_utils.update_fit_result(idx, updated)
                fit_results[idx] = updated
    elif result.family == "Student t":
        corr = _rebuild_corr(result.params, dim)
        nu = result.params.get("nu")
        if corr is None or not isinstance(nu, float):
            st.warning(
                "Student t model requires correlation entries and "
                "nu to compute metrics."
            )
        else:
            try:
                loglik = student_t_pseudo_loglik(U, corr, nu)
                k_params = dim * (dim - 1) // 2 + 1
                aic, bic = information_criteria(
                    loglik, k_params=k_params, n=n_obs
                )
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
    _show_dataframe(frame.loc[:, display_columns])
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
        chart = chart.properties(width="container")
        _show_altair_chart(chart)

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

selected_result = fit_results[options[choice]]
model = _build_copula_model(selected_result, dim)

if dim != 2:
    st.info(
        "Copula density comparison is available only for bivariate data."
    )
elif model is None:
    st.info(
        "Unable to reconstruct the selected copula for density visualization."
    )
else:
    st.subheader("Empirical vs Model Density")
    try:
        plot_copula_density(U[:, :2], model, "Copula Density Comparison")
    except ValueError as exc:
        st.warning(f"Density plot failed: {exc}")
