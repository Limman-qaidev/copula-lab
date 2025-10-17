from __future__ import annotations

import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


class BaseCopula(Protocol):
    """Protocol describing the minimal copula interface for diagnostics."""

    def pdf(self, U: np.ndarray) -> np.ndarray:
        """Evaluate the copula density on points inside (0, 1)^d."""

    def rvs(self, n: int, seed: int | None = None) -> np.ndarray:
        """Draw random variates from the copula."""


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
from src.workflows.calibration import (  # noqa: E402
    get_specs_for_dimension,
    reconstruct_corr,
    run_spec,
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

dataset_entry = session_utils.get_dataset()
if dataset_entry is not None:
    dataset_values, dataset_columns = dataset_entry
    if dataset_values.shape[1] == dim:
        st.caption(
            "Evaluating models on columns: " + ", ".join(dataset_columns)
        )


labels = st.session_state.get("U_columns")
label_tuple = tuple(labels) if isinstance(labels, (list, tuple)) else None

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

specs_for_dim = get_specs_for_dimension(dim)
existing_pairs = {(result.family, result.method) for result in fit_results}
missing_specs = [
    spec
    for spec in specs_for_dim
    if (spec.family, spec.method) not in existing_pairs
]
if missing_specs:
    missing_labels = ", ".join(
        f"{spec.family} ({spec.method})" for spec in missing_specs
    )
    st.info(
        "Additional copula calibrations are available: {labels}.".format(
            labels=missing_labels,
        )
    )
    if st.button("Calibrate missing copulas", type="secondary"):
        new_results: list[FitResult] = []
        for spec in missing_specs:
            try:
                outcome = run_spec(spec, U, label_tuple)
            except ValueError as exc:
                st.warning(
                    "Failed to calibrate {family} ({method}): {msg}".format(
                        family=spec.family,
                        method=spec.method,
                        msg=exc,
                    )
                )
                continue
            session_utils.append_fit_result(outcome.result)
            fit_results.append(outcome.result)
            new_results.append(outcome.result)
            logger.info(
                "Auto-calibrated %s via %s for comparison diagnostics.",
                outcome.result.family,
                outcome.result.method,
            )
        if new_results:
            st.success(
                "Added {count} copulas to the comparison set.".format(
                    count=len(new_results)
                )
            )


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


def _import_seaborn() -> Any | None:
    """Return the seaborn module if available."""

    if importlib.util.find_spec("seaborn") is None:
        return None
    return importlib.import_module("seaborn")


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
    """Backward-compatible alias retained for persisted sessions."""

    return reconstruct_corr(params, dim)


def _build_copula_model(result: FitResult, dim: int) -> BaseCopula | None:
    """Instantiate a copula model from stored calibration parameters."""

    if result.family == "Gaussian":
        corr = reconstruct_corr(result.params, dim)
        if corr is None:
            return None
        return GaussianCopula(corr=corr)

    if result.family == "Student t":
        corr = reconstruct_corr(result.params, dim)
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


def plot_density_comparison(
    U_emp: np.ndarray,
    copula_model: BaseCopula,
    title: str,
    *,
    grid_size: int = 100,
) -> None:
    """Overlay empirical and model copula densities on a shared chart."""

    data = np.asarray(U_emp, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("U_emp must be a 2D array of pseudo-observations.")
    if data.shape[1] != 2:
        raise ValueError(
            "Density comparison is implemented for bivariate copulas only."
        )
    if np.any((data <= 0.0) | (data >= 1.0)):
        raise ValueError("Pseudo-observations must lie inside (0, 1).")

    grid = np.linspace(0.001, 0.999, grid_size, dtype=np.float64)
    u1, u2 = np.meshgrid(grid, grid, indexing="xy")
    grid_points = np.column_stack([u1.ravel(), u2.ravel()])
    model_pdf = copula_model.pdf(grid_points).reshape(u1.shape)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sns_module = _import_seaborn()
    use_hexbin = data.shape[0] > 5000 or sns_module is None
    if use_hexbin:
        hex_map = ax.hexbin(
            data[:, 0],
            data[:, 1],
            gridsize=60,
            cmap="magma",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        fig.colorbar(hex_map, ax=ax, label="Empirical density")
    else:
        assert sns_module is not None
        sns_any = cast(Any, sns_module)
        kde = sns_any.kdeplot(
            x=data[:, 0],
            y=data[:, 1],
            fill=True,
            cmap="magma",
            bw_adjust=0.7,
            levels=100,
            ax=ax,
        )
        if kde.collections:
            fig.colorbar(
                kde.collections[0],
                ax=ax,
                label="Empirical density",
            )
    if sns_module is None:
        st.info("Install seaborn to access KDE-based diagnostics.")

    ax.contour(
        u1,
        u2,
        model_pdf,
        levels=10,
        colors="cyan",
        linewidths=1.0,
    )
    ax.set_xlabel("u₁")
    ax.set_ylabel("u₂")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


rows: list[dict[str, Any]] = []
for idx, result in enumerate(fit_results):
    loglik = result.loglik
    aic = result.aic
    bic = result.bic
    if result.family == "Gaussian":
        corr = reconstruct_corr(result.params, dim)
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
        corr = reconstruct_corr(result.params, dim)
        nu = result.params.get("nu")
        if corr is None or not isinstance(nu, float):
            st.warning(
                "Student t model requires correlation entries and nu to "
                "compute metrics."
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

criterion_label = "BIC"
st.caption(
    "Ranking criterion: Bayesian Information Criterion (lower is better)."
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
            .mark_bar()
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

tab_options = [
    (row["Index"], f"{row['Family']} ({row['Method']})") for row in sorted_rows
]
option_labels = [label for _, label in tab_options]
selected_labels = st.multiselect(
    "Copulas to visualize",
    option_labels,
    default=option_labels,
)
selected_indices = {
    index for index, label in tab_options if label in selected_labels
}
if not selected_indices:
    st.info("Select at least one copula to render diagnostics.")

best_row = sorted_rows[0] if sorted_rows else None
if best_row is not None:
    session_utils.set_best_model_index(best_row["Index"])
    st.success(
        "Best copula by {criterion}: {label}.".format(
            criterion=criterion_label,
            label=f"{best_row['Family']} ({best_row['Method']})",
        )
    )

models_for_tabs: list[tuple[str, BaseCopula]] = []
for row in sorted_rows:
    if row["Index"] not in selected_indices:
        continue
    result = fit_results[row["Index"]]
    model = _build_copula_model(result, dim)
    if model is None:
        message = (
            "Failed to rebuild the copula {label} for density diagnostics."
        ).format(label=f"{result.family} ({result.method})")
        st.warning(message)
        continue
    tab_label = f"{result.family} ({result.method})"
    models_for_tabs.append((tab_label, model))

if not models_for_tabs:
    st.info("No calibrated copulas are available for density diagnostics.")
else:
    st.subheader("Density comparison by copula")
    tabs = st.tabs([label for label, _ in models_for_tabs])
    for tab, (label, model) in zip(tabs, models_for_tabs):
        with tab:
            try:
                plot_density_comparison(
                    U,
                    model,
                    f"{label}: empirical vs. theoretical density",
                )
            except ValueError as exc:
                st.warning(
                    f"Failed to render density comparison for {label}: {exc}"
                )
