"""Interactive sandbox for building and sampling copulas."""

from __future__ import annotations

import ast
import io
from typing import Callable, Dict, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.models.copulas.archimedean import (
    AMHCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from src.models.copulas.gaussian import GaussianCopula
from src.models.copulas.student_t import StudentTCopula


_CLIP = 1e-6
_DEFAULT_SAMPLE_SIZE = 2000
_DEFAULT_DIMENSION = 2
_PAIRPLOT_MAX_DIM = 6


def _make_equicorr(dim: int, rho: float) -> np.ndarray:
    """Return an equicorrelation matrix compatible with the requested dim."""

    lower = -1.0 / (dim - 1)
    if not (lower < rho < 0.999999):
        raise ValueError(
            "Equicorrelation rho must lie in (-1/(d-1), 0.999999)."
        )
    corr = np.full((dim, dim), rho, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    return corr


def _render_preset_controls(dim: int) -> Tuple[str, Mapping[str, float]]:
    """Render preset parameter controls and return the selection."""

    family = st.selectbox(
        "Preset copula family",
        options=(
            "Gaussian",
            "Student t",
            "Clayton",
            "Gumbel",
            "Frank",
            "Joe",
            "AMH",
        ),
    )
    params: Dict[str, float] = {}
    if family in {"Gaussian", "Student t"}:
        rho = st.slider(
            "Equicorrelation parameter (rho)",
            min_value=float(max(-0.95, -1.0 / (dim - 1) + 1e-3)),
            max_value=0.95,
            value=0.4,
            step=0.01,
        )
        params["rho"] = float(rho)
        if family == "Student t":
            nu = st.slider(
                "Degrees of freedom (nu)",
                min_value=2.2,
                max_value=30.0,
                value=6.0,
                step=0.1,
            )
            params["nu"] = float(nu)
    elif family == "Clayton":
        theta = st.slider(
            "Dependence parameter (theta)",
            min_value=0.2,
            max_value=10.0,
            value=1.5,
            step=0.1,
        )
        params["theta"] = float(theta)
    elif family == "Gumbel":
        theta = st.slider(
            "Dependence parameter (theta)",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
        )
        params["theta"] = float(theta)
    elif family == "Frank":
        theta = st.slider(
            "Dependence parameter (theta)",
            min_value=-20.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
        )
        params["theta"] = float(theta)
    elif family == "Joe":
        theta = st.slider(
            "Dependence parameter (theta)",
            min_value=1.0,
            max_value=10.0,
            value=2.5,
            step=0.1,
        )
        params["theta"] = float(theta)
    else:  # AMH
        st.info("AMH copula supports dimension 2 only.")
        params["theta"] = float(
            st.slider(
                "Dependence parameter (theta)",
                min_value=-0.95,
                max_value=0.95,
                value=0.3,
                step=0.01,
            )
        )
    return family, params


def _sample_preset(
    family: str,
    dim: int,
    params: Mapping[str, float],
    n: int,
    seed: int | None,
) -> np.ndarray:
    """Generate samples from a preset copula selection."""

    if family == "Gaussian":
        corr = _make_equicorr(dim, params["rho"])
        return GaussianCopula(corr=corr).rvs(n, seed=seed)
    if family == "Student t":
        corr = _make_equicorr(dim, params["rho"])
        return StudentTCopula(corr=corr, nu=params["nu"]).rvs(n, seed=seed)
    if family == "Clayton":
        return ClaytonCopula(theta=params["theta"], dim=dim).rvs(n, seed=seed)
    if family == "Gumbel":
        return GumbelCopula(theta=params["theta"], dim=dim).rvs(n, seed=seed)
    if family == "Frank":
        return FrankCopula(theta=params["theta"], dim=dim).rvs(n, seed=seed)
    if family == "Joe":
        return JoeCopula(theta=params["theta"], dim=dim).rvs(n, seed=seed)
    if dim != 2:
        raise ValueError("AMH copula only supports dimension two.")
    return AMHCopula(theta=params["theta"], dim=2).rvs(n, seed=seed)


class ExpressionValidator(ast.NodeVisitor):
    """AST visitor that enforces a whitelist of safe expressions."""

    _allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.Load,
        ast.Name,
        ast.Constant,
        ast.Compare,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.Eq,
        ast.NotEq,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.IfExp,
    )

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self._allowed_nodes):
            raise ValueError(
                "Unsupported expression element: "
                f"{type(node).__name__}"
            )
        super().generic_visit(node)


def _compile_expression(
    expr: str, variables: Tuple[str, ...]
) -> Callable[[np.ndarray], np.ndarray]:
    """Compile a safe NumPy-ready expression for sandbox evaluation."""

    tree = ast.parse(expr, mode="eval")
    ExpressionValidator().visit(tree)
    compiled = compile(tree, filename="<sandbox>", mode="eval")
    namespace: Dict[str, object] = {
        "np": np,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "arctan": np.arctan,
        "abs": np.abs,
        "pi": np.pi,
        "power": np.power,
        "minimum": np.minimum,
        "maximum": np.maximum,
    }

    def evaluator(points: np.ndarray) -> np.ndarray:
        local: Dict[str, np.ndarray] = {
            variables[i]: points[:, i] for i in range(points.shape[1])
        }
        local.update(namespace)
        result = eval(compiled, {"__builtins__": {}}, local)
        array = np.asarray(result, dtype=np.float64)
        if array.ndim == 0:
            return np.full(points.shape[0], float(array), dtype=np.float64)
        if array.shape[0] != points.shape[0]:
            raise ValueError(
                "Expression must return a vector matching the sample size."
            )
        return array

    return evaluator


def _sample_custom_density(
    density_fn: Callable[[np.ndarray], np.ndarray],
    dim: int,
    n: int,
    seed: int | None,
) -> np.ndarray:
    """Sample points on (0, 1)^d using importance resampling."""

    rng = np.random.default_rng(seed)
    pool = max(10 * n, 5000)
    candidates = rng.uniform(_CLIP, 1.0 - _CLIP, size=(pool, dim))
    weights = density_fn(candidates)
    finite = np.isfinite(weights)
    weights = np.clip(weights, a_min=0.0, a_max=None)
    weights = np.where(finite, weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(
            "Density expression must evaluate to positive values."
        )
    probs = weights / total
    indices = rng.choice(pool, size=n, replace=True, p=probs)
    return candidates[indices]


def _summarise_samples(samples: np.ndarray) -> pd.DataFrame:
    """Return summary statistics for the generated sample."""

    df = pd.DataFrame(
        samples,
        columns=[f"u{i + 1}" for i in range(samples.shape[1])],
    )
    summary = df.describe(percentiles=[0.1, 0.5, 0.9]).T
    summary = summary.rename(columns={
        "50%": "median",
        "10%": "p10",
        "90%": "p90",
    })
    return summary[["mean", "std", "min", "p10", "median", "p90", "max"]]


def _plot_samples(samples: np.ndarray, title: str) -> None:
    """Display visual diagnostics for sandbox samples."""

    dim = samples.shape[1]
    df = pd.DataFrame(samples, columns=[f"u{i + 1}" for i in range(dim)])
    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.kdeplot(
            data=df,
            x="u1",
            y="u2",
            fill=True,
            cmap="magma",
            thresh=0.01,
            levels=40,
            ax=ax,
        )
        ax.scatter(df["u1"], df["u2"], s=8, alpha=0.4, color="cyan")
        ax.set_xlabel("u₁")
        ax.set_ylabel("u₂")
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        st.pyplot(fig, clear_figure=True)
        return
    if dim <= _PAIRPLOT_MAX_DIM:
        grid = sns.pairplot(df.sample(min(len(df), 2000)), corner=True)
        grid.fig.suptitle(title, y=1.02)
        st.pyplot(grid.fig, clear_figure=True)
        return
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, ax=ax, cmap="magma", vmin=-1.0, vmax=1.0)
    ax.set_title(f"{title} — correlation heatmap")
    st.pyplot(fig, clear_figure=True)


def _render_custom_builder(dim: int) -> Tuple[str, str]:
    """Collect custom C(u) and c(u) expressions from the user."""

    variable_names = [f"u{i + 1}" for i in range(dim)]
    default_cdf = " * ".join(variable_names)
    cdf_expr = st.text_area(
        "Custom copula C(u)",
        value=default_cdf,
        help=(
            "Enter a NumPy-style expression using variables u1, u2, ... up to"
            " the selected dimension."
        ),
    )
    density_expr = st.text_area(
        "Custom copula density c(u)",
        value="1 + 0.5*sin(2*pi*(u1 + u2))",
        help="Provide a positive expression to define the sampling weights.",
    )
    st.caption(
        "Expressions may reference numpy operations (exp, log, sqrt, sin, cos)"
        " and constants such as pi."
    )
    return cdf_expr.strip(), density_expr.strip()


def _preview_custom_functions(
    cdf_expr: str,
    density_expr: str,
    dim: int,
    density_fn: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Visualise the declared C(u) and c(u) for diagnostic purposes."""

    if dim != 2:
        st.info(
            "Preview plots for custom expressions are available in 2D only."
        )
        return
    grid = np.linspace(_CLIP, 1.0 - _CLIP, 80)
    u1, u2 = np.meshgrid(grid, grid, indexing="xy")
    flat = np.column_stack([u1.ravel(), u2.ravel()])
    density = density_fn(flat).reshape(u1.shape)
    try:
        cdf_fn = _compile_expression(cdf_expr, ("u1", "u2"))
        cdf_vals = cdf_fn(flat).reshape(u1.shape)
    except Exception as exc:
        st.warning(f"Failed to evaluate C(u): {exc}")
        cdf_vals = None

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    cs = ax[0].contourf(u1, u2, density, cmap="magma", levels=40)
    ax[0].set_title("Declared density c(u)")
    ax[0].set_xlabel("u₁")
    ax[0].set_ylabel("u₂")
    fig.colorbar(cs, ax=ax[0])

    if cdf_vals is not None:
        cs2 = ax[1].contourf(u1, u2, cdf_vals, cmap="magma", levels=40)
        ax[1].set_title("Declared copula C(u)")
        ax[1].set_xlabel("u₁")
        ax[1].set_ylabel("u₂")
        fig.colorbar(cs2, ax=ax[1])
    else:
        ax[1].axis("off")
    st.pyplot(fig, clear_figure=True)


st.title("Sandbox")
st.write(
    "Experiment with synthetic copula data: choose a preset family or define"
    " custom Sklar functions, generate pseudo-observations of any dimension,"
    " and export the resulting sample."
)

with st.sidebar:
    st.header("Configuration")
    generation_mode = st.radio(
        "Generation mode",
        options=("Preset copula", "Custom expressions"),
    )
    dim = int(
        st.number_input(
            "Dimension",
            min_value=2,
            max_value=10,
            value=_DEFAULT_DIMENSION,
        )
    )
    n_samples = int(
        st.number_input(
            "Sample size",
            min_value=100,
            max_value=50000,
            value=_DEFAULT_SAMPLE_SIZE,
            step=100,
        )
    )
    seed_input = st.number_input(
        "Random seed (optional)",
        min_value=0,
        max_value=1_000_000,
        value=0,
        step=1,
    )
    seed = int(seed_input) if seed_input > 0 else None

st.divider()

samples: np.ndarray | None = None
metadata: Dict[str, str] = {}

try:
    if generation_mode == "Preset copula":
        family, preset_params = _render_preset_controls(dim)
        if family == "AMH" and dim != 2:
            st.error("AMH copula requires dimension 2.")
        else:
            samples = _sample_preset(
                family=family,
                dim=dim,
                params=preset_params,
                n=n_samples,
                seed=seed,
            )
            metadata = {
                "family": family,
                **{k: f"{v:.4f}" for k, v in preset_params.items()},
            }
    else:
        variable_names = tuple(f"u{i + 1}" for i in range(dim))
        c_expr, d_expr = _render_custom_builder(dim)
        density_fn = _compile_expression(d_expr, variable_names)
        _preview_custom_functions(c_expr, d_expr, dim, density_fn)
        samples = _sample_custom_density(
            density_fn=density_fn,
            dim=dim,
            n=n_samples,
            seed=seed,
        )
        metadata = {"family": "Custom", "density": d_expr[:60]}
except Exception as exc:
    st.error(f"Generation failed: {exc}")

if samples is not None:
    st.subheader("Sample overview")
    df = pd.DataFrame(
        samples,
        columns=[f"u{i + 1}" for i in range(samples.shape[1])],
    )
    st.write("First rows of the generated pseudo-observations:")
    st.dataframe(df.head(20))

    st.write("Summary statistics across dimensions:")
    st.dataframe(_summarise_samples(samples))

    _plot_samples(samples, title="Sandbox sample diagnostics")

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    st.download_button(
        label="Download sample as CSV",
        data=csv_bytes,
        file_name="copula_sandbox_samples.csv",
        mime="text/csv",
    )

    st.caption(
        "Sampling assumes unit hypercube support. For custom densities,"
        " weights are normalised automatically via importance resampling."
    )
