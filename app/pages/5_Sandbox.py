"""Interactive sandbox for building and sampling copulas."""

from __future__ import annotations

import ast
import io
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

try:  # pragma: no cover - seaborn optional in lean environments
    import seaborn as _seaborn  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - diagnostics degrade gracefully
    _seaborn = None
import streamlit as st
from scipy import stats  # type: ignore[import-untyped]

from src.models.copulas.archimedean import (
    AMHCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from src.models.copulas.gaussian import GaussianCopula
from src.models.copulas.student_t import StudentTCopula

sns = cast(Any, _seaborn)

_CLIP = 1e-6
_DEFAULT_SAMPLE_SIZE = 2000
_DEFAULT_DIMENSION = 2
_PAIRPLOT_MAX_DIM = 6


Array = NDArray[np.float64]
ParamDict = Dict[str, float | Array]


@dataclass(frozen=True)
class MarginalParameter:
    """Specification for a scalar marginal parameter."""

    name: str
    label: str
    default: float
    min_value: float
    max_value: float
    step: float


@dataclass
class MarginalSelection:
    """User-selected marginal distribution and its parameters."""

    name: str
    params: Dict[str, float]


MARGINAL_LIBRARY: Dict[str, Tuple[MarginalParameter, ...]] = {
    "Uniform (0, 1)": tuple(),
    "Normal": (
        MarginalParameter("mean", "Mean", 0.0, -5.0, 5.0, 0.1),
        MarginalParameter("std", "Standard deviation", 1.0, 0.1, 5.0, 0.1),
    ),
    "Student t": (
        MarginalParameter("df", "Degrees of freedom", 5.0, 2.0, 30.0, 0.1),
        MarginalParameter("loc", "Location", 0.0, -5.0, 5.0, 0.1),
        MarginalParameter("scale", "Scale", 1.0, 0.1, 5.0, 0.1),
    ),
    "Lognormal": (
        MarginalParameter("mean", "Log-mean", 0.0, -2.0, 2.0, 0.05),
        MarginalParameter(
            "sigma",
            "Log-standard deviation",
            0.25,
            0.05,
            1.5,
            0.05,
        ),
    ),
    "Exponential": (MarginalParameter("rate", "Rate", 1.0, 0.05, 5.0, 0.05),),
    "Gamma": (
        MarginalParameter("shape", "Shape", 2.0, 0.2, 10.0, 0.1),
        MarginalParameter("scale", "Scale", 1.0, 0.1, 5.0, 0.1),
    ),
    "Beta": (
        MarginalParameter("alpha", "Alpha", 2.0, 0.2, 10.0, 0.1),
        MarginalParameter("beta", "Beta", 2.0, 0.2, 10.0, 0.1),
    ),
}

MARGINAL_OPTIONS = tuple(MARGINAL_LIBRARY.keys())


def _default_corr_frame(dim: int) -> pd.DataFrame:
    """Return an identity matrix formatted for editing."""

    values = np.eye(dim, dtype=np.float64)
    labels = [f"u{i + 1}" for i in range(dim)]
    return pd.DataFrame(values, columns=labels, index=labels)


def _render_corr_matrix(dim: int) -> Array:
    """Display and validate a user-edited correlation matrix."""

    session_key = f"sandbox_corr_default_{dim}"
    default = st.session_state.get(session_key)
    if not isinstance(default, pd.DataFrame) or default.shape != (dim, dim):
        default = _default_corr_frame(dim)
    st.caption(
        "Edit the correlation matrix (values in [-0.99, 0.99], "
        "diagonal fixed)."
    )
    editor_key = f"corr_editor_{dim}"
    edited = st.data_editor(
        default,
        key=editor_key,
        num_rows="fixed",
        use_container_width=True,
    )
    st.session_state[session_key] = edited
    corr = edited.to_numpy(dtype=np.float64)
    if corr.shape != (dim, dim):
        raise ValueError("Correlation matrix shape is invalid.")
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    off_diag = corr - np.eye(dim)
    if np.any(np.abs(off_diag) >= 0.999):
        raise ValueError("Correlation magnitudes must be below 0.999.")
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(corr)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Correlation matrix must be positive definite.")
    return np.asarray(corr, dtype=np.float64)


def _render_preset_controls(dim: int) -> Tuple[str, ParamDict]:
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
    params: ParamDict = {}
    if family in {"Gaussian", "Student t"}:
        corr = _render_corr_matrix(dim)
        params["corr"] = corr
        if family == "Student t":
            nu = st.slider(
                "Degrees of freedom (nu)",
                min_value=2.0,
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


def _require_scalar_theta(
    params: Mapping[str, float | Array], family: str
) -> float:
    """Extract a scalar dependence parameter from the selection."""

    value = params.get("theta")
    if value is None or isinstance(value, np.ndarray):
        raise ValueError(
            f"{family} copula requires a scalar dependence parameter."
        )
    return float(value)


def _sample_preset(
    family: str,
    dim: int,
    params: Mapping[str, float | Array],
    n: int,
    seed: int | None,
) -> Array:
    """Generate samples from a preset copula selection."""

    if family == "Gaussian":
        corr_param = params.get("corr")
        if not isinstance(corr_param, np.ndarray):
            raise ValueError("Gaussian copula requires a correlation matrix.")
        samples = GaussianCopula(corr=corr_param).rvs(
            n,
            seed=seed,
        )
        return np.asarray(samples, dtype=np.float64)
    if family == "Student t":
        corr_param = params.get("corr")
        nu_param = params.get("nu")
        if not isinstance(corr_param, np.ndarray) or nu_param is None:
            raise ValueError(
                "Student t copula requires correlation matrix and nu."
            )
        samples = StudentTCopula(
            corr=corr_param,
            nu=float(nu_param),
        ).rvs(
            n,
            seed=seed,
        )
        return np.asarray(samples, dtype=np.float64)
    if family == "Clayton":
        theta = _require_scalar_theta(params, family)
        samples = ClaytonCopula(theta=theta, dim=dim).rvs(
            n,
            seed=seed,
        )
        return np.asarray(samples, dtype=np.float64)
    if family == "Gumbel":
        theta = _require_scalar_theta(params, family)
        samples = GumbelCopula(theta=theta, dim=dim).rvs(
            n,
            seed=seed,
        )
        return np.asarray(samples, dtype=np.float64)
    if family == "Frank":
        theta = _require_scalar_theta(params, family)
        samples = FrankCopula(theta=theta, dim=dim).rvs(
            n,
            seed=seed,
        )
        return np.asarray(samples, dtype=np.float64)
    if family == "Joe":
        theta = _require_scalar_theta(params, family)
        samples = JoeCopula(theta=theta, dim=dim).rvs(
            n,
            seed=seed,
        )
        return np.asarray(samples, dtype=np.float64)
    if dim != 2:
        raise ValueError("AMH copula only supports dimension two.")
    theta = _require_scalar_theta(params, "AMH")
    samples = AMHCopula(theta=theta, dim=2).rvs(
        n,
        seed=seed,
    )
    return np.asarray(samples, dtype=np.float64)


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
                "Unsupported expression element: " f"{type(node).__name__}"
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
        local: Dict[str, object] = {
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
) -> Array:
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
    return np.asarray(candidates[indices], dtype=np.float64)


def _render_marginal_controls(dim: int) -> Tuple[MarginalSelection, ...]:
    """Collect marginal distribution choices from the user."""

    st.subheader("Marginal distributions")
    st.caption(
        "Choose an output distribution for each variable. Uniform margins "
        "leave the pseudo-observations unchanged."
    )
    selections: List[MarginalSelection] = []
    for idx in range(dim):
        label = f"Variable X{idx + 1}"
        expanded = idx < 2
        with st.expander(label, expanded=expanded):
            option_key = f"marginal_name_{dim}_{idx}"
            name = st.selectbox(
                "Distribution",
                options=MARGINAL_OPTIONS,
                key=option_key,
            )
            params: Dict[str, float] = {}
            for parameter in MARGINAL_LIBRARY[name]:
                field_key = f"marginal_param_{dim}_{idx}_{parameter.name}"
                value = st.number_input(
                    parameter.label,
                    min_value=parameter.min_value,
                    max_value=parameter.max_value,
                    value=parameter.default,
                    step=parameter.step,
                    key=field_key,
                )
                params[parameter.name] = float(value)
            selections.append(MarginalSelection(name=name, params=params))
    return tuple(selections)


def _build_margin_transform(
    name: str, params: Mapping[str, float]
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a transformation that maps U to the chosen marginal."""

    def clip(values: np.ndarray) -> np.ndarray:
        return np.clip(values, _CLIP, 1.0 - _CLIP)

    if name == "Uniform (0, 1)":
        return lambda u: np.asarray(u, dtype=np.float64)
    if name == "Normal":
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        if std <= 0.0:
            raise ValueError("Normal standard deviation must be positive.")
        dist = stats.norm(loc=mean, scale=std)

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    if name == "Student t":
        df = params.get("df", 5.0)
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        if df <= 2.0:
            raise ValueError("Student t degrees of freedom must exceed 2.")
        if scale <= 0.0:
            raise ValueError("Student t scale must be positive.")
        dist = stats.t(df, loc=loc, scale=scale)

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    if name == "Lognormal":
        mean = params.get("mean", 0.0)
        sigma = params.get("sigma", 0.25)
        if sigma <= 0.0:
            raise ValueError("Lognormal sigma must be positive.")
        dist = stats.lognorm(s=sigma, scale=np.exp(mean))

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    if name == "Exponential":
        rate = params.get("rate", 1.0)
        if rate <= 0.0:
            raise ValueError("Exponential rate must be positive.")
        dist = stats.expon(scale=1.0 / rate)

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    if name == "Gamma":
        shape = params.get("shape", 2.0)
        scale = params.get("scale", 1.0)
        if shape <= 0.0 or scale <= 0.0:
            raise ValueError("Gamma shape and scale must be positive.")
        dist = stats.gamma(a=shape, scale=scale)

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    if name == "Beta":
        alpha = params.get("alpha", 2.0)
        beta = params.get("beta", 2.0)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("Beta parameters must be positive.")
        dist = stats.beta(a=alpha, b=beta)

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    raise ValueError(f"Unsupported marginal distribution: {name}")


def _apply_marginals(
    samples: np.ndarray, selections: Sequence[MarginalSelection]
) -> np.ndarray:
    """Map pseudo-observations to user-selected margins."""

    dim = samples.shape[1]
    if len(selections) != dim:
        raise ValueError(
            "Marginal selections must match the sample dimension."
        )
    transformed = np.empty_like(samples, dtype=np.float64)
    for idx, selection in enumerate(selections):
        transform = _build_margin_transform(selection.name, selection.params)
        transformed[:, idx] = transform(samples[:, idx])
    return transformed


def _summarise_marginals(
    selections: Sequence[MarginalSelection],
) -> Tuple[str, ...]:
    """Format marginal configuration for display."""

    formatted: List[str] = []
    for idx, selection in enumerate(selections):
        if selection.params:
            params = ", ".join(
                f"{key}={value:.3g}" for key, value in selection.params.items()
            )
            formatted.append(f"X{idx + 1}: {selection.name} ({params})")
        else:
            formatted.append(f"X{idx + 1}: {selection.name}")
    return tuple(formatted)


def _summarise_samples(samples: np.ndarray, prefix: str) -> pd.DataFrame:
    """Return summary statistics for the generated sample."""

    df = pd.DataFrame(
        samples,
        columns=[f"{prefix}{i + 1}" for i in range(samples.shape[1])],
    )
    summary = df.describe(percentiles=[0.1, 0.5, 0.9]).T
    summary = summary.rename(
        columns={
            "50%": "median",
            "10%": "p10",
            "90%": "p90",
        }
    )
    return summary[["mean", "std", "min", "p10", "median", "p90", "max"]]


def _plot_samples(
    samples: np.ndarray,
    title: str,
    unit_cube: bool,
    prefix: str,
) -> None:
    """Display visual diagnostics for sandbox samples."""

    dim = samples.shape[1]
    df = pd.DataFrame(
        samples,
        columns=[f"{prefix}{i + 1}" for i in range(dim)],
    )
    if dim == 2:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        if sns is None:
            hex_map = ax.hexbin(
                df[f"{prefix}1"],
                df[f"{prefix}2"],
                gridsize=60,
                cmap="magma",
                extent=(0.0, 1.0, 0.0, 1.0) if unit_cube else None,
            )
            fig.colorbar(hex_map, ax=ax, label="Density")
        else:
            sns.kdeplot(
                data=df,
                x=f"{prefix}1",
                y=f"{prefix}2",
                fill=True,
                cmap="magma",
                thresh=0.01,
                levels=40,
                ax=ax,
            )
        ax.scatter(
            df[f"{prefix}1"],
            df[f"{prefix}2"],
            s=8,
            alpha=0.4,
            color="cyan",
        )
        ax.set_xlabel(f"{prefix}₁")
        ax.set_ylabel(f"{prefix}₂")
        ax.set_title(title)
        if unit_cube:
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        return
    if dim <= _PAIRPLOT_MAX_DIM:
        sample = df.sample(min(len(df), 2000))
        if sns is None:
            axes = pd.plotting.scatter_matrix(sample, figsize=(6.0, 6.0))
            plt.suptitle(title, y=0.95)
            fig = axes[0, 0].get_figure()
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            grid = sns.pairplot(sample, corner=True)
            grid.fig.set_size_inches(6.0, 6.0)
            grid.fig.suptitle(title, y=1.02)
            st.pyplot(grid.fig, clear_figure=True, use_container_width=True)
        return
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if sns is None:
        im = ax.imshow(corr, cmap="magma", vmin=-1.0, vmax=1.0)
        fig.colorbar(im, ax=ax, label="Correlation")
    else:
        sns.heatmap(corr, ax=ax, cmap="magma", vmin=-1.0, vmax=1.0)
    ax.set_title(f"{title} — correlation heatmap")
    st.pyplot(fig, clear_figure=True, use_container_width=True)


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

    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.6))
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
    st.pyplot(fig, clear_figure=True, use_container_width=True)


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


ParamDict = Dict[str, float | np.ndarray]


@dataclass(frozen=True)
class MarginalParameter:
    """Specification for a scalar marginal parameter."""

    name: str
    label: str
    default: float
    min_value: float
    max_value: float
    step: float


@dataclass
class MarginalSelection:
    """User-selected marginal distribution and its parameters."""

    name: str
    params: Dict[str, float]


MARGINAL_LIBRARY: Dict[str, Tuple[MarginalParameter, ...]] = {
    "Uniform (0, 1)": tuple(),
    "Normal": (
        MarginalParameter("mean", "Mean", 0.0, -5.0, 5.0, 0.1),
        MarginalParameter("std", "Standard deviation", 1.0, 0.1, 5.0, 0.1),
    ),
    "Student t": (
        MarginalParameter("df", "Degrees of freedom", 5.0, 2.0, 30.0, 0.1),
        MarginalParameter("loc", "Location", 0.0, -5.0, 5.0, 0.1),
        MarginalParameter("scale", "Scale", 1.0, 0.1, 5.0, 0.1),
    ),
    "Lognormal": (
        MarginalParameter("mean", "Log-mean", 0.0, -2.0, 2.0, 0.05),
        MarginalParameter(
            "sigma",
            "Log-standard deviation",
            0.25,
            0.05,
            1.5,
            0.05,
        ),
    ),
    "Exponential": (
        MarginalParameter("rate", "Rate", 1.0, 0.05, 5.0, 0.05),
    ),
    "Gamma": (
        MarginalParameter("shape", "Shape", 2.0, 0.2, 10.0, 0.1),
        MarginalParameter("scale", "Scale", 1.0, 0.1, 5.0, 0.1),
    ),
    "Beta": (
        MarginalParameter("alpha", "Alpha", 2.0, 0.2, 10.0, 0.1),
        MarginalParameter("beta", "Beta", 2.0, 0.2, 10.0, 0.1),
    ),
}

MARGINAL_OPTIONS = tuple(MARGINAL_LIBRARY.keys())


def _default_corr_frame(dim: int) -> pd.DataFrame:
    """Return an identity matrix formatted for editing."""

    values = np.eye(dim, dtype=np.float64)
    labels = [f"u{i + 1}" for i in range(dim)]
    return pd.DataFrame(values, columns=labels, index=labels)


def _render_corr_matrix(dim: int) -> np.ndarray:
    """Display and validate a user-edited correlation matrix."""

    session_key = f"sandbox_corr_default_{dim}"
    default = st.session_state.get(session_key)
    if not isinstance(default, pd.DataFrame) or default.shape != (dim, dim):
        default = _default_corr_frame(dim)
    st.caption(
        "Edit the correlation matrix (values in [-0.99, 0.99], "
        "diagonal fixed)."
    )
    editor_key = f"corr_editor_{dim}"
    edited = st.data_editor(
        default,
        key=editor_key,
        num_rows="fixed",
        use_container_width=True,
    )
    st.session_state[session_key] = edited
    corr = edited.to_numpy(dtype=np.float64)
    if corr.shape != (dim, dim):
        raise ValueError("Correlation matrix shape is invalid.")
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    off_diag = corr - np.eye(dim)
    if np.any(np.abs(off_diag) >= 0.999):
        raise ValueError("Correlation magnitudes must be below 0.999.")
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(corr)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Correlation matrix must be positive definite.")
    return corr


def _render_preset_controls(dim: int) -> Tuple[str, ParamDict]:
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
    params: ParamDict = {}
    if family in {"Gaussian", "Student t"}:
        corr = _render_corr_matrix(dim)
        params["corr"] = corr
        if family == "Student t":
            nu = st.slider(
                "Degrees of freedom (nu)",
                min_value=2.0,
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
    params: Mapping[str, float | np.ndarray],
    n: int,
    seed: int | None,
) -> np.ndarray:
    """Generate samples from a preset copula selection."""

    if family == "Gaussian":
        corr_param = params.get("corr")
        if not isinstance(corr_param, np.ndarray):
            raise ValueError("Gaussian copula requires a correlation matrix.")
        return GaussianCopula(corr=corr_param).rvs(n, seed=seed)
    if family == "Student t":
        corr_param = params.get("corr")
        nu_param = params.get("nu")
        if not isinstance(corr_param, np.ndarray) or nu_param is None:
            raise ValueError(
                "Student t copula requires correlation matrix and nu."
            )
        return StudentTCopula(corr=corr_param, nu=float(nu_param)).rvs(
            n,
            seed=seed,
        )
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


def _render_marginal_controls(dim: int) -> Tuple[MarginalSelection, ...]:
    """Collect marginal distribution choices from the user."""

    st.subheader("Marginal distributions")
    st.caption(
        "Choose an output distribution for each variable. Uniform margins "
        "leave the pseudo-observations unchanged."
    )
    selections: List[MarginalSelection] = []
    for idx in range(dim):
        label = f"Variable X{idx + 1}"
        expanded = idx < 2
        with st.expander(label, expanded=expanded):
            option_key = f"marginal_name_{dim}_{idx}"
            name = st.selectbox(
                "Distribution",
                options=MARGINAL_OPTIONS,
                key=option_key,
            )
            params: Dict[str, float] = {}
            for parameter in MARGINAL_LIBRARY[name]:
                field_key = f"marginal_param_{dim}_{idx}_{parameter.name}"
                value = st.number_input(
                    parameter.label,
                    min_value=parameter.min_value,
                    max_value=parameter.max_value,
                    value=parameter.default,
                    step=parameter.step,
                    key=field_key,
                )
                params[parameter.name] = float(value)
            selections.append(MarginalSelection(name=name, params=params))
    return tuple(selections)


def _build_margin_transform(
    name: str, params: Mapping[str, float]
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a transformation that maps U to the chosen marginal."""

    def clip(values: np.ndarray) -> np.ndarray:
        return np.clip(values, _CLIP, 1.0 - _CLIP)

    if name == "Uniform (0, 1)":
        return lambda u: np.asarray(u, dtype=np.float64)
    if name == "Normal":
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        if std <= 0.0:
            raise ValueError("Normal standard deviation must be positive.")
        dist = stats.norm(loc=mean, scale=std)

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    if name == "Student t":
        df = params.get("df", 5.0)
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        if df <= 2.0:
            raise ValueError("Student t degrees of freedom must exceed 2.")
        if scale <= 0.0:
            raise ValueError("Student t scale must be positive.")
        dist = stats.t(df, loc=loc, scale=scale)

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    if name == "Lognormal":
        mean = params.get("mean", 0.0)
        sigma = params.get("sigma", 0.25)
        if sigma <= 0.0:
            raise ValueError("Lognormal sigma must be positive.")
        dist = stats.lognorm(s=sigma, scale=np.exp(mean))

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    if name == "Exponential":
        rate = params.get("rate", 1.0)
        if rate <= 0.0:
            raise ValueError("Exponential rate must be positive.")
        dist = stats.expon(scale=1.0 / rate)

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    if name == "Gamma":
        shape = params.get("shape", 2.0)
        scale = params.get("scale", 1.0)
        if shape <= 0.0 or scale <= 0.0:
            raise ValueError("Gamma shape and scale must be positive.")
        dist = stats.gamma(a=shape, scale=scale)

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    if name == "Beta":
        alpha = params.get("alpha", 2.0)
        beta = params.get("beta", 2.0)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("Beta parameters must be positive.")
        dist = stats.beta(a=alpha, b=beta)

        def transform(u: np.ndarray) -> np.ndarray:
            return np.asarray(dist.ppf(clip(u)), dtype=np.float64)

        return transform
    raise ValueError(f"Unsupported marginal distribution: {name}")


def _apply_marginals(
    samples: np.ndarray, selections: Sequence[MarginalSelection]
) -> np.ndarray:
    """Map pseudo-observations to user-selected margins."""

    dim = samples.shape[1]
    if len(selections) != dim:
        raise ValueError(
            "Marginal selections must match the sample dimension."
        )
    transformed = np.empty_like(samples, dtype=np.float64)
    for idx, selection in enumerate(selections):
        transform = _build_margin_transform(selection.name, selection.params)
        transformed[:, idx] = transform(samples[:, idx])
    return transformed


def _summarise_marginals(
    selections: Sequence[MarginalSelection],
) -> Tuple[str, ...]:
    """Format marginal configuration for display."""

    formatted: List[str] = []
    for idx, selection in enumerate(selections):
        if selection.params:
            params = ", ".join(
                f"{key}={value:.3g}" for key, value in selection.params.items()
            )
            formatted.append(
                f"X{idx + 1}: {selection.name} ({params})"
            )
        else:
            formatted.append(f"X{idx + 1}: {selection.name}")
    return tuple(formatted)


def _summarise_samples(samples: np.ndarray, prefix: str) -> pd.DataFrame:
    """Return summary statistics for the generated sample."""

    df = pd.DataFrame(
        samples,
        columns=[f"{prefix}{i + 1}" for i in range(samples.shape[1])],
    )
    summary = df.describe(percentiles=[0.1, 0.5, 0.9]).T
    summary = summary.rename(columns={
        "50%": "median",
        "10%": "p10",
        "90%": "p90",
    })
    return summary[["mean", "std", "min", "p10", "median", "p90", "max"]]


def _plot_samples(
    samples: np.ndarray,
    title: str,
    unit_cube: bool,
    prefix: str,
) -> None:
    """Display visual diagnostics for sandbox samples."""

    dim = samples.shape[1]
    df = pd.DataFrame(
        samples,
        columns=[f"{prefix}{i + 1}" for i in range(dim)],
    )
    if dim == 2:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.kdeplot(
            data=df,
            x=f"{prefix}1",
            y=f"{prefix}2",
            fill=True,
            cmap="magma",
            thresh=0.01,
            levels=40,
            ax=ax,
        )
        ax.scatter(
            df[f"{prefix}1"],
            df[f"{prefix}2"],
            s=8,
            alpha=0.4,
            color="cyan",
        )
        ax.set_xlabel(f"{prefix}₁")
        ax.set_ylabel(f"{prefix}₂")
        ax.set_title(title)
        if unit_cube:
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        return
    if dim <= _PAIRPLOT_MAX_DIM:
        grid = sns.pairplot(df.sample(min(len(df), 2000)), corner=True)
        grid.fig.set_size_inches(6.0, 6.0)
        grid.fig.suptitle(title, y=1.02)
        st.pyplot(grid.fig, clear_figure=True, use_container_width=True)
        return
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(corr, ax=ax, cmap="magma", vmin=-1.0, vmax=1.0)
    ax.set_title(f"{title} — correlation heatmap")
    st.pyplot(fig, clear_figure=True, use_container_width=True)


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

    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.6))
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
    st.pyplot(fig, clear_figure=True, use_container_width=True)


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

marginal_choices = _render_marginal_controls(dim)

st.divider()

samples: np.ndarray | None = None
transformed: np.ndarray | None = None
metadata: Dict[str, object] = {}

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
                "mode": "Preset copula",
                "dimension": dim,
                "sample_size": n_samples,
                "seed": seed if seed is not None else "random",
            }
            corr_param = preset_params.get("corr")
            if isinstance(corr_param, np.ndarray):
                metadata["corr"] = (
                    np.asarray(
                        corr_param,
                        dtype=np.float64,
                    )
                    .round(3)
                    .tolist()
                )
            for key, value in preset_params.items():
                if key == "corr":
                    continue
                metadata[key] = f"{float(value):.4f}"
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
        metadata = {
            "family": "Custom density",
            "mode": "Custom expressions",
            "dimension": dim,
            "sample_size": n_samples,
            "seed": seed if seed is not None else "random",
            "cdf": c_expr.strip(),
            "density": d_expr.strip(),
        }
    if samples is not None:
        transformed = _apply_marginals(samples, marginal_choices)
except Exception as exc:
    st.error(f"Generation failed: {exc}")

if samples is not None and transformed is not None:
    st.subheader("Sample overview")
    overview = {
        "dimension": dim,
        "sample_size": n_samples,
        "copula": metadata,
        "marginals": _summarise_marginals(marginal_choices),
    }
    st.json(overview)

    df_uniform = pd.DataFrame(
        samples,
        columns=[f"u{i + 1}" for i in range(samples.shape[1])],
    )
    df_transformed = pd.DataFrame(
        transformed,
        columns=[f"x{i + 1}" for i in range(transformed.shape[1])],
    )

    tab_uniform, tab_transformed = st.tabs(
        ["Pseudo-observations", "Transformed data"]
    )

    with tab_uniform:
        st.write("First rows of the generated pseudo-observations:")
        st.dataframe(df_uniform.head(20))
        st.write("Summary statistics across dimensions:")
        st.dataframe(_summarise_samples(samples, prefix="u"))
        _plot_samples(
            samples,
            title="Pseudo-observations diagnostics",
            unit_cube=True,
            prefix="u",
        )
        csv_uniform = io.StringIO()
        df_uniform.to_csv(csv_uniform, index=False)
        st.download_button(
            label="Download pseudo-observations",
            data=csv_uniform.getvalue().encode("utf-8"),
            file_name="copula_pseudo_observations.csv",
            mime="text/csv",
        )

    with tab_transformed:
        st.write("First rows of the transformed sample:")
        st.dataframe(df_transformed.head(20))
        st.write("Summary statistics across dimensions:")
        st.dataframe(_summarise_samples(transformed, prefix="x"))
        _plot_samples(
            transformed,
            title="Transformed sample diagnostics",
            unit_cube=False,
            prefix="x",
        )
        csv_transformed = io.StringIO()
        df_transformed.to_csv(csv_transformed, index=False)
        st.download_button(
            label="Download transformed sample",
            data=csv_transformed.getvalue().encode("utf-8"),
            file_name="copula_transformed_sample.csv",
            mime="text/csv",
        )

    st.caption(
        "Sampling occurs on the unit hypercube; marginal transformations are "
        "applied post hoc using the selected distributions."
    )
