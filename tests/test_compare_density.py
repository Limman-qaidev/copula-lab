"""Tests for the density comparison plot utilities."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from pytest import MonkeyPatch
from typing_extensions import Protocol


class _StreamlitStub:
    def __init__(self) -> None:
        self.calls: List[tuple[str, Any]] = []

    def pyplot(self, fig: object, **kwargs: object) -> None:  # noqa: D401
        """Record pyplot calls for inspection."""

        self.calls.append(("pyplot", (fig, kwargs)))

    def info(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("info", (args, kwargs)))


def _load_plot_function() -> Dict[str, Any]:
    path = Path("app/pages/3_Compare.py")
    module_ast = ast.parse(path.read_text())
    selected: list[ast.stmt] = []
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == "BaseCopula":
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_import_seaborn",
            "plot_density_comparison",
        }:
            selected.append(node)

    st_stub = _StreamlitStub()
    env: Dict[str, Any] = {
        "np": np,
        "plt": plt,
        "importlib": importlib,
        "Protocol": Protocol,
        "Any": Any,
        "st": st_stub,
    }
    exec(  # noqa: S102 - controlled execution for test isolation
        compile(ast.Module(selected, []), str(path), "exec"),
        env,
    )
    env["st_stub"] = st_stub
    return env


class _DummyCopula:
    def __init__(self) -> None:
        self._calls: List[np.ndarray] = []

    def pdf(self, U: np.ndarray) -> np.ndarray:  # noqa: N802 - interface match
        self._calls.append(np.asarray(U))
        return np.ones(U.shape[0], dtype=np.float64)

    def rvs(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.1, 0.9, size=(n, 2))


def test_density_comparison_falls_back_to_hexbin(
    monkeypatch: MonkeyPatch,
) -> None:
    env = _load_plot_function()
    copula = _DummyCopula()
    data = np.full((20, 2), 0.5, dtype=np.float64)

    env["_import_seaborn"] = lambda: None
    recorded: List[object] = []

    st_stub = env["st_stub"]

    def _record_plot(fig: object, **_kwargs: object) -> None:
        recorded.append(fig)

    monkeypatch.setattr(st_stub, "pyplot", _record_plot)
    monkeypatch.setattr(st_stub, "info", lambda *_args, **_kwargs: None)

    env["plot_density_comparison"](
        data,
        copula,
        "fallback density",
        grid_size=10,
    )

    assert recorded, "Expected matplotlib figure to be rendered"
    assert copula._calls, "Expected model pdf to be evaluated"
    grid_eval = copula._calls[0]
    assert grid_eval.shape == (100, 2)
    assert np.all((grid_eval > 0.0) & (grid_eval < 1.0))
