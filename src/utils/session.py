from __future__ import annotations

from typing import List, Optional, Sequence, cast

import numpy as np
import streamlit as st

from src.utils.results import FitResult
from src.utils.types import FloatArray


def set_seed(seed: int) -> None:
    """Store an integer seed in the Streamlit session."""

    st.session_state["seed"] = int(seed)


def get_seed(default: int = 42) -> int:
    """Retrieve the stored seed or return a default value."""

    return int(st.session_state.get("seed", default))


def set_U(U: FloatArray) -> None:
    """Persist pseudo-observations in the session state."""

    st.session_state["U"] = np.asarray(U, dtype=np.float64)


def has_U() -> bool:
    """Check whether pseudo-observations are available."""

    return "U" in st.session_state


def get_U() -> Optional[FloatArray]:
    """Fetch pseudo-observations from the session state."""

    if "U" not in st.session_state:
        return None
    return cast(FloatArray, st.session_state["U"])


def append_fit_result(result: FitResult) -> None:
    """Append a calibration result to the session history."""

    results = list(get_fit_results())
    results.append(result)
    st.session_state["fit_results"] = results


def get_fit_results() -> Sequence[FitResult]:
    """Retrieve the stored calibration results."""

    if "fit_results" not in st.session_state:
        return []
    raw_results = st.session_state["fit_results"]
    return cast(Sequence[FitResult], raw_results)


def update_fit_result(index: int, result: FitResult) -> None:
    """Replace a calibration result in the session history."""

    results: List[FitResult] = list(get_fit_results())
    if index < 0 or index >= len(results):
        raise IndexError("Result index out of range")
    results[index] = result
    st.session_state["fit_results"] = results


def get_best_model_index() -> Optional[int]:
    """Return the index of the model marked as best, if any."""

    if "best_model_index" not in st.session_state:
        return None
    return int(st.session_state["best_model_index"])


def set_best_model_index(index: Optional[int]) -> None:
    """Persist the preferred model index in the session."""

    if index is None:
        st.session_state.pop("best_model_index", None)
    else:
        st.session_state["best_model_index"] = int(index)
