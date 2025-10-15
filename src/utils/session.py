from __future__ import annotations

from typing import Optional, cast

import numpy as np
import streamlit as st

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
