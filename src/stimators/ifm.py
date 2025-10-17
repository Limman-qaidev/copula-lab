"""Backward-compatible re-export of IFM estimators."""

from __future__ import annotations

from src.estimators.ifm import (  # noqa: F401
    gaussian_ifm,
    gaussian_ifm_corr,
)
from src.estimators.student_t import (  # noqa: F401
    student_t_ifm,
    student_t_pmle,
)

__all__ = [
    "gaussian_ifm",
    "gaussian_ifm_corr",
    "student_t_ifm",
    "student_t_pmle",
]
