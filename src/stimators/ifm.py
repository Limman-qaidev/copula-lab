"""Backward-compatible re-export of IFM estimators."""

from __future__ import annotations

from src.estimators.ifm import gaussian_ifm  # noqa: F401
from src.estimators.student_t import (  # noqa: F401
    student_t_ifm,
    student_t_pmle,
)

__all__ = ["gaussian_ifm", "student_t_ifm", "student_t_pmle"]
