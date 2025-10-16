"""Backward-compatible re-export of IFM estimators."""

from __future__ import annotations

from src.estimators.ifm import gaussian_ifm  # noqa: F401

__all__ = ["gaussian_ifm"]
