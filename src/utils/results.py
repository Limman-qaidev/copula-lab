from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


@dataclass(frozen=True)
class FitResult:
    """Minimal structure to persist calibration outcomes."""

    family: str
    params: Mapping[str, float]
    method: str
    loglik: float | None = None
    aic: float | None = None
    bic: float | None = None

    def with_metrics(
        self, *, loglik: float | None, aic: float | None, bic: float | None
    ) -> "FitResult":
        """Return a copy updated with information criteria metrics."""

        return replace(self, loglik=loglik, aic=aic, bic=bic)
