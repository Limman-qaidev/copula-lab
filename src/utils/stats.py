"""Basic statistical utilities with strict typing and logging.

This module demonstrates the required style:
- Google-style docstrings.
- Strict typing for all public functions.
- 79-char line length.
- Logging for errors and key steps.

Assumptions
-----------
- Inputs are finite real numbers.
- No NaNs are allowed (validated by the caller).

Limitations
-----------
- Designed for small-to-medium arrays in memory.

References
----------
- Jorion, P. (2007). Value at Risk. McGraw-Hill.
"""

from __future__ import annotations

import logging
from statistics import mean, pstdev
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["basic_moments"]


def basic_moments(values: List[float]) -> Dict[str, Optional[float]]:
    """Compute basic moments for a numeric series.

    Args:
        values (List[float]): Numeric series. Must be finite and not NaN.

    Returns:
        Dict[str, Optional[float]]: Dictionary with 'mean' and 'std'.
        Returns None for both keys if the input list is empty.

    Raises:
        ValueError: If any element is not finite.

    Notes:
        - Std is population std (denominator n) for reproducibility.
        - Units should be documented by the caller (%, bps, etc.).

    Examples:
        >>> basic_moments([1.0, 2.0, 3.0])
        {'mean': 2.0, 'std': 0.816496580927726}
    """
    if not values:
        logger.warning("Empty series passed to basic_moments.")
        return {"mean": None, "std": None}

    for x in values:
        if not _is_finite(x):
            logger.error("Non-finite value detected: %s", x)
            raise ValueError("All inputs must be finite floats.")

    m = float(mean(values))
    s = float(pstdev(values))
    return {"mean": m, "std": s}


def _is_finite(x: float) -> bool:
    """Return True if x is a finite float (no inf, no NaN)."""
    return not (x == float("inf") or x == float("-inf") or x != x)
