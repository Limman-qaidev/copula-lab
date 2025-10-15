from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def empirical_pit(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert observations into empirical PIT pseudo-observations.

    Args:
        x:
            Array with shape ``(n, 2)`` containing raw observations.

    Returns:
        ``NDArray[np.float64]`` with shape ``(n, 2)`` inside the open unit
        square.

    Raises:
        ValueError: If ``x`` is not two-dimensional, does not have exactly
            two columns, or contains NaNs.
    """

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("La matriz debe tener forma (n, 2).")

    if not np.isfinite(x).all():
        raise ValueError("Los datos contienen valores no finitos o NaN.")

    n_obs = x.shape[0]
    if n_obs == 0:
        raise ValueError("Se requieren observaciones para construir el PIT.")

    ranks = np.empty_like(x, dtype=np.float64)

    for col in range(2):
        values = x[:, col]
        order = np.argsort(values, kind="mergesort")
        sorted_vals = values[order]

        unique_vals, first_idx, counts = np.unique(
            sorted_vals, return_index=True, return_counts=True
        )

        average_ranks = np.empty_like(unique_vals, dtype=np.float64)
        for i, (start, count) in enumerate(zip(first_idx, counts)):
            # Ranks are one-based indices.
            start_rank = float(start + 1)
            end_rank = float(start + count)
            average_ranks[i] = (start_rank + end_rank) / 2.0

        column_ranks = np.empty(n_obs, dtype=np.float64)
        for start, count, avg in zip(first_idx, counts, average_ranks):
            slice_indices = order[slice(start, start + count)]
            column_ranks[slice_indices] = avg

        ranks[:, col] = column_ranks

    return ranks / float(n_obs + 1)
