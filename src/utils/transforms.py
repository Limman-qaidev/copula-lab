from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def empirical_pit(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert observations into empirical PIT pseudo-observations.

    The function supports datasets with an arbitrary number of columns.
    It computes average ranks for ties and rescales them by ``n + 1`` so the
    output lives in the open hypercube ``(0, 1)^d``.
    """

    if x.ndim != 2:
        raise ValueError("Input data must be two-dimensional.")

    if not np.isfinite(x).all():
        raise ValueError("Input data contains non-finite values or NaN.")

    n_obs, n_dim = x.shape
    if n_obs == 0:
        raise ValueError("At least one observation is required for the PIT.")

    ranks = np.empty_like(x, dtype=np.float64)

    for col in range(n_dim):
        values = x[:, col]
        order = np.argsort(values, kind="mergesort")
        sorted_vals = values[order]

        unique_vals, first_idx, counts = np.unique(
            sorted_vals, return_index=True, return_counts=True
        )

        average_ranks = np.empty_like(unique_vals, dtype=np.float64)
        for idx, (start, count) in enumerate(zip(first_idx, counts)):
            start_rank = float(start + 1)
            end_rank = float(start + count)
            average_ranks[idx] = (start_rank + end_rank) / 2.0

        column_ranks = np.empty(n_obs, dtype=np.float64)
        for start, count, avg in zip(first_idx, counts, average_ranks):
            slice_indices = order[slice(start, start + count)]
            column_ranks[slice_indices] = avg

        ranks[:, col] = column_ranks

    return np.asarray(ranks / float(n_obs + 1), dtype=np.float64)
