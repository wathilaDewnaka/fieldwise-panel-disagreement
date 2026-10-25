"""Shared metric helpers.

Moved verbatim from backend/training/{metrics}.py — bodies unchanged.
"""
from __future__ import annotations
import numpy as np


_EPS = 1e-8


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def _avg_rank(x: np.ndarray) -> np.ndarray:
    """1-based average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x), dtype=float)
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks
