"""Spearman rank correlation, tie-safe."""
from __future__ import annotations

import numpy as np

from ._util import _avg_rank

__all__ = ["spearman"]


def spearman(u, v, min_n: int = 3) -> float:
    """Spearman rank correlation (average ranks for ties). NaN when fewer than
    `min_n` pairs, or when either side is constant."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(u) < min_n:
        return float("nan")
    ru, rv = _avg_rank(u), _avg_rank(v)
    if ru.std() < 1e-12 or rv.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ru, rv)[0, 1])
