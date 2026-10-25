"""Paired, fold-level comparisons of one method against another.

A study with k folds has k paired differences per outcome (method minus rival,
same items, same calibration halvings). These are the two questions a triage
comparison asks of them: is the method LOWER (its interval wholly below 0), and
is it NON-INFERIOR (its interval's upper bound below a declared margin)."""
from __future__ import annotations

import numpy as np

__all__ = ["paired_t_interval", "paired_comparison"]


def paired_t_interval(x, y, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi) of x - y over pairs with a Student-t interval on n - 1
    degrees of freedom; NaN pairs are dropped and fewer than 2 give NaNs."""
    from scipy.stats import t as student
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan")
    h = student.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def paired_comparison(x, y, margin: float = 0.0, level: float = 0.95) -> dict:
    """x against y, lower is better: the interval of x - y, whether it is wholly
    below 0 (`lower`), whether its upper bound is below `margin` (`noninferior`),
    and the pairs where x < y (`fewer`) and x <= y (`not_more`)."""
    m, lo, hi = paired_t_interval(x, y, level)
    pairs = [(a, b) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
    return {"diff": m, "lo": lo, "hi": hi, "n": len(pairs),
            "lower": bool(hi < 0), "noninferior": bool(hi < margin),
            "fewer": sum(a < b for a, b in pairs),
            "not_more": sum(a <= b + 1e-9 for a, b in pairs)}
