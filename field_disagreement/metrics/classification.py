"""Discrimination and operating-point metrics.

Moved verbatim from backend/training/{metrics}.py — bodies unchanged.
"""
from __future__ import annotations
import numpy as np
from ._util import _avg_rank


def roc_auc(y_true, y_score) -> float:
    """Mann-Whitney rank AUC; nan if only one class present."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_rank(y_score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def subgroup_metrics(y_true, y_prob, mask, min_n: int = 10) -> dict:
    """AUC / ECE / Brier restricted to a subgroup.

    Returns nans when the subgroup is too small or single-class to support them,
    rather than a number computed from four cases. Pooled metrics are dominated
    by the easy majority, so the interesting comparison is usually inside a
    subgroup - but only when the subgroup is big enough to carry one."""
    from .calibration import brier, ece
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    if m.sum() < min_n or len(np.unique(y[m])) < 2:
        return {"auc": float("nan"), "ece": float("nan"), "brier": float("nan")}
    return {"auc": roc_auc(y[m], p[m]), "ece": ece(y[m], p[m]),
            "brier": brier(y[m], p[m])}
