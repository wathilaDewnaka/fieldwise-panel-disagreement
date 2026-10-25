"""Calibration metrics for the safety case (ECE / Brier / NLL).

Moved verbatim from backend/training/{metrics}.py — bodies unchanged, plus
temperature scaling (`fit_temperature` / `apply_temperature`), which is fitted on
VALIDATION and applied unchanged to test. Every arm gets the same treatment, so
it rescales the ladder rather than favouring a rung: the comparison stays like
for like and ECE stops measuring "how hard did this arm's auxiliary losses push
the logits" instead of "how well does this arm rank cases".
"""
from __future__ import annotations
import numpy as np
from ._util import _EPS


# --------------------------------------------------------------------------- #
# calibration 
# --------------------------------------------------------------------------- #
def _binary(y_true, y_prob):
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_prob, dtype=float), _EPS, 1 - _EPS)
    return y, p


def brier(y_true, y_prob) -> float:
    """Mean squared error of the probability (binary Brier score)."""
    y, p = _binary(y_true, y_prob)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def nll(y_true, y_prob) -> float:
    """Binary negative log-likelihood (cross-entropy)."""
    y, p = _binary(y_true, y_prob)
    if not len(y):
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y_true, y_prob, n_bins: int = 15) -> float:
    """Expected Calibration Error (confidence of the predicted class)."""
    y, p = _binary(y_true, y_prob)
    if not len(y):
        return float("nan")
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e, n = 0.0, len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.sum():
            e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


# --------------------------------------------------------------------------- #
# temperature scaling - fitted on val, frozen, applied to test
# --------------------------------------------------------------------------- #
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float).ravel(), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def apply_temperature(y_prob, temperature: float) -> np.ndarray:
    """Rescale probabilities by dividing their logits by `temperature`.

    T > 1 softens (the usual direction for an over-confident net), T < 1 sharpens.
    The ORDER of the probabilities is untouched, so AUC and every ranking-based
    selective-prediction metric are invariant to this - only calibration numbers
    (ECE / NLL / Brier) move. That is what makes it a safe thing to apply to an
    ablation: it cannot manufacture a discrimination gap."""
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        return np.clip(np.asarray(y_prob, dtype=float).ravel(), _EPS, 1 - _EPS)
    return 1.0 / (1.0 + np.exp(-_logit(y_prob) / t))


def fit_temperature(y_true, y_prob, lo: float = 0.05, hi: float = 20.0,
                    n_grid: int = 200, n_refine: int = 3) -> float:
    """The temperature minimising validation NLL.

    Coarse-to-fine grid search on log T: the objective is 1-D and convex in
    practice, N is one validation fold, and a deterministic search keeps the fold
    results reproducible without adding an optimiser dependency."""
    y, p = _binary(y_true, y_prob)
    if not len(y) or y.min() == y.max():
        return 1.0
    z = _logit(p)
    best, lo_, hi_ = 1.0, float(lo), float(hi)
    for _ in range(max(1, n_refine)):
        grid = np.exp(np.linspace(np.log(lo_), np.log(hi_), n_grid))
        losses = [nll(y, 1.0 / (1.0 + np.exp(-z / t))) for t in grid]
        i = int(np.argmin(losses))
        best = float(grid[i])
        lo_, hi_ = float(grid[max(0, i - 1)]), float(grid[min(len(grid) - 1, i + 1)])
    return best
