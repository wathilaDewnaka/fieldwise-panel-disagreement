from __future__ import annotations
import numpy as np


"""
Split-conformal prediction (LAC).

Least-Ambiguous set-valued Classifier (a.k.a. LAC / threshold conformal): wraps
a trained classifier with a distribution-free marginal coverage guarantee on a
held-out calibration split.

  nonconformity   s(x, y) = 1 - p_hat[y | x]
  threshold       q_hat   = Quantile_{ceil((n+1)(1-alpha))/n} ( {s_j}_cal )
  prediction set  C(x)    = { y : p_hat[y | x] >= 1 - q_hat }

Guarantee: Pr[ y in C(x) ] >= 1 - alpha (marginal, exchangeability assumed).

Probabilities are passed as full class distributions (N, K). For the binary
malignancy head use `binary_probs(p)` to build (N, 2) from the malignant prob.
"""


def binary_probs(p) -> np.ndarray:
    """(N,) malignant prob -> (N, 2) class distribution [benign, malignant]."""
    p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
    return np.stack([1.0 - p, p], axis=1)


def lac_calibrate(probs_cal, y_cal, alpha: float = 0.1) -> float:
    """Conformal threshold q_hat from the calibration split."""
    probs = np.asarray(probs_cal, dtype=float)
    y = np.asarray(y_cal, dtype=int)
    n = len(y)
    if n == 0:
        return 1.0
    s = 1.0 - probs[np.arange(n), y]                     # nonconformity scores
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = float(min(level, 1.0))
    return float(np.quantile(s, level, method="higher"))


def lac_predict(probs, q_hat: float) -> np.ndarray:
    """Boolean membership matrix C (N, K): True where class is in the set."""
    probs = np.asarray(probs, dtype=float)
    return probs >= (1.0 - q_hat)


def empirical_coverage(probs, y, q_hat: float) -> float:
    """Fraction of points whose true label falls inside the prediction set."""
    y = np.asarray(y, dtype=int)
    if not len(y):
        return float("nan")
    sets = lac_predict(probs, q_hat)
    return float(sets[np.arange(len(y)), y].mean())


def avg_set_size(probs, q_hat: float) -> float:
    """Average prediction-set size |C(x)| over the points."""
    sets = lac_predict(probs, q_hat)
    return float(sets.sum(axis=1).mean()) if len(sets) else float("nan")


def evaluate(probs_test, y_test, probs_cal, y_cal, alpha: float = 0.1) -> dict:
    """Calibrate on (cal) and report coverage + set size on (test)."""
    q_hat = lac_calibrate(probs_cal, y_cal, alpha)
    return {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "q_hat": q_hat,
        "empirical_coverage": empirical_coverage(probs_test, y_test, q_hat),
        "avg_set_size": avg_set_size(probs_test, q_hat),
        "n_cal": int(len(np.asarray(y_cal))),
        "n_test": int(len(np.asarray(y_test))),
    }
