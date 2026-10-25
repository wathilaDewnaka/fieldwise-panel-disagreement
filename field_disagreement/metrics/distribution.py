"""Evaluation of a predicted opinion distribution against the panel's own.

A single scalar cannot say whether opinion-distribution learning worked, because
the three ways of being wrong are different failures:

  * EMD  - is the mass in the right PLACE on the ordinal scale? Penalises a
           prediction of rating 1 when the panel said 5 far more than one of 4.
  * KL   - is the whole SHAPE right? Category-blind, and the quantity the `kl`
           training objective actually minimises, so reporting it is what makes
           the kl-vs-emd ablation a fair comparison rather than one arm being
           scored on its own loss.
  * entropy error / correlation - is the predicted amount of DISAGREEMENT right?
           A model can nail the mean rating and still be confidently wrong about
           how divided the readers were, and that is precisely the quantity the
           deferral policy consumes. Getting it wrong is the failure that matters
           clinically, and neither EMD nor KL isolates it.
"""
from __future__ import annotations
import numpy as np
from ._util import _EPS, _avg_rank


def _pair(P, Q) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.ndim == 1:
        P, Q = P[None], Q[None]
    return P, Q


# --------------------------------------------------------------------------- #
# opinion-distribution distance
# --------------------------------------------------------------------------- #
def emd_distance(P, Q) -> float:
    """Mean 1-Wasserstein (EMD) over rating CDFs for a batch of (N, K) dists."""
    P, Q = _pair(P, Q)
    return float(np.mean(np.abs(np.cumsum(P, 1) - np.cumsum(Q, 1)).sum(1)))


def kl_divergence(P, Q) -> float:
    """Mean KL(Q || P) in nats, with `P` the PREDICTION and `Q` the panel's true
    histogram - the same orientation and argument order as `emd_distance` and as
    `triage.opinion_kl_loss(logits, q)`, so the training loss and the reported
    metric are the same quantity."""
    P, Q = _pair(P, Q)
    P = np.clip(P, _EPS, None)
    Q = np.clip(Q, _EPS, None)
    return float(np.mean((Q * (np.log(Q) - np.log(P))).sum(1)))


def distribution_entropy(P, normalize: bool = True) -> np.ndarray:
    """Per-case Shannon entropy of an (N, K) stack of histograms, divided by
    log K when `normalize` so it reads as a fraction of maximum disagreement."""
    P = np.clip(np.asarray(P, dtype=float), _EPS, None)
    if P.ndim == 1:
        P = P[None]
    h = -(P * np.log(P)).sum(axis=1)
    return h / np.log(P.shape[1]) if normalize else h


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    # Degeneracy is "the signal takes one value", which is a question about
    # DISTINCT VALUES, not about spread. Testing `np.std(...) == 0` gets it wrong:
    # the mean of n identical floats is computed as sum/n and can round, so a
    # bit-identical array reports std ~1e-16 and slips through - `prior` opinion
    # baselines do exactly this. corrcoef then divides by ~0, returns the right
    # answer (nan) and emits a RuntimeWarning per fold. Counting distinct values
    # is exact and says what is actually meant.
    if len(a) < 2 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    return float(np.corrcoef(_avg_rank(a), _avg_rank(b))[0, 1])


def entropy_error(P, Q) -> dict:
    """How well the predicted panel spread matches the observed one.

    `mae` is the average absolute gap in normalised entropy; `bias` is the signed
    mean (positive = the model over-states disagreement, which is the safe
    direction for a deferral system, negative = it is over-confident about
    consensus, which is the dangerous one). `corr` is the Spearman rank
    correlation - whether the model at least ORDERS cases by contentiousness
    correctly, which is all the deferral policy needs from this signal."""
    hp = distribution_entropy(P)
    hq = distribution_entropy(Q)
    return {"entropy_mae": float(np.mean(np.abs(hp - hq))),
            "entropy_bias": float(np.mean(hp - hq)),
            "entropy_corr": _spearman(hp, hq)}


# --------------------------------------------------------------------------- #
# what a model WITHOUT an opinion head can produce
#
# The ladder's A_cls arm has no distribution to score, so EMD / KL / entropy
# print as "-" and the opinion columns end up comparing three of our own variants
# against each other with no reference point. That is not an ablation, it is a
# family portrait. These two baselines fix it, and neither needs a retrain:
#
#   prior             the training marginal histogram, same for every case. The
#                     floor: what you get for predicting the dataset and ignoring
#                     the image.
#   class_conditional p * mean histogram of TRAIN malignant panels
#                     + (1-p) * mean histogram of TRAIN benign panels.
#                     The ceiling for a classifier-only system: it uses
#                     everything A_cls knows about the case, optimally decoded.
#
# Beating `class_conditional` is the claim the opinion head has to make, because
# a deployed A_cls could emit exactly this at zero cost. Note what it CANNOT do:
# its entropy is a deterministic function of p, so it can only call a case
# contested when it is also unsure of the class. An opinion head that separates
# those two things should show it in `entropy_corr`, and that gap is the point.
# --------------------------------------------------------------------------- #
def fit_opinion_baselines(q_train, y_train) -> dict:
    """Fit both classifier-only opinion baselines on the TRAINING panels.

    Returns the pieces `predict_opinion_baseline` needs. Fitted on train only, so
    applying them to test is the same protocol the model itself gets."""
    Q = np.asarray(q_train, dtype=float)
    y = np.asarray(y_train).astype(int).ravel()
    if Q.ndim == 1:
        Q = Q[None]
    k = Q.shape[1]
    uniform = np.full(k, 1.0 / k)
    prior = Q.mean(axis=0) if len(Q) else uniform
    pos = Q[y == 1].mean(axis=0) if (y == 1).any() else prior
    neg = Q[y == 0].mean(axis=0) if (y == 0).any() else prior
    return {"prior": prior, "malignant": pos, "benign": neg}


def predict_opinion_baseline(prob, fitted: dict, kind: str = "class_conditional"):
    """(N, K) baseline opinion distributions for a batch of binary probabilities."""
    p = np.asarray(prob, dtype=float).ravel()[:, None]
    if kind == "prior":
        return np.repeat(np.asarray(fitted["prior"], float)[None], len(p), axis=0)
    if kind == "class_conditional":
        return p * np.asarray(fitted["malignant"], float)[None] + \
               (1.0 - p) * np.asarray(fitted["benign"], float)[None]
    raise ValueError(f"unknown opinion baseline {kind!r}")


def predicted_spread(P, ratings=None) -> np.ndarray:
    """s.d. of each predicted rating histogram - `triage.opinion_spread` without
    the torch branch, so metrics do not import the training package."""
    P = np.clip(np.asarray(P, dtype=float), _EPS, None)
    if P.ndim == 1:
        P = P[None]
    P = P / P.sum(axis=-1, keepdims=True)
    r = (np.arange(1, P.shape[-1] + 1, dtype=float) if ratings is None
         else np.asarray(ratings, dtype=float))
    mu = (P * r).sum(axis=-1, keepdims=True)
    return np.sqrt(np.clip((P * (r - mu) ** 2).sum(axis=-1), 0.0, None))


def dispersion_reliability(P, disagreement, n_bins: int = 10,
                           ratings=None) -> dict:
    """Reliability of PREDICTED PANEL DISPERSION - the ECE construction, with
    the panel's disagreement in place of the label.

    Bins items by predicted spread and compares, in each bin, the mean
    predicted dispersion against the mean dispersion the panel actually had.
    A well-calibrated panel head satisfies E[d | sigma_hat = s] = s: when it
    says "this panel will disagree by about one rating point", the panels it
    says that about disagree by about one rating point.

    Bin edges are EQUAL WIDTH over [0, (K-1)/2], the range a spread can take on
    a K-point scale, and are therefore the same for every arm - quantile bins
    would give each arm its own edges and the arms would no longer be
    comparable, which is the whole point of the table this feeds.

    Returns the per-bin curve (the reliability diagram) plus the counts, so a
    bin holding four items can be seen for what it is."""
    d = np.asarray(disagreement, dtype=float).ravel()
    s = predicted_spread(P, ratings)
    k = np.asarray(P).shape[-1]
    hi = (k - 1) / 2.0
    edges = np.linspace(0.0, hi, int(n_bins) + 1)
    out = {"bin_lo": [], "bin_hi": [], "n": [], "predicted": [], "actual": []}
    if s.size != d.size or not d.size:
        return out
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, n_bins - 1)
    for b in range(int(n_bins)):
        m = idx == b
        if not m.any():
            continue
        out["bin_lo"].append(float(edges[b]))
        out["bin_hi"].append(float(edges[b + 1]))
        out["n"].append(int(m.sum()))
        out["predicted"].append(float(s[m].mean()))
        out["actual"].append(float(d[m].mean()))
    return out


def dispersion_calibration_error(P, disagreement, n_bins: int = 10,
                                 ratings=None) -> float:
    """DCE - the count-weighted mean |predicted - actual| across the bins of
    `dispersion_reliability`. ECE's formula, in rating points.

    Why this is not `ece`: that one asks whether the predicted PROBABILITY of
    the label matches how often the label occurs. On LIDC no ECE contrast
    survives - every interval includes zero - and it never had much to find,
    because temperature scaling already fixes what a classifier gets wrong
    there. DCE asks a question the label cannot answer at all: does the model
    know how much the EXPERTS will disagree, and is that estimate calibrated
    rather than merely correlated? `spread_bias` is one global gap and cancels
    a head that over-states united panels and under-states divided ones; the
    binned version does not.

    NEVER REPORT IT ALONE. It inherits ECE's blind spot: a head with no
    resolution - one that predicts the same dispersion for every case - lands
    every item in a single bin, where the bin's own mean is compared against
    itself, and scores a LOW DCE while being useless. Measured: a constant
    predictor on a two-valued cohort scores DCE 0.10 with `spread_mae` 0.70.
    So calibration (this), ranking (`spread_corr`) and error (`spread_mae`)
    are reported together, and a claim needs all three - which is the same
    reason a classifier is never reported on ECE alone."""
    rel = dispersion_reliability(P, disagreement, n_bins, ratings)
    n = np.asarray(rel["n"], dtype=float)
    if not n.size:
        return float("nan")
    gap = np.abs(np.asarray(rel["predicted"]) - np.asarray(rel["actual"]))
    return float((n * gap).sum() / n.sum())


def spread_error(P, disagreement, ratings=None) -> dict:
    """Predicted panel dispersion against the panel's OWN dispersion statistic.

        spread_mae   |sigma_hat - d| in rating points
        spread_bias  mean(sigma_hat) - mean(d): does it over- or under-state how
                     divided panels are, the recalibration half
        spread_corr  Spearman(sigma_hat, d): does it know WHICH panels split,
                     the half the contribution actually claims

    This is the read-out to rank contestedness by. `entropy_error` answers the
    same two questions about a DIFFERENT functional of the histogram: entropy
    cannot see where the mass sits, so 1/1/5/5 and 2/2/4/4 are identical to it
    while their spreads are 2.0 and 1.0 - and the cohort's `disagreement`
    column, the deferral head's `tau_d`, and the split-vs-agreed contrast are
    all defined on the spread. Both are reported; the spread is the one the
    claim is made on."""
    d = np.asarray(disagreement, dtype=float).ravel()
    s = predicted_spread(P, ratings)
    if s.size != d.size or not d.size:
        return {}
    return {"spread_mae": float(np.mean(np.abs(s - d))),
            "spread_bias": float(np.mean(s) - np.mean(d)),
            "spread_corr": _spearman(s, d),
            # the CALIBRATION half: ECE's construction on panel dispersion.
            # Reported beside the label's `ece`, which on LIDC is a null.
            "spread_dce": dispersion_calibration_error(P, d, ratings=ratings)}


def opinion_metrics(P, Q, disagreement=None) -> dict:
    """The full opinion-head report: placement (EMD), shape (KL) and spread
    (entropy error / correlation).

    Pass the panel's raw `disagreement` (rating std) to also get the correlation
    between predicted entropy and the dispersion statistic the deferral head was
    supervised on - the check that the opinion head has learned the same notion
    of contentiousness by a completely different route."""
    out = {"opinion_emd": emd_distance(P, Q), "opinion_kl": kl_divergence(P, Q)}
    out.update(entropy_error(P, Q))
    if disagreement is not None and len(np.asarray(disagreement).ravel()):
        out["entropy_disagreement_corr"] = _spearman(distribution_entropy(P),
                                                     disagreement)
        out.update(spread_error(P, disagreement))
    return out
