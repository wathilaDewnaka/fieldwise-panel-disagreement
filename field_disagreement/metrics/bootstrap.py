"""Bootstrap CIs and per-fold aggregation.

Moved verbatim from backend/training/{metrics}.py — bodies unchanged.
"""
from __future__ import annotations
import numpy as np
from .classification import roc_auc
from .distribution import _spearman


# --------------------------------------------------------------------------- #
# bootstrap 95% CI (effect-size reporting)
# --------------------------------------------------------------------------- #
def bootstrap_auc_ci(y_true, y_score, n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = 0):
    """Point AUC plus a bootstrap (1-alpha) CI. Used for the score-3
    agreed-vs-split separation effect size."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    point = roc_auc(y, s)
    n = len(y)
    if n == 0 or np.isnan(point):
        return {"auc": point, "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = roc_auc(y[idx], s[idx])
        if not np.isnan(a):
            boots.append(a)
    if not boots:
        return {"auc": point, "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {"auc": point, "ci95_lo": lo, "ci95_hi": hi}


def disagreement_separation(disagreement, signals: dict,
                            split: str = "nonzero", seed: int = 0,
                            tau: float | None = None, clusters=None) -> dict:
    """Can a signal tell a UNITED panel from a DIVIDED one, on the borderline
    items the field normally discards?

    Those items never enter a supervised binary loss, so a system claiming it
    "knows when not to decide" has to rank the divided panels above the united
    ones using nothing it was trained on. Separation AUC per signal with a
    bootstrap CI; a CI straddling 0.5 is a null result and should be reported so.

    `signals` is {name: per-item score}. A CONSTANT signal is left unscored
    rather than returned as 0.5, which would read as "measured, no effect"
    instead of "this arm has no such signal".

      "threshold" united (d == 0) vs GENUINELY divided (d >= `tau`), dropping
                 the middle. The contrast the design is about and the one to
                 report: `tau` is the cohort's declared `contested_d`, the
                 same number the deferral head is supervised against, so it is
                 not a cut chosen to suit the answer.
      "nonzero"  united (d == 0) vs ANY non-unanimous panel (d > 0). Much
                 weaker than it sounds: with 4 raters on a 1-5 scale a single
                 rater one step out already counts as divided, so most of the
                 "divided" side is near-agreement and every signal is pulled
                 towards 0.5. Kept as the historical default and as a
                 sensitivity check, not as the headline.
      "median"   below vs above median d. Where most panels disagree slightly
                 this separates "contested" from "more contested", a weaker claim.

    `clusters` (one id per item, e.g. patient) resamples clusters rather than
    items, which is what the CI has to do when one subject contributes several
    nodules - item-level intervals are too narrow by roughly the intra-cluster
    correlation.

    Pool the folds before calling this: per fold the contrast can be a dozen
    united panels against ninety divided ones, which no bootstrap can resolve.
    """
    d = np.asarray(disagreement, dtype=float).ravel()
    nan = {"auc": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    keep = np.ones(d.shape, dtype=bool)
    if split == "nonzero":
        group = (d > 0.0).astype(int)
    elif split == "median":
        group = (d > np.median(d)).astype(int) if d.size else np.zeros(0, dtype=int)
    elif split == "threshold":
        if tau is None:
            raise ValueError("split='threshold' needs the cohort's `tau` "
                             "(contested_d), declared before the run")
        keep = (d <= 0.0) | (d >= float(tau))
        group = (d >= float(tau)).astype(int)
    else:
        raise ValueError(f"unknown split {split!r} (threshold|nonzero|median)")

    group = group[keep]
    cl = None if clusters is None else np.asarray(clusters).ravel()[keep]
    n_high, n_low = int(group.sum()), int((group == 0).sum())
    out = {"n": int(keep.sum()), "n_divided": n_high, "n_united": n_low,
           "split": split, "tau": (None if tau is None else float(tau)),
           "signals": {}}
    if n_high < 2 or n_low < 2:
        # One side of the contrast is empty - there is nothing to separate, which
        # is a property of the cohort and not a failure of any signal.
        out["signals"] = {k: dict(nan) for k in signals}
        return out

    for name, values in signals.items():
        v = np.asarray(values, dtype=float).ravel()
        if v.size != d.size or float(np.std(v)) == 0.0:
            out["signals"][name] = dict(nan)
            continue
        v = v[keep]
        out["signals"][name] = (bootstrap_auc_ci(group, v, seed=seed) if cl is None
                                else _clustered_auc_ci(group, v, cl, seed=seed))
    return out


def _clustered_auc_ci(y, score, clusters, n_boot: int = 2000, seed: int = 0) -> dict:
    """Separation AUC with a CLUSTER bootstrap: whole subjects are resampled, so
    a patient contributing five nodules moves in and out together."""
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(score, dtype=float).ravel()
    cl = np.asarray(clusters).ravel()
    point = roc_auc(y, s)
    groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([groups[i] for i in
                               rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[pick])) < 2:
            continue
        boots.append(roc_auc(y[pick], s[pick]))
    if not boots:
        return {"auc": point, "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": float(point), "ci95_lo": float(lo), "ci95_hi": float(hi),
            "n_boot": len(boots)}


def rank_agreement(disagreement, signals: dict, clusters=None,
                   n_boot: int = 2000, seed: int = 0) -> dict:
    """THRESHOLD-FREE version of `disagreement_separation`: Spearman rho between
    each signal and the panel's dispersion, over every item.

    Report it beside the split table. It uses the whole borderline set rather
    than its two ends, and - the reason it matters for a claim anyone will
    challenge - no cut-off was chosen, so "you picked the threshold that
    worked" has no purchase on it. `clusters` resamples subjects.

    Also returns each signal's rho MINUS the first signal's, paired on the same
    items: the contrast to confidence is the claim, and a paired interval is
    tighter and more honest than two overlapping marginal ones."""
    d = np.asarray(disagreement, dtype=float).ravel()
    names = list(signals)
    vals = {k: np.asarray(v, dtype=float).ravel() for k, v in signals.items()}
    out: dict = {"n": int(d.size), "signals": {}}
    if d.size < 8 or not names:
        return out
    ref = names[0]
    cl = np.arange(d.size) if clusters is None else np.asarray(clusters).ravel()
    groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    rng = np.random.default_rng(seed)
    picks = [np.concatenate([groups[i] for i in
                             rng.integers(0, len(groups), len(groups))])
             for _ in range(n_boot)]
    for name in names:
        v = vals[name]
        if v.size != d.size or float(np.std(v)) == 0.0:
            out["signals"][name] = {"rho": float("nan")}
            continue
        boots = [_spearman(v[p], d[p]) for p in picks]
        boots = [b for b in boots if np.isfinite(b)]
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots
                  else (float("nan"), float("nan")))
        block = {"rho": float(_spearman(v, d)), "ci95_lo": float(lo),
                 "ci95_hi": float(hi)}
        if name != ref and vals[ref].size == d.size:
            diffs = [_spearman(v[p], d[p]) - _spearman(vals[ref][p], d[p])
                     for p in picks]
            diffs = [x for x in diffs if np.isfinite(x)]
            if diffs:
                dlo, dhi = np.percentile(diffs, [2.5, 97.5])
                block.update({f"vs_{ref}": float(block["rho"] -
                                                 _spearman(vals[ref], d)),
                              f"vs_{ref}_lo": float(dlo),
                              f"vs_{ref}_hi": float(dhi)})
        out["signals"][name] = block
    return out


def _exact_sign_p(wins: int, n: int) -> float:
    """Two-sided exact sign-test p for `wins` successes in `n` paired folds."""
    if n == 0:
        return float("nan")
    from math import comb
    tail = lambda k: sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    k = max(wins, n - wins)
    return float(min(1.0, 2 * tail(k)))


def paired_fold_test(per_fold_a: list[dict], per_fold_b: list[dict], key: str,
                     lower_is_better: bool = False) -> dict:
    """Compare two arms on `key` using the SAME folds, paired.

    Arms are trained on identical splits, so the fold is a blocking factor and
    the per-fold difference removes the fold-to-fold variance that swamps an
    unpaired comparison. With 5 folds the reported `p` is an exact sign test -
    honest about what 5 paired observations can support (its floor is 0.0625, so
    it can never be 'significant' at 0.05, which is a fact about the protocol and
    should be reported as such rather than swapped for a test that flatters it).

    `delta` is always oriented so POSITIVE = arm A is better, whichever direction
    the metric improves in."""
    a = np.array([f.get(key, np.nan) for f in per_fold_a], dtype=float)
    b = np.array([f.get(key, np.nan) for f in per_fold_b], dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    n = len(a)
    if n == 0:
        return {"delta": float("nan"), "ci95_lo": float("nan"),
                "ci95_hi": float("nan"), "wins": 0, "n": 0, "p": float("nan")}
    d = (b - a) if lower_is_better else (a - b)
    mean = float(d.mean())
    std = float(d.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    wins = int((d > 0).sum())
    return {"delta": mean, "ci95_lo": mean - half, "ci95_hi": mean + half,
            "wins": wins, "n": n, "p": _exact_sign_p(wins, n),
            # The per-fold differences themselves. A claim of the form "positive
            # on every fold" is about these, not about the mean and its CI: five
            # deltas of +0.02 and one of -0.30 average to something positive and
            # are not the same finding as five of +0.05.
            "per_fold": [float(x) for x in d]}


# --------------------------------------------------------------------------- #
# paired AUC difference, cluster bootstrap
# --------------------------------------------------------------------------- #
def paired_auc_bootstrap(y, score_a, score_b, clusters,
                         n_boot: int = 5000, seed: int = 0) -> dict:
    """AUC(a) - AUC(b) on the same items, with a cluster-resampled CI.

    The classification-parity check between two arms: an interval containing 0
    says the two discriminate about equally well. AUC is not a mean of per-item quantities, so it
    has to be recomputed inside each bootstrap draw rather than averaged."""
    y = np.asarray(y)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    c = np.asarray(list(clusters))
    point = roc_auc(y, a) - roc_auc(y, b)
    if len(y) == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "n_clusters": 0}
    uniq = np.unique(c)
    index_of = {u: np.where(c == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([index_of[u]
                              for u in rng.choice(uniq, len(uniq), replace=True)])
        d = roc_auc(y[idx], a[idx]) - roc_auc(y[idx], b[idx])
        if not np.isnan(d):
            boots.append(d)
    lo, hi = ((float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
              if boots else (float("nan"), float("nan")))
    return {"delta": float(point), "lo": lo, "hi": hi,
            "n": len(y), "n_clusters": len(uniq)}


def aggregate_folds(per_fold: list[dict]) -> dict:
    """mean, std and 95% CI across folds for each scalar metric. CI is the normal-approx
    mean +/- 1.96 * std / sqrt(k) over the k folds."""
    # THE UNION of every fold's scalar keys, not fold 0's. A signal is dropped
    # from a fold when it carries no information there - `panel_signals` omits
    # `epistemic` when MC dropout gives it zero variance, and omits
    # `panel_corrected` when an arm has no opinion head or no fitted corrector.
    # Reading the key list off `per_fold[0]` broke that two ways at once:
    #
    #   * a key present in fold 0 and MISSING later raised KeyError and killed
    #     the run during AGGREGATION - after every fold had been trained, which
    #     is the most expensive possible moment to fail;
    #   * a key missing from fold 0 but present in folds 1..k-1 was dropped
    #     SILENTLY, so a metric measured on four folds reported nothing at all.
    #
    # `n` already records how many folds contributed, so a partially-present key
    # aggregates over the folds that have it and says so, which is the same
    # contract NaN values have always had here.
    keys, seen = [], set()
    for f in per_fold:
        for k, v in f.items():
            if k not in seen and isinstance(v, (int, float)) \
                    and not isinstance(v, bool):
                seen.add(k)
                keys.append(k)
    out = {}
    for k in keys:
        vals = np.array([f.get(k, float("nan")) for f in per_fold], dtype=float)
        vals = vals[~np.isnan(vals)]
        n = len(vals)
        if n == 0:
            out[k] = {"mean": float("nan"), "std": float("nan"),
                      "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n": 0}
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        out[k] = {"mean": mean, "std": std,
                  "ci95_lo": mean - half, "ci95_hi": mean + half, "n": n}
    return out
