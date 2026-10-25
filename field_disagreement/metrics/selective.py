"""Selective prediction: what the decide-or-defer policy actually buys.

The point of deferring is that the cases you keep are ones you get right. That is
only a claim until it is measured AGAINST A BASELINE, because any monotone
function of the predicted probability already produces a downward-sloping
risk-coverage curve. So the headline number here is never a single AURC - it is
`compare_selectors`, one row per selection rule on identical data:

    confidence   |p - 0.5|                the classical baseline every selective
                                          -prediction paper reports
    defer_head   P(defer)                 the panel-risk head
    uncertainty  MC-dropout epistemic     model spread, not panel spread
    policy       learned combination      fitted against the model's own errors

THIS TABLE SCORES ONE OF THE TWO RISKS. Every column here is about the model
being wrong, so it ranks `r_err` estimators and confidence is the baseline to
beat. A panel-risk signal is not competing for this column - it is scored on
`disagreement_separation` (does it find the divided panels) and on
`contested_leakage` (does the deployed rule stop auto-clearing them). A
mechanism earns its place by beating the baseline on the risk it estimates;
losing the other one is information about what the two signals see, not a
failure, and the pair of results is what justifies routing on both.

Orientation convention: every function here takes CONFIDENCE (higher = decide
first); a defer score is its negation.
"""
from __future__ import annotations
import numpy as np
from .calibration import _binary
from .classification import roc_auc


# --------------------------------------------------------------------------- #
# selective prediction: risk-coverage / AURC 
# --------------------------------------------------------------------------- #
def risk_coverage(y_true, y_prob, confidence=None):
    """Selective-risk vs coverage by descending confidence. Returns
    (coverages, risks, aurc). Higher confidence is decided first; risk is the
    error rate among the decided fraction. AURC = area under the curve."""
    y, p = _binary(y_true, y_prob)
    if not len(y):
        return np.array([]), np.array([]), float("nan")
    conf = np.abs(p - 0.5) if confidence is None else np.asarray(confidence, float)
    err = ((p >= 0.5).astype(int) != y).astype(float)
    order = np.argsort(-conf)
    cum_err = np.cumsum(err[order])
    n = len(y)
    coverages = np.arange(1, n + 1) / n
    risks = cum_err / np.arange(1, n + 1)
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # np.trapz removed in NumPy 2.0
    aurc = float(_trapz(risks, coverages))
    return coverages, risks, aurc


def _decided_mask(confidence, coverage: float) -> np.ndarray:
    """The top-`coverage` fraction by confidence, as a boolean mask."""
    conf = np.asarray(confidence, dtype=float).ravel()
    n = len(conf)
    k = round(min(max(coverage, 0.0), 1.0) * n)
    mask = np.zeros(n, dtype=bool)
    if k:
        mask[np.argsort(-conf)[:k]] = True
    return mask


def risk_at_coverage(y_true, y_prob, confidence=None, coverage: float = 0.8) -> float:
    """Error rate among the cases kept at a fixed coverage."""
    y, p = _binary(y_true, y_prob)
    if not len(y):
        return float("nan")
    conf = np.abs(p - 0.5) if confidence is None else confidence
    m = _decided_mask(conf, coverage)
    if not m.any():
        return float("nan")
    return float((((p[m] >= 0.5).astype(int)) != y[m]).mean())


def sensitivity_at_coverage(y_true, y_prob, confidence=None,
                            coverage: float = 0.8) -> dict:
    """Malignancy sensitivity once the deferred cases are removed.

    Accuracy at fixed coverage can improve simply by dropping the hard positives,
    which for a screening system is the failure mode, not the goal. Two numbers
    keep that honest:

      sensitivity        recall among the DECIDED malignant cases - the quality of
                         the answers the system actually stands behind;
      missed_malignant   malignant cases called benign, over ALL malignant cases
                         including the deferred ones. A deferred cancer is not
                         missed (a reader still sees it), so this counts only the
                         confident false negatives - the ones that reach nobody.
      deferred_malignant fraction of malignant cases sent to a reader, so a policy
                         that hits its sensitivity by deferring every positive is
                         visible rather than flattering.
    """
    y, p = _binary(y_true, y_prob)
    nan = {"sensitivity": float("nan"), "missed_malignant": float("nan"),
           "deferred_malignant": float("nan")}
    if not len(y) or not (y == 1).any():
        return nan
    conf = np.abs(p - 0.5) if confidence is None else confidence
    m = _decided_mask(conf, coverage)
    pos = y == 1
    n_pos = int(pos.sum())
    decided_pos = pos & m
    if not decided_pos.any():
        return {"sensitivity": float("nan"), "missed_malignant": 0.0,
                "deferred_malignant": 1.0}
    caught = int((p[decided_pos] >= 0.5).sum())
    return {"sensitivity": caught / int(decided_pos.sum()),
            "missed_malignant": float((int(decided_pos.sum()) - caught) / n_pos),
            "deferred_malignant": float((pos & ~m).sum() / n_pos)}


def coverage_at_sensitivity(y_true, y_prob, confidence=None,
                            target_sensitivity: float = 0.98) -> float:
    """The largest fraction of cases the system can auto-report while holding
    sensitivity among the DECIDED cases at `target_sensitivity`.

    This is the inverse of `risk_at_coverage` and it is the way the question is
    actually posed in a reading room: sensitivity is the constraint a screening
    programme will not trade, and workload is what you are allowed to optimise. It
    also makes the deferral comparison honest in a way a fixed-coverage table
    cannot - two rules held at 80% coverage can differ in sensitivity, and a
    reader has no way to price that difference. Held at equal sensitivity, the
    better rule is simply the one that hands back more cases.

    Returns nan when no coverage attains the target (including the degenerate
    case of a model that cannot reach it at any operating point)."""
    y, p = _binary(y_true, y_prob)
    if not len(y) or not (y == 1).any():
        return float("nan")
    conf = np.abs(p - 0.5) if confidence is None else np.asarray(confidence, float)
    order = np.argsort(-conf)
    y_o, p_o = y[order], p[order]
    pos = (y_o == 1)
    hit = pos & (p_o >= 0.5)
    n_dec_pos = np.cumsum(pos)                 # malignant cases decided so far
    n_caught = np.cumsum(hit)
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = np.where(n_dec_pos > 0, n_caught / np.maximum(n_dec_pos, 1), np.nan)
    ok = np.isfinite(sens) & (sens >= float(target_sensitivity)) & (n_dec_pos > 0)
    if not ok.any():
        return float("nan")
    return float((np.flatnonzero(ok)[-1] + 1) / len(y))


def error_concentration(y_true, y_prob, defer_mask) -> dict:
    """Do the deferred cases actually contain the errors?

    `lift` is the error rate among deferred divided by the overall error rate: 1.0
    means the policy referred cases at random, and > 1 means it found them. This
    is the single number that says whether deferral is selecting for difficulty or
    just shrinking the test set."""
    y, p = _binary(y_true, y_prob)
    m = np.asarray(defer_mask, dtype=bool).ravel()
    if not len(y):
        return {"deferred_error_rate": float("nan"), "kept_error_rate": float("nan"),
                "error_lift": float("nan"), "defer_rate": float("nan")}
    err = ((p >= 0.5).astype(int) != y).astype(float)
    base = float(err.mean())
    d = float(err[m].mean()) if m.any() else float("nan")
    k = float(err[~m].mean()) if (~m).any() else float("nan")
    return {"deferred_error_rate": d, "kept_error_rate": k,
            "error_lift": (d / base) if base > 0 else float("nan"),
            "defer_rate": float(m.mean())}


def rank_error_auc(y_true, y_prob, defer_score) -> float:
    """AUC of the defer score at ranking the model's own errors above its hits.

    Coverage-free, so it compares selection rules without first agreeing an
    operating point. 0.5 is a coin flip; nan when the model made no errors."""
    y, p = _binary(y_true, y_prob)
    if not len(y):
        return float("nan")
    err = ((p >= 0.5).astype(int) != y).astype(int)
    if err.min() == err.max():
        return float("nan")
    return roc_auc(err, np.asarray(defer_score, dtype=float).ravel())


def selective_metrics(y_true, y_prob, confidence=None, defer_mask=None,
                      coverages=(0.7, 0.8, 0.9),
                      sensitivities=(0.95, 0.98)) -> dict:
    """One selection rule, fully characterised: AURC, the risk / sensitivity at
    each requested coverage, the error-ranking AUC, and - when an explicit
    `defer_mask` is supplied rather than a coverage sweep - where the errors
    ended up."""
    _, _, area = risk_coverage(y_true, y_prob, confidence)
    out = {"aurc": area,
           "rank_error_auc": rank_error_auc(
               y_true, y_prob,
               -np.asarray(confidence if confidence is not None
                           else np.abs(_binary(y_true, y_prob)[1] - 0.5), float))}
    for c in coverages:
        tag = f"{round(c * 100)}"
        out[f"risk@{tag}"] = risk_at_coverage(y_true, y_prob, confidence, c)
        s = sensitivity_at_coverage(y_true, y_prob, confidence, c)
        out[f"sens@{tag}"] = s["sensitivity"]
        out[f"missed_mal@{tag}"] = s["missed_malignant"]
    # Workload at a fixed sensitivity rather than a coverage we chose.
    for ts in sensitivities:
        out[f"cov@sens{round(ts * 100)}"] = coverage_at_sensitivity(
            y_true, y_prob, confidence, ts)
    if defer_mask is not None:
        out.update(error_concentration(y_true, y_prob, defer_mask))
    return out


# --------------------------------------------------------------------------- #
# the PANEL risk: what the rule auto-decided that the experts would have argued about
# --------------------------------------------------------------------------- #
def contested_leakage(defer_score, disagreement, d0: float,
                      coverage: float = 0.8) -> dict:
    """Of the cases a rule AUTO-DECIDES, what share would the panel have split on?

        leakage    = P(d > d0 | auto-decided)
        enrichment = P(d > d0 | deferred) / P(d > d0)
        base       = P(d > d0) over the whole test set

    The operational reading of the contribution, and the one number a
    confidence-only system cannot optimise: it ranks the model's own errors, so
    the contested cases it happens to clear are whatever falls out. Leakage
    BELOW the base rate at equal coverage is the claim; enrichment above 1 says
    the referral queue concentrates the cases a human is actually needed for.

    `d0` is the cohort's declared split threshold (`contested_d`), not a
    quantile of this test set - it has to mean the same thing across folds and
    cohorts. Coverage is the exact top-k by score (`_decided_mask`), so every
    rule in a comparison auto-decides the same NUMBER of cases and the leakage
    columns are directly comparable."""
    s = np.asarray(defer_score, dtype=float).ravel()
    d = np.asarray(disagreement, dtype=float).ravel()
    nan = float("nan")
    if s.size == 0 or s.size != d.size:
        return {"leakage": nan, "enrichment": nan, "base": nan, "coverage": nan,
                "n_auto": 0, "n_deferred": 0}
    contested = d > d0
    base = float(contested.mean())
    keep = _decided_mask(-s, coverage)                 # lowest defer score first
    n_auto, n_def = int(keep.sum()), int((~keep).sum())
    lk = float(contested[keep].mean()) if n_auto else nan
    df = float(contested[~keep].mean()) if n_def else nan
    return {"leakage": lk,
            "enrichment": (df / base) if (n_def and base > 0) else nan,
            "base": base, "coverage": float(n_auto / s.size),
            "n_auto": n_auto, "n_deferred": n_def,
            "leakage_lift": (lk / base) if (n_auto and base > 0) else nan}


# the SERVED outcome: what a decide-or-defer rule did to each kind of case
SERVED_OUTCOMES = ("auto", "error", "missed_pos", "missed_per_1000", "false_fast",
                   "contested_neg", "contested_auto", "high_neg", "obvious_pos_auto",
                   "obvious_neg_auto")


def served_outcomes(auto, y_prob, y_true, disagreement, d0: float, *,
                    high_q: float = 0.9, obvious: float = 0.9) -> dict:
    """Score one decide-or-defer outcome (`auto`: True = decided without a reader)
    on the things a clinician asks about, each over the WHOLE split:

      auto              share decided without a reader                (higher = less reading)
      error             wrong among the auto-decided                  (lower is better)
      missed_pos        positives the system cleared as negative,
                        over all positives - the missed-disease rate  (lower is better)
      missed_per_1000   the same, per 1000 cases                      (lower is better)
      false_fast        negatives the system called positive, over
                        all negatives - needless work-ups             (lower is better)
      contested_neg     contested cases (d > d0) cleared as negative,
                        over all contested                            (lower is better)
      contested_auto    contested cases decided without a reader at
                        all (either call), over all contested         (lower is better)
      high_neg          HIGHLY contested cases (d at or above this
                        split's `high_q` quantile) cleared as
                        negative, over all of them                    (lower is better)
      obvious_pos_auto  true positives the model is sure of
                        (p >= obvious) decided without a reader       (higher is better)
      obvious_neg_auto  true negatives it is sure of
                        (p <= 1 - obvious) decided without a reader   (higher is better)

    "Cleared as negative" is auto-decided with p < 0.5. A share whose denominator
    is empty is NaN. `d0` is the cohort's declared cut; `high_q` is a stratum of
    this split, the same for every rule scored on it."""
    auto = np.asarray(auto, dtype=bool).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    y = np.asarray(y_true).astype(int).ravel()
    d = np.asarray(disagreement, dtype=float).ravel()
    nan = float("nan")
    if not auto.size:
        return {k: nan for k in SERVED_OUTCOMES}
    neg_call = auto & (p < 0.5)
    pos, contested = y == 1, d > d0
    high = d >= np.quantile(d, high_q)

    def share(num, den):
        return float((num & den).sum() / den.sum()) if den.any() else nan

    wrong = (p >= 0.5).astype(int) != y
    return {"auto": float(auto.mean()),
            "error": float(wrong[auto].mean()) if auto.any() else nan,
            "missed_pos": share(neg_call, pos),
            "missed_per_1000": float((neg_call & pos).sum() / auto.size * 1000),
            "false_fast": share(auto & (p >= 0.5), ~pos),
            "contested_neg": share(neg_call, contested),
            "contested_auto": share(auto, contested),
            "high_neg": share(neg_call, high),
            "obvious_pos_auto": share(auto, pos & (p >= obvious)),
            "obvious_neg_auto": share(auto, ~pos & (p <= 1.0 - obvious))}


def expected_reader_accuracy(q_true, y_true, neutral: int | None = None) -> np.ndarray:
    """P(a single panel member agrees with the panel's own verdict), per item.

    Read straight off the observed histogram: the share of the panel whose
    rating sits on the reference label's side of the scale. A 4/0 panel gives
    1.0 and a 2/2 panel gives 0.5 - the case where asking ONE more reader is a
    coin flip.

    This is the quantity a learning-to-defer system needs and almost never has:
    the expert's per-case accuracy. A multi-rater panel supplies it directly,
    which is what makes the deferral claim testable here rather than assumed.

    `neutral` is the index of a rating that takes no side (LIDC's "3"), whose
    mass is counted as NOT agreeing - a reader who declines to call it has not
    resolved the case. Items whose panel is entirely neutral score 0.

    The reference label is derived from the same panel, so this is an OPTIMISTIC
    estimate of a fresh reader and is reported as a simulation, never as a
    measured second-reader accuracy."""
    Q = np.asarray(q_true, dtype=float)
    if Q.ndim == 1:
        Q = Q[None]
    y = np.asarray(y_true).astype(int).ravel()
    k = Q.shape[1]
    mid = (k - 1) / 2.0 if neutral is None else float(neutral)
    idx = np.arange(k)
    pos, neg = (idx > mid), (idx < mid)
    return np.where(y == 1, Q[:, pos].sum(axis=1), Q[:, neg].sum(axis=1))


def team_accuracy(y_true, y_prob, defer_score, reader_accuracy,
                  coverage: float = 0.8) -> dict:
    """Accuracy of the HUMAN-AI TEAM: the model answers what it keeps, a reader
    answers what it refers.

        team = P(model right | kept) * coverage
             + E[reader accuracy | deferred] * (1 - coverage)

    The metric selective prediction leaves out. A risk-coverage curve prices a
    referral at zero cost and assumes the human is always right, so it ranks a
    rule by the errors it removes and never asks whether the case it handed over
    is one a human can settle. On a divided panel the single-reader term is near
    0.5, so referring it buys almost nothing - and referring a case the panel
    agreed on but the model finds hard buys almost a whole correction.

    Reported beside `risk@coverage`, not instead of it: the two answer different
    questions and a rule can win one and lose the other."""
    y, p = _binary(y_true, y_prob)
    acc = np.asarray(reader_accuracy, dtype=float).ravel()
    s = np.asarray(defer_score, dtype=float).ravel()
    if not len(y) or acc.size != y.size or s.size != y.size:
        return {"team_accuracy": float("nan"), "model_accuracy": float("nan"),
                "reader_share": float("nan")}
    keep = _decided_mask(-s, coverage)
    model_right = ((p >= 0.5).astype(int) == y)
    cov = float(keep.mean())
    kept_acc = float(model_right[keep].mean()) if keep.any() else 0.0
    ref_acc = float(acc[~keep].mean()) if (~keep).any() else 0.0
    return {"team_accuracy": kept_acc * cov + ref_acc * (1.0 - cov),
            "model_accuracy": float(model_right.mean()),
            "kept_accuracy": kept_acc,
            "referred_reader_accuracy": ref_acc,
            "reader_share": 1.0 - cov}


def compare_selectors(y_true, y_prob, selectors: dict,
                      coverages=(0.7, 0.8, 0.9),
                      sensitivities=(0.95, 0.98)) -> dict:
    """The head-to-head table: `{name: confidence array}` -> `{name: metrics}`.

    `confidence` is inserted automatically as the baseline every other rule has to
    beat, so no caller can accidentally report a deferral mechanism without its
    control."""
    _, p = _binary(y_true, y_prob)
    rules = {"confidence": np.abs(p - 0.5)}
    rules.update(selectors)
    return {name: selective_metrics(y_true, y_prob, conf, coverages=coverages,
                                    sensitivities=sensitivities)
            for name, conf in rules.items()}
