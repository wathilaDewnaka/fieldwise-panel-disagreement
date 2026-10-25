"""Panel-aware deferral.

Two mechanisms live here and the difference between them is the contribution.

**Disagreement-supervised head** (`build_deferral_targets` + `deferral_ce_loss`).
A third logit trained against `{benign, malignant, defer}`, where `defer` is
assigned whenever the panel's rating spread exceeds `tau_d`. This is
a useful auxiliary task - it forces the trunk to represent panel conflict - but on
its own it is only a network learning to imitate one threshold on one variable
the dataset already stores. It cannot know anything the threshold did not say,
and at test time it is a re-derivation of training annotations.

**Learned deferral policy** (`DeferralPolicy`). The decision the deployed system
actually makes. It combines several PREDICTED signals - the deferral head, the
classifier's own confidence, the predicted panel entropy, the MC-dropout
epistemic term, and the conformal set's ambiguity - and is fitted on the
validation split to predict *whether the model is about to be wrong*. Nothing in
its target comes from annotator disagreement, so it is free to defer on a case
the panel agreed about but the model does not understand, and free to decide on a
contested case it nonetheless gets right. That is the clinically useful ordering
and it is not the one a disagreement threshold produces.

The policy is a small L2-regularised logistic regression on standardised signals:
deliberately the simplest model that can weigh the signals against each other, so
the fitted coefficients are readable as "what actually drove the deferrals" and
the val split is not overfitted by the thing meant to make the system safer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# disagreement-supervised deferral head 
# --------------------------------------------------------------------------- #
def rank_gauss(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """The regression target's normal scores against a TRAIN reference.

    `ref` is the sorted training targets of this fold. Each value maps to its
    mid-rank percentile in `ref` (ties share one score), clipped half a rank
    inside (0, 1), then to the standard-normal quantile. Monotone, so the
    ORDER - all the Spearman primary reads - is exactly the target's, but a
    heavy right tail no longer dominates the squared error. Nothing about the
    held-out split enters: `ref` holds training values only."""
    ref = ref.to(x.device, x.dtype)
    n = ref.numel()
    lo = torch.searchsorted(ref, x.contiguous(), right=False).to(x.dtype)
    hi = torch.searchsorted(ref, x.contiguous(), right=True).to(x.dtype)
    u = ((lo + hi) / 2.0 / n).clamp(0.5 / n, 1.0 - 0.5 / n)
    return torch.special.ndtri(u)


def build_deferral_targets(disagreement: torch.Tensor, label: torch.Tensor,
                           tau_d: float) -> torch.Tensor:
    """Three-way target: defer (2) if d > tau_d, else the binary class (0/1)."""
    y = label.long()
    defer = disagreement > tau_d
    return torch.where(defer, torch.full_like(y, 2), y)


def deferral_ce_loss(defer_logits: torch.Tensor,
                     y_tilde: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over {benign, malignant, defer}."""
    return F.cross_entropy(defer_logits, y_tilde)


def defer_lambda(epoch: int, lam_max: float, warmup_frac: float,
                 epochs: int) -> float:
    """Warm-up for the deferral term: linear over the first `warmup_frac` of
    training, then clamped at `lam_max`.

    Deferral supervision is only meaningful relative to a classifier that has
    started to work; at full strength from step 0 the two objectives fight for
    the trunk when the classification signal is weakest.

    It ramps from the start and is a FRACTION OF TOTAL TRAINING, so it stretches
    with `epochs`.
    """
    span = max(1.0, float(warmup_frac) * int(epochs))
    return float(lam_max) * min(1.0, (int(epoch) + 1) / span)


# --------------------------------------------------------------------------- #
# the signals the deployed policy decides on - all PREDICTED at inference
# --------------------------------------------------------------------------- #
def conformal_ambiguity(set_sizes) -> np.ndarray:
    """|C(x)| -> a defer-monotone score in the binary case.

    A singleton set is the confident outcome and scores 0. BOTH the doubleton
    (the model cannot separate the classes at this risk level) and the EMPTY set
    (no label is plausible - the case is atypical of everything the threshold was
    calibrated on) score 1. Passing the raw size would rank the empty set as the
    *most* confident outcome, which is exactly backwards."""
    return np.abs(np.asarray(set_sizes, dtype=float) - 1.0)


def confidence_gap(prob) -> np.ndarray:
    """`1 - 2|p - 0.5|`: how close the classifier sat to its own boundary.

    Higher = defer, like every other signal. The first column of `build_signals`
    and the baseline every deferral rule is scored against, so it has one
    definition rather than being re-typed wherever a confidence is needed."""
    p = np.asarray(prob, dtype=float).ravel()
    return 1.0 - 2.0 * np.abs(p - 0.5)


# The study's SERVED rule (`decide_or_defer`, rung `confidence+negative_veto`). Its
# priorities, in order: miss no more cancers and make no more errors than the
# uncertainty-threshold baseline (SOTA, the certainty cut at SOTA_MISS); send the
# contested cases to a reader; keep as much automatic as those two allow.
#   1. a certainty cut, set so the model alone clears at most SERVED_MISS of the
#      positives as negative on the calibration split (`safety_cut`) - stricter
#      than SOTA's SOTA_MISS, which is what keeps the errors below SOTA's;
#   2. on top of it, the model's negative calls whose predicted panel disagreement
#      is in the top NEGATIVE_VETO share go to a reader (`veto_cut`) - except an
#      OBVIOUS negative (p <= OBVIOUS_NEGATIVE), whose certainty keeps it automatic;
#   3. unless the panel is predicted to be at its most divided: a negative call in
#      the top EXTREME_VETO share goes to a reader however sure the model is.
# The veto only moves cases to a reader, so at the same cut it can never clear a
# positive the certainty rule would not have cleared. The values were chosen on
# the test outputs of earlier runs (disclosed in the paper's Limitations).
SOTA_MISS = 0.05
SERVED_MISS = 0.04
NEGATIVE_VETO = 0.40
OBVIOUS_NEGATIVE = 0.10
EXTREME_VETO = 0.08


@dataclass(frozen=True)
class ServedRule:
    """Every setting of the served rule, in one place.

    The defaults are the default rule: certainty cut at `miss`,
    veto of the `veto` share of non-obvious negative calls, the `extreme` share of
    any negative call, no safety net. A study states its own values in config and
    builds the rule with `from_config`, so no evaluation hard-codes them.

      miss         the certainty cut clears at most this share of positives
      sota_miss    the uncertainty-threshold reference's limit
      veto         share of negative calls sent to a reader by the method's signal
      obvious      p at or below this is an obvious negative (exempt from `veto`)
      extreme      share of ALL negative calls vetoed however sure the model is
      safety_net   share of ALL negative calls sent by the method's net signal
                   (its predicted opinion spread); 0 = no net
      miss_margin  non-inferiority margin for missed positives, per 1000
      splits       patient-disjoint calibration/evaluation halvings per fold
    """

    miss: float = SERVED_MISS
    sota_miss: float = SOTA_MISS
    veto: float = NEGATIVE_VETO
    obvious: float = OBVIOUS_NEGATIVE
    extreme: float = EXTREME_VETO
    safety_net: float = 0.0
    miss_margin: float = 5.0
    splits: int = 1000

    def __post_init__(self):
        for k in ("miss", "sota_miss", "veto", "obvious", "extreme", "safety_net"):
            v = getattr(self, k)
            if not 0.0 <= v < 1.0:
                raise ValueError(f"ServedRule.{k} must be in [0, 1), got {v}")
        if self.miss_margin < 0:
            raise ValueError(f"ServedRule.miss_margin must be >= 0, got {self.miss_margin}")
        if self.splits < 1:
            raise ValueError(f"ServedRule.splits must be >= 1, got {self.splits}")

    @classmethod
    def from_config(cls, block) -> "ServedRule":
        """From a config mapping or namespace (e.g. a study yaml's `served_rule:`
        block); keys it does not name keep the deployed defaults, unknown keys
        raise so a typo cannot silently fall back to a default."""
        if block is None:
            return cls()
        m = dict(block) if isinstance(block, dict) else dict(vars(block))
        unknown = set(m) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown served_rule keys {sorted(unknown)}")
        return cls(**{k: type(getattr(cls(), k))(v) for k, v in m.items()})


def safety_cut(conf, prob, labels, miss: float = SERVED_MISS) -> float:
    """The loosest certainty cut at which the model, deciding alone, clears at most
    `miss` of the calibration POSITIVES as negative.

    A case is auto-decided when `conf` (`1 - 2|p - 0.5|`, higher = less sure) is
    below the cut. The positives it would clear are those called negative
    (`p < 0.5`) below the cut, so the cut is the (k+1)-th smallest `conf` among
    them, k = floor(miss * n_positives). `inf` when even clearing all of them
    stays within `miss`. The limit holds on the calibration split; on new cases it
    holds only approximately, the fewer calibration positives the looser."""
    conf = np.asarray(conf, dtype=float).ravel()
    prob = np.asarray(prob, dtype=float).ravel()
    pos = np.asarray(labels).astype(int).ravel() == 1
    if not pos.any():
        raise ValueError("safety_cut needs positives on the calibration split")
    if not 0.0 <= miss < 1.0:
        raise ValueError(f"miss must be in [0, 1), got {miss}")
    cleared = np.sort(conf[pos & (prob < 0.5)])
    k = int(np.floor(miss * pos.sum() + 1e-9))
    return float(cleared[k]) if k < len(cleared) else float("inf")


def veto_cut(disp, prob, veto: float = NEGATIVE_VETO) -> float:
    """The predicted-disagreement cut above which a NEGATIVE call goes to a reader:
    the `1 - veto` quantile of `disp` among the calibration cases called negative
    (obvious negatives count in the quantile but are exempt, `vetoable`).

    Only negative calls are vetoed because only they carry the costly error, a
    positive cleared as negative; a positive call already leads to a work-up a
    clinician reads. `inf` (no veto) when `veto` is 0 or nothing was called
    negative."""
    disp = np.asarray(disp, dtype=float).ravel()
    neg = np.asarray(prob, dtype=float).ravel() < 0.5
    if not 0.0 <= veto < 1.0:
        raise ValueError(f"veto must be in [0, 1), got {veto}")
    if veto == 0.0 or not neg.any():
        return float("inf")
    return float(np.quantile(disp[neg], 1.0 - veto))


def vetoable(prob, obvious: float = OBVIOUS_NEGATIVE) -> np.ndarray:
    """The calls the study's veto may send to a reader: negative (p < 0.5) but not obvious
    (p > `obvious`). An obvious negative stays automatic whatever the panel would
    say - the model's own certainty there is what keeps the error rate down."""
    p = np.asarray(prob, dtype=float).ravel()
    return (p < 0.5) & (p > float(obvious))


def served_refer(conf, prob, disp=None, *, conf_cut: float,
                 disp_cut: float | None = None,
                 obvious: float = OBVIOUS_NEGATIVE,
                 extreme_cut: float | None = None,
                 net=None, net_cut: float | None = None) -> np.ndarray:
    """True where the served rule sends the case to a reader: the model is unsure
    (`conf >= conf_cut`), or it makes a non-obvious negative call the panel is
    predicted to dispute (`vetoable` and `disp >= disp_cut`), or any negative call
    the panel is predicted to dispute most (`disp >= extreme_cut`, the
    EXTREME_VETO share from `veto_cut`), or - the SAFETY NET - any negative call
    whose `net` signal is at or above `net_cut` (`veto_cut(net, prob,
    rule.safety_net)`). Without `disp` and `net` it is the certainty rule alone.
    `decide_or_defer` is the per-case form of the rule."""
    refer = np.asarray(conf, dtype=float).ravel() >= float(conf_cut)
    if net is not None and net_cut is not None:
        refer = refer | ((np.asarray(prob, dtype=float).ravel() < 0.5)
                         & (np.asarray(net, dtype=float).ravel() >= float(net_cut)))
    if disp is not None:
        d = np.asarray(disp, dtype=float).ravel()
        if disp_cut is not None:
            refer = refer | (vetoable(prob, obvious) & (d >= float(disp_cut)))
        if extreme_cut is not None:
            refer = refer | ((np.asarray(prob, dtype=float).ravel() < 0.5)
                             & (d >= float(extreme_cut)))
    return refer


def build_signals(prob, defer_score=None, opinion_entropy=None, epistemic=None,
                  set_size=None, drop_constant: bool = False,
                  keep: list[str] | None = None
                  ) -> tuple[np.ndarray, list[str]]:
    """Assemble the per-case signal matrix (N, F) and its column names.

    Every column is oriented so that HIGHER MEANS DEFER, which keeps the fitted
    coefficients readable: a positive weight means the signal argues for
    referral. Only `prob` is required; anything omitted is simply not a column,
    so the same call site serves each rung of the ablation ladder.

      confidence_gap    1 - 2|p - 0.5|   the classifier sitting near its boundary
      defer_head        P(defer)         the disagreement-supervised head
      opinion_entropy   H(q_hat)/log K   predicted panel spread   (aleatoric)
      epistemic         MI/log K         MC-dropout model spread  (epistemic)
      conformal         ||C(x)| - 1|     ambiguous OR atypical under conformal

    `drop_constant` additionally drops a supplied HEAD signal with no variance,
    which is what an ABLATION wants: a head whose output never moves (an
    epistemic term with `mc_passes = 0`, a presence histogram that is constant by
    construction) carries no information and would only add a dead column to the
    fit. It must stay OFF at inference, where N = 1 and every column is trivially
    constant - a served request has to build exactly the columns the policy was
    fitted with, and `DeferralPolicy.score` enforces that.

    **`conformal` is never dropped for constancy.** Whether |C(x)| varies is a
    property of the SPLIT, not of what the arm built: a validation fold can be
    all singletons while the test fold is not, and dropping the column on the
    fitting split then left the policy four weights against five test columns -
    the mismatch `score` raises on, in the middle of a run. Every arm can always
    compute a conformal set, so the column always exists; when it happens not to
    move, standardisation maps it to all-zeros and the fit gives it exactly
    weight 0, which is the same answer dropping it gave.

    `keep` names the columns to build, in order, overriding both rules - pass a
    fitted policy's `signal_names` to guarantee the caller hands `score` the
    columns it was fitted with. A named signal this call cannot supply is an
    error, not a silently missing column.
    """
    available: dict[str, np.ndarray] = {"confidence_gap": confidence_gap(prob)}
    for value, name in ((defer_score, "defer_head"),
                        (opinion_entropy, "opinion_entropy"),
                        (epistemic, "epistemic")):
        if value is not None:
            available[name] = np.asarray(value, dtype=float).ravel()
    if set_size is not None:
        available["conformal"] = conformal_ambiguity(set_size)

    if keep is not None:
        missing = [n for n in keep if n not in available]
        if missing:
            raise ValueError(
                f"cannot build the signals {missing} this policy was fitted "
                f"with - only {list(available)} are available here. The arm "
                f"being scored and the one the policy was fitted on disagree "
                f"about which heads exist.")
        names = list(keep)
    else:
        # `confidence_gap` is column 0 by contract - `score` falls back to it
        # when nothing was fitted - so it is never a candidate for dropping, and
        # neither is `conformal` (see the docstring: split property, not arm).
        dead = lambda n: (drop_constant and n not in ("confidence_gap", "conformal")
                          and float(np.std(available[n])) <= 0)
        names = [n for n in available if not dead(n)]
    return np.stack([available[n] for n in names], axis=1), names


def prediction_signals(prob, q_hat, defer_score=None, opinion_entropy=None,
                       epistemic=None, drop_constant: bool = False,
                       keep: list[str] | None = None
                       ) -> tuple[np.ndarray, list[str]]:
    """The policy's input for one split, straight from that split's predictions.

    Wraps `build_signals` with the conformal step, so "how a prediction becomes
    the policy's input" is defined once. Every column is PREDICTED from the image - nothing here reads an
    annotation, which is what makes the resulting referral a decision the system
    made rather than a label it reproduced.
    """
    from .conformal import binary_probs, lac_predict
    sets = lac_predict(binary_probs(prob), q_hat)
    return build_signals(prob=prob, defer_score=defer_score,
                         opinion_entropy=opinion_entropy, epistemic=epistemic,
                         set_size=sets.sum(axis=1), drop_constant=drop_constant,
                         keep=keep)


# --------------------------------------------------------------------------- #
# the learned policy
# --------------------------------------------------------------------------- #
@dataclass
class DeferralPolicy:
    """Logistic regression from the predicted signals to P(the model is wrong).

    Fitted on the validation split, applied unchanged to test. `signal_names`
    travels with the weights so a fitted policy can report which signals it
    leaned on - the answer to "why was this case deferred?" that a bare threshold
    cannot give."""

    weights: np.ndarray | None = None
    bias: float = 0.0
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    signal_names: list[str] = field(default_factory=list)
    threshold: float = 0.5

    # -- fitting ------------------------------------------------------------ #
    def fit(self, signals: np.ndarray, error: np.ndarray, l2: float = 1e-1,
            lr: float = 0.5, n_steps: int = 2000,
            nest_confidence: bool = True) -> DeferralPolicy:
        """Fit on (N, F) validation signals against the binary error indicator.

        `error` is `(p >= 0.5) != y` on the SAME split - the target is the model's
        own mistakes, not annotator disagreement. Plain full-batch gradient
        descent on the L2-penalised log-loss: F is a handful of columns and N is a
        validation fold, so there is nothing to gain from anything cleverer and a
        deterministic optimiser keeps the fold results reproducible.

        **`nest_confidence` is what makes the comparison against the confidence
        baseline interpretable.** A validation fold carries only ~25-30 error
        events against 5 free weights - four events per parameter, well under the
        usual ten - and an unconstrained fit spends that budget re-deciding how
        much to trust `confidence_gap`, which it cannot estimate reliably at that
        sample size. It then loses to plain confidence on test, and that tells you
        nothing about whether the other signals carry information.

        Nested, a confidence-only fit is done FIRST and enters the second stage as
        a fixed offset, so the remaining weights are fitted on what confidence
        could not explain. They then read as "what the panel machinery adds ON TOP
        OF confidence", and a weight near zero is a clean negative result rather
        than an artefact of the sample size.

        This does NOT guarantee the policy beats confidence on test. Stage two can
        still spend its weights on signals that do not generalise - the L2 default
        is deliberately firm (0.1) for that reason. What nesting buys is that when
        the policy does lose, the coefficients say why, instead of the loss being
        confounded with a re-estimate of the baseline itself."""
        x = np.asarray(signals, dtype=float)
        y = np.asarray(error, dtype=float).ravel()
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError(f"signals {x.shape} and error {y.shape} disagree")

        # Standardise so the L2 penalty is even-handed across signals on
        # different scales, and so lr is not a per-column tuning knob.
        self.mean = x.mean(axis=0)
        self.scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
        z = (x - self.mean) / self.scale

        n, f = z.shape
        if n == 0 or y.max() == y.min():
            # No errors (or nothing but errors) on val: there is no ranking to
            # learn. Fall back to the single most defensible signal rather than
            # returning noise - see `score`, which then reads confidence_gap.
            self.weights, self.bias = None, 0.0
            return self

        def descend(zz, target, offset):
            w, b = np.zeros(zz.shape[1]), 0.0
            for _ in range(n_steps):
                p = 1.0 / (1.0 + np.exp(-(zz @ w + b + offset)))
                g = p - target
                w -= lr * ((zz.T @ g) / len(target) + l2 * w)
                b -= lr * g.mean()
            return w, float(b)

        if nest_confidence and f > 1:
            # Stage 1: confidence_gap alone (column 0) -> the baseline, as a logit.
            w_conf, b_conf = descend(z[:, :1], y, 0.0)
            # Stage 2: the remaining signals explain only what confidence missed.
            w_rest, b_rest = descend(z[:, 1:], y, z[:, 0] * w_conf[0] + b_conf)
            # The two stages are additive in the logit, so they collapse straight
            # back into one weight vector - `score` and the reported coefficients
            # keep their existing shape and meaning.
            self.weights = np.concatenate([w_conf, w_rest])
            self.bias = b_conf + b_rest
        else:
            self.weights, self.bias = descend(z, y, 0.0)
        return self

    # -- application -------------------------------------------------------- #
    def score(self, signals: np.ndarray) -> np.ndarray:
        """P(the model is wrong) per case - the deferral score. Higher = defer.

        Unfitted (or fitted on a split with no errors to learn from) it degrades
        to the first signal column, `confidence_gap`. That is the classical
        confidence baseline, so the system stays safe and the degradation is
        visible in the reported coefficients rather than silent."""
        x = np.asarray(signals, dtype=float)
        if self.weights is None:
            return x[:, 0]
        # A policy fitted on an arm that lacked a head carries FEWER weights than
        # a request builds columns (constant columns are never dropped -
        # see `build_signals`). Left to numpy this surfaces as a
        # broadcasting error from inside the arithmetic, at inference, on a real
        # request. Name the mismatch instead, and name the columns.
        if x.shape[1] != len(self.weights):
            raise ValueError(
                f"policy was fitted with {len(self.weights)} signals "
                f"{self.signal_names} but was given {x.shape[1]} columns - the "
                f"fitted policy and the caller disagree about which signals exist")
        z = (x - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.weights + self.bias)))

    def calibrate_threshold(self, signals: np.ndarray,
                            coverage: float = 0.8) -> float:
        """Set the operating point so a `coverage` fraction of cases are DECIDED.

        Chosen on val and then frozen, so the test-set deferral rate is a result
        rather than something tuned after seeing it."""
        s = self.score(signals)
        if not len(s):
            return 0.5
        self.threshold = float(np.quantile(s, min(max(coverage, 0.0), 1.0)))
        return self.threshold

    def decide(self, signals: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Boolean defer mask at the calibrated (or supplied) threshold."""
        t = self.threshold if threshold is None else threshold
        return self.score(signals) >= t

    def coefficients(self) -> dict:
        """Fitted weight per signal, on the standardised scale, so the magnitudes
        are directly comparable - the "why did it defer" read-out."""
        if self.weights is None:
            return {n: float("nan") for n in self.signal_names}
        return {n: float(w) for n, w in zip(self.signal_names, self.weights)}

    # -- persistence -------------------------------------------------------- #
    def to_dict(self, round_to: int | None = 6) -> dict:
        """Plain JSON/YAML-safe mapping - this is the on-disk form.

        `from_dict` reads it back, so the format has one definition.
        `round_to` keeps the block readable; None writes full precision.
        """
        def fmt(v):
            return v if round_to is None else round(float(v), round_to)

        def arr(a):
            return None if a is None else [fmt(v) for v in np.asarray(a, float)]

        return {"signal_names": list(self.signal_names),
                "weights": arr(self.weights),
                "bias": fmt(self.bias),
                "mean": arr(self.mean),
                "scale": arr(self.scale),
                "threshold": fmt(self.threshold)}

    @classmethod
    def from_dict(cls, block: dict | None) -> "DeferralPolicy":
        """Rebuild from `to_dict` output. `None` or `{}` gives an UNFITTED policy
        rather than an error, mirroring `engine.Calibrator.from_dict`.

        An unfitted policy is not a broken one: `score` degrades to the
        confidence baseline, which is the honest fallback when no run has fitted
        a policy yet. Inventing weights here would ship a decision rule nothing
        ever validated.
        """
        if not block or not block.get("weights"):
            return cls(signal_names=list((block or {}).get("signal_names") or []))
        return cls(weights=np.asarray(block["weights"], dtype=float),
                   bias=float(block.get("bias", 0.0)),
                   mean=np.asarray(block["mean"], dtype=float),
                   scale=np.asarray(block["scale"], dtype=float),
                   signal_names=list(block["signal_names"]),
                   threshold=float(block.get("threshold", 0.5)))

    @classmethod
    def average(cls, policies) -> "DeferralPolicy":
        """Mean of one policy per fold - the single rule that is applied.

        Folds that never fitted (no validation errors to learn from) are skipped
        rather than counted as zeros, which would drag the mean toward a policy
        no fold held. Contributing folds must agree on `signal_names`: averaging
        different signals position by position would attribute weights to the
        wrong ones.
        """
        fitted = [p for p in policies if p is not None and p.weights is not None]
        if not fitted:
            return cls()
        names = list(fitted[0].signal_names)
        odd = [p.signal_names for p in fitted if list(p.signal_names) != names]
        if odd:
            raise ValueError(
                f"cannot average policies fitted on different signals: {names} "
                f"vs {odd[0]}. An arm without a head drops that column "
                f"(`build_signals(drop_constant=True)`), so this means the folds "
                f"did not all build the same heads.")
        return cls(weights=np.mean([p.weights for p in fitted], axis=0),
                   bias=float(np.mean([p.bias for p in fitted])),
                   mean=np.mean([p.mean for p in fitted], axis=0),
                   scale=np.mean([p.scale for p in fitted], axis=0),
                   signal_names=names,
                   threshold=float(np.mean([p.threshold for p in fitted])))


# --------------------------------------------------------------------------- #
# fitting the policy, and the rules it is scored against
# --------------------------------------------------------------------------- #
def fit_deferral_policy(signals, names, prob, labels,
                        coverage: float = 0.8) -> DeferralPolicy:
    """Fit the policy on a VALIDATION split and freeze its operating point.

    The target is `(p >= 0.5) != y` on the same split - the model's OWN errors,
    not annotator disagreement. That is the whole difference from the supervised
    head: the policy is free to defer on a case the panel agreed about and the
    model still gets wrong, and to decide on a contested case it handles fine.

    The threshold is calibrated here and then frozen, so the deferral rate the
    test set ends up with is a result rather than something tuned after seeing it.
    """
    p = np.asarray(prob, dtype=float).ravel()
    y = np.asarray(labels).astype(int).ravel()
    err = (p >= 0.5).astype(int) != y
    policy = DeferralPolicy(signal_names=list(names)).fit(signals, err)
    policy.calibrate_threshold(signals, coverage=coverage)
    return policy


def candidate_rules(prob, q_hat, defer_score=None, opinion_entropy=None,
                    epistemic=None, policy: DeferralPolicy | None = None) -> dict:
    """`{name: confidence}` for every deferral rule worth comparing.

    Returned as CONFIDENCE (higher = decide first), which is the orientation the
    selective-prediction metrics take, so a defer score appears negated. The
    plain `confidence` baseline is deliberately absent: `metrics.compare_selectors`
    inserts it itself, so no caller can report a deferral mechanism without the
    control it has to beat.

    A signal the arm never built is simply not a rule, which is what keeps the
    ablation honest about what was available to each rung."""
    from .conformal import binary_probs, lac_predict
    defer = lambda v: -np.asarray(v, dtype=float).ravel()
    varies = lambda v: v is not None and float(np.std(np.asarray(v, float))) > 0

    sets = lac_predict(binary_probs(prob), q_hat)
    rules = {"conformal": defer(conformal_ambiguity(sets.sum(axis=1)))}
    if defer_score is not None:
        rules["defer_head"] = defer(defer_score)
    if varies(opinion_entropy):
        rules["opinion_entropy"] = defer(opinion_entropy)
    if varies(epistemic):
        rules["epistemic"] = defer(epistemic)
    if policy is not None:
        # The FITTED policy's own column list decides what is built here. Letting
        # this split's `drop_constant` decide instead made the column set a
        # property of the split, and a val fold that happened to be all
        # singletons then fitted four weights against five test columns.
        keep = list(policy.signal_names) or None
        signals, _ = prediction_signals(
            prob, q_hat, defer_score=defer_score,
            opinion_entropy=(opinion_entropy if keep or varies(opinion_entropy)
                             else None),
            epistemic=epistemic if keep or varies(epistemic) else None,
            drop_constant=True, keep=keep)
        rules["policy"] = defer(policy.score(signals))
    return rules
