"""`decide_or_defer` — the deployed decision path, as one framework call.

This is the inference façade of the framework. Everything it needs already
existed in `field_disagreement.triage`; what did not exist was a single place where the
pieces are composed in the right order, so the composition lived in the serving
layer and could only be exercised through an HTTP client.

The path, in order, and why that order:

    pool over members   fold ensemble (and MC-dropout samples) -> one prediction,
                        keeping the SPREAD, because the spread is the epistemic term
    uncertainty         mutual-information split of the opinion distribution
                        -> (aleatoric, epistemic)
    conformal set       C(x) at the calibrated risk level -> ambiguous OR atypical
    signals             every quantity above, oriented so higher means defer
    policy              the fitted logistic rule -> decide or defer
    empty-set override  an empty C(x) forces a deferral whatever the rule said

Every input to the decision is PREDICTED from the case at hand. Nothing here
reads an annotation, an agreement surface, or a label — which is what makes the
referral a judgement the system made rather than a training target it reproduced,
and what lets this function run on an uploaded image that carries no ground truth.

The function is pure: arrays in, a `Decision` out, no I/O and no torch. A caller
that wants pictures, PNGs or an HTTP body builds them from the `Decision`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..triage.deferral import (OBVIOUS_NEGATIVE, DeferralPolicy, build_signals,
                               confidence_gap)
from ..triage.uncertainty import entropy_decomposition
from .calibration import Calibrator

DECIDE = "decide"
DEFER = "defer"
# The study's served rung: the certainty cut, plus a veto on the negative calls the panel
# is predicted to dispute (`triage.safety_cut`, `triage.veto_cut`).
VETO_RULE = "confidence+negative_veto"


@dataclass(frozen=True)
class SignalContribution:
    """One row of the "why was this deferred?" read-out.

    `weight` is the fitted coefficient on the standardised scale, or None when no
    policy is in play — a value with no weight beside it is context, not a reason,
    and the distinction should survive into the response.
    """
    name: str
    value: float
    weight: float | None = None


# eq=False: the array fields would make a generated __eq__ return an array and a
# generated __hash__ raise, both at the call site rather than here.
@dataclass(frozen=True, eq=False)
class Prediction:
    """What the model said about ONE case, pooled over whatever members exist.

    `prob` is the positive-class probability. The rest are optional because the
    ablation ladder is real: an arm trained without an opinion head has no panel
    spread to report, and the decision has to degrade to what it does have rather
    than fabricate the missing column.
    """
    prob: float
    opinion: np.ndarray | None = None
    defer_probs: np.ndarray | None = None
    aleatoric: float | None = None
    epistemic: float | None = None
    n_members: int = 1
    # The study's predicted panel disagreement: one value per report field (or one).
    dispersion: np.ndarray | None = None

    @property
    def panel_disp(self) -> float | None:
        """The study's deferral score: the predicted disagreement, averaged over the
        report's fields - the number the router ranks cases on."""
        if self.dispersion is None or not np.size(self.dispersion):
            return None
        return float(np.mean(self.dispersion))

    @property
    def defer_score(self) -> float | None:
        """P(defer) from the disagreement-supervised head.

        The LAST column by construction: `triage.deferral.build_deferral_targets`
        assigns index K to defer and leaves 0..K-1 as the class labels, so the
        defer logit is always the trailing one however many classes there are.
        """
        if self.defer_probs is None or not len(self.defer_probs):
            return None
        return float(np.asarray(self.defer_probs, dtype=float).ravel()[-1])

    @classmethod
    def pool(cls, probs, opinions=None, defer_probs=None,
             members=None, dispersions=None) -> "Prediction":
        """Collapse the per-member stacks into one prediction, keeping the spread.

        `probs` (M,), `opinions` (M, K), `defer_probs` (M, D) and `dispersions`
        (M, P) are the per-member outputs — one row per fold of the deployed
        ensemble. They are averaged.

        The uncertainty split is taken over `members` when given, otherwise over
        `opinions`. That parameter exists for MC dropout: T stochastic passes
        pooled across folds is a better spread estimate than the folds alone, but
        pooling BOTH would count each fold twice. It is also what lets a single
        loaded checkpoint still report an epistemic term — a fold ensemble needs
        the whole run to exist, MC dropout needs only the one model.

        With one member and no sampling, the epistemic term is exactly 0 when an
        opinion head exists (one row has no spread) and `n_members` is 1, which is
        how a caller tells "no spread" from "measured small spread". With neither
        opinions nor sampled `members` both terms are None: not estimable, which
        must not be read as zero uncertainty.
        """
        p = np.asarray(probs, dtype=float).ravel()
        if not p.size:
            raise ValueError("pool() needs at least one member probability")

        opinion = (None if opinions is None
                   else np.atleast_2d(np.asarray(opinions, dtype=float)).mean(axis=0))
        defer = (None if defer_probs is None
                 else np.atleast_2d(np.asarray(defer_probs, dtype=float)).mean(axis=0))
        disp = (None if dispersions is None
                else np.atleast_2d(np.asarray(dispersions, dtype=float)).mean(axis=0))

        spread = members if members is not None else opinions
        aleatoric = epistemic = None
        n_members = int(p.size)
        if spread is not None:
            stack = np.atleast_2d(np.asarray(spread, dtype=float))
            u = entropy_decomposition(stack, normalize=True)
            aleatoric, epistemic = float(u["aleatoric"]), float(u["epistemic"])
            n_members = int(len(stack))

        return cls(prob=float(p.mean()), opinion=opinion, defer_probs=defer,
                   aleatoric=aleatoric, epistemic=epistemic, n_members=n_members,
                   dispersion=disp)


@dataclass(frozen=True)
class Decision:
    """The outcome, with everything needed to justify it.

    `score >= threshold` is the referral test, `rule` names which mechanism was
    actually in force, and `signals` says what each input contributed. A verdict
    without those three is a number a clinician has no way to interrogate.
    """
    verdict: str
    rule: str
    score: float
    threshold: float
    label: str
    prediction_set: list[str]
    signals: list[SignalContribution] = field(default_factory=list)
    forced_by_empty_set: bool = False
    alpha: float | None = None
    # which of the served rule's two tests referred the case, as reason codes
    triggers: tuple[str, ...] = ()

    @property
    def deferred(self) -> bool:
        return self.verdict == DEFER

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "rule": self.rule,
                "score": self.score, "threshold": self.threshold,
                "label": self.label, "prediction_set": list(self.prediction_set),
                "signals": [{"name": s.name, "value": s.value, "weight": s.weight}
                            for s in self.signals],
                "forced_by_empty_set": self.forced_by_empty_set,
                "alpha": self.alpha, "triggers": list(self.triggers)}


def decide_or_defer(prediction: Prediction, *,
                    calibrator: Calibrator | None = None,
                    policy: DeferralPolicy | None = None,
                    threshold: float = 0.5,
                    disp_threshold: float | None = None,
                    conf_threshold: float | None = None,
                    extreme_threshold: float | None = None) -> Decision:
    """One prediction -> decide or defer, with the reasons attached.

    `calibrator` supplies C(x) and the class names; an unfitted or absent one
    still works and degrades to the argmax singleton. `policy` is the fitted
    `DeferralPolicy` — when it is absent the decision falls back down a ladder:

        policy        the fitted rule, the thing the study evaluated
        confidence+negative_veto
                      the study's SERVED rule, when both `disp_threshold` and
                      `conf_threshold` are given and the model predicts
                      disagreement. Refer when the classifier is unsure
                      (`1 - 2|p - 0.5|` at or above `conf_threshold`, the cut
                      `triage.safety_cut` sets from a missed-positive limit), OR
                      when it calls the case negative but not obviously so
                      (OBVIOUS_NEGATIVE < p < 0.5) and the panel is predicted to
                      dispute it (`panel_disp` at or above `disp_threshold`, from
                      `triage.veto_cut`). Only the negative call is vetoed: it is
                      the call whose error is a missed positive; an obvious one
                      stays automatic unless `panel_disp` is at or above
                      `extreme_threshold` (the EXTREME_VETO share, the panel at its
                      most divided). `Decision.triggers` says which test fired.
        panel_disp    the study's predicted panel disagreement alone at `disp_threshold`
                      (no `conf_threshold`): the disagreement-only rung the
                      evaluation's `deferral_rate` reads. Not a served rule.
        confidence    with `conf_threshold` and no disagreement head: the
                      certainty cut alone, `1 - 2|p - 0.5|` at `conf_threshold`
        defer_head    the supervised head's own P(defer) at `threshold`
        confidence    1 - 2|p - 0.5|, the classifier sitting near its boundary

    Each rung is weaker than the one above and `Decision.rule` says which one ran.
    Inventing policy weights when none were fitted would produce a decision rule
    nothing ever validated, so the fallback is deliberate rather than an error.

    An EMPTY conformal set overrides every rung. Reporting `decide` next to a set
    containing no plausible label is a contradiction the reader would have to
    resolve, so it is resolved here and flagged with `forced_by_empty_set`.
    """
    cal = calibrator if calibrator is not None else Calibrator()
    cset = cal.prediction_set(prediction.prob)

    # Only signals the arm actually produced become columns — see `build_signals`.
    # `drop_constant` stays OFF: N=1 here, so every column is trivially constant
    # and dropping them would leave the policy with nothing to score.
    #
    # `keep` is what makes the FITTED POLICY authoritative about the columns: it
    # selects exactly its own signals, in exactly its own order, and names
    # anything this prediction cannot supply. A policy fitted on an arm that
    # lacked a head otherwise carries fewer weights than a served request builds
    # columns, and left to positional luck that surfaces as a broadcasting error
    # inside the arithmetic, on a real request.
    signals, names = build_signals(
        prob=[prediction.prob],
        defer_score=(None if prediction.defer_score is None
                     else [prediction.defer_score]),
        opinion_entropy=(None if prediction.aleatoric is None
                         else [prediction.aleatoric]),
        epistemic=(None if prediction.epistemic is None
                   else [prediction.epistemic]),
        set_size=[len(cset)],
        keep=(list(policy.signal_names)
              if policy is not None and policy.signal_names else None))

    weights: dict[str, float] = {}
    if policy is not None:
        score = float(policy.score(signals)[0])
        cut, rule = float(policy.threshold), "policy"
        weights = policy.coefficients()
    elif disp_threshold is not None and prediction.panel_disp is not None:
        if conf_threshold is None:
            score, cut, rule = prediction.panel_disp, float(disp_threshold), "panel_disp"
        else:
            score = float(confidence_gap([prediction.prob])[0])
            cut, rule = float(conf_threshold), VETO_RULE
    elif conf_threshold is not None:
        score = float(confidence_gap([prediction.prob])[0])
        cut, rule = float(conf_threshold), "confidence"
    elif prediction.defer_score is not None:
        score, cut, rule = prediction.defer_score, float(threshold), "defer_head"
    else:
        score, cut, rule = float(signals[0, 0]), float(threshold), "confidence"

    refer = score >= cut
    triggers: tuple[str, ...] = ()
    contributions = [SignalContribution(n, float(v), weights.get(n))
                     for n, v in zip(names, signals[0])]
    if rule == VETO_RULE:
        vetoed = prediction.prob < 0.5 and (
            (prediction.prob > OBVIOUS_NEGATIVE
             and prediction.panel_disp >= float(disp_threshold))
            or (extreme_threshold is not None
                and prediction.panel_disp >= float(extreme_threshold)))
        triggers = tuple(code for code, hit in (("LOW_CONFIDENCE", refer),
                                                ("PANEL_WOULD_DISPUTE_NEGATIVE", vetoed))
                         if hit)
        refer = refer or vetoed
        contributions.append(SignalContribution("panel_disp", prediction.panel_disp))
    return Decision(
        verdict=DEFER if (refer or not cset) else DECIDE,
        rule=rule, score=score, threshold=cut,
        label=cal.label_for(prediction.prob),
        prediction_set=cset,
        signals=contributions,
        forced_by_empty_set=(not cset) and not refer,
        alpha=cal.alpha if cal else None,
        triggers=triggers,
    )


def decide_batch(pred: dict, *, calibrator: Calibrator | None = None,
                 policy: DeferralPolicy | None = None,
                 threshold: float = 0.5,
                 disp_threshold: float | None = None,
                 conf_threshold: float | None = None,
                 extreme_threshold: float | None = None) -> dict:
    """`decide_or_defer` over a whole split of already-computed predictions.

    A study evaluating a deferral rule has the forward pass done - it needs the
    VERDICTS, not another inference. Scoring the rule by hand
    (`policy.decide(...)`) would skip the empty-set override; this runs the full
    decision per case.

    `pred` is a dict of per-case arrays: `probs` and, where the arm built the
    head, `opinion`, `defer`, `dispersion`, `aleatoric`, `epistemic`. Heads the arm never
    built are simply absent and do not become signals, so the thinner rungs of
    an ablation run through unchanged.

    Returns arrays aligned with `pred["probs"]`, plus the single `rule` in force
    (a property of the fitted artifacts, not of the case).
    """
    probs = np.asarray(pred["probs"], dtype=float).ravel()

    def column(key):
        v = pred.get(key)
        return None if v is None else np.asarray(v, dtype=float)

    opinion, defer = column("opinion"), column("defer")
    disp = column("dispersion")
    if disp is not None:
        disp = disp.reshape(len(probs), -1)
    aleatoric, epistemic = column("aleatoric"), column("epistemic")

    decisions = [
        decide_or_defer(
            Prediction(
                prob=float(p),
                opinion=None if opinion is None else opinion[i],
                defer_probs=None if defer is None else defer[i],
                aleatoric=None if aleatoric is None else float(aleatoric[i]),
                epistemic=None if epistemic is None else float(epistemic[i]),
                dispersion=None if disp is None else disp[i]),
            calibrator=calibrator, policy=policy, threshold=threshold,
            disp_threshold=disp_threshold, conf_threshold=conf_threshold,
            extreme_threshold=extreme_threshold)
        for i, p in enumerate(probs)]

    rules = {d.rule for d in decisions}
    return {
        "deferred": np.array([d.deferred for d in decisions], dtype=bool),
        "score": np.array([d.score for d in decisions], dtype=float),
        "set_size": np.array([len(d.prediction_set) for d in decisions], dtype=int),
        "forced_by_empty_set": np.array([d.forced_by_empty_set for d in decisions],
                                        dtype=bool),
        # One rung is in force for the whole split; a set of size > 1 cannot
        # happen and is worth seeing rather than silently collapsing.
        "rule": rules.pop() if len(rules) == 1 else sorted(rules),
        "n": len(decisions),
    }
