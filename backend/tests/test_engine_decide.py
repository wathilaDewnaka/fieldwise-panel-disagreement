"""`field_disagreement.engine` - the decision for one case, as a pure function.

The rules that matter clinically (an empty conformal set forces a referral; an
uncalibrated threshold never implies a guarantee; the fallback ladder degrades in
a stated order) are asserted directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from field_disagreement import Calibrator, Prediction, decide_or_defer
from field_disagreement.engine.decide import VETO_RULE
from field_disagreement.triage import DeferralPolicy
from field_disagreement.triage.conformal import binary_probs, lac_calibrate

CLASSES = ("negative", "positive")


def _policy(names, weights, threshold=0.5, bias=0.0):
    """A policy with KNOWN weights, so the score is arithmetic rather than a fit."""
    return DeferralPolicy(
        weights=np.asarray(weights, dtype=float), bias=bias,
        mean=np.zeros(len(names)), scale=np.ones(len(names)),
        signal_names=list(names), threshold=threshold)


# --------------------------------------------------------------------------- #
# Calibrator
# --------------------------------------------------------------------------- #
def test_an_uncalibrated_threshold_gives_a_singleton_not_a_guarantee():
    cal = Calibrator(class_names=CLASSES)
    assert not cal, "an unfitted calibrator must be falsy"
    assert cal.prediction_set(0.55) == ["positive"]
    assert cal.prediction_set(0.45) == ["negative"]


def test_the_set_widens_and_empties_as_the_threshold_moves():
    wide = Calibrator(q_hat=0.9, class_names=CLASSES)
    tight = Calibrator(q_hat=0.35, class_names=CLASSES)
    assert wide.prediction_set(0.55) == ["negative", "positive"]
    assert tight.prediction_set(0.55) == []    # neither label reaches 1 - q_hat


def test_fit_matches_the_underlying_lac_calibration():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 200)
    labels = (rng.uniform(0, 1, 200) < probs).astype(int)

    cal = Calibrator.fit(probs, labels, alpha=0.1, class_names=CLASSES)
    assert cal.q_hat == pytest.approx(
        lac_calibrate(binary_probs(probs), labels, alpha=0.1))
    assert cal.n_calibration == 200 and cal.alpha == 0.1
    assert cal, "a fitted calibrator must be truthy"


def test_a_fitted_calibrator_survives_a_round_trip_to_disk_form():
    cal = Calibrator(q_hat=0.42, alpha=0.05, class_names=CLASSES,
                     n_calibration=123, empirical_coverage=0.96)
    back = Calibrator.from_dict(cal.to_dict())
    assert (back.q_hat, back.alpha, back.class_names) == (0.42, 0.05, CLASSES)
    assert back.n_calibration == 123 and back.empirical_coverage == 0.96


def test_an_absent_calibration_block_loads_as_unfitted_rather_than_raising():
    # A decision can be built before a run has produced a calibration.
    assert Calibrator.from_dict(None).q_hat is None
    assert Calibrator.from_dict({}).q_hat is None


def test_fit_rejects_a_class_name_count_that_cannot_be_right():
    with pytest.raises(ValueError, match="class names"):
        Calibrator.fit([0.2, 0.8], [0, 1], class_names=("a", "b", "c"))


# --------------------------------------------------------------------------- #
# pooling and the uncertainty split
# --------------------------------------------------------------------------- #
def test_pool_averages_the_members():
    pred = Prediction.pool([0.2, 0.4, 0.6],
                           opinions=[[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]],
                           defer_probs=[[0.1, 0.2, 0.7], [0.3, 0.2, 0.5]])
    assert pred.prob == pytest.approx(0.4)
    assert pred.opinion == pytest.approx([0.5, 0, 0, 0, 0.5])
    assert pred.defer_score == pytest.approx(0.6)   # the LAST defer column


@pytest.mark.parametrize("opinions, aleatoric, epistemic", [
    ([[0.2] * 5, [0.2] * 5], 1.0, 0.0),              # agreed-on maximal panel spread
    ([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], 0.0, 0.0),  # agreed-on certainty
    ([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]], 0.0, 0.43),  # each member certain, of opposites
])
def test_pooling_splits_panel_spread_from_ensemble_spread(opinions, aleatoric, epistemic):
    pred = Prediction.pool([0.5] * len(opinions), opinions=opinions)
    assert pred.aleatoric == pytest.approx(aleatoric, abs=1e-2)
    assert pred.epistemic == pytest.approx(epistemic, abs=1e-2)
    assert pred.aleatoric >= 0.0 and pred.epistemic >= 0.0, \
        "a negative uncertainty is a float-error leak"


def test_one_member_has_no_epistemic_term_to_measure():
    pred = Prediction.pool([0.5], opinions=[[0.5, 0.5, 0, 0, 0]])
    assert pred.epistemic == 0.0
    assert pred.n_members == 1


def test_mc_samples_replace_the_folds_as_the_spread_estimate():
    """`members` is the whole reason a single checkpoint can report an epistemic
    term: the folds agree, the MC samples do not, and the answer follows the
    samples rather than silently pooling both."""
    folds = [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
    samples = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]]
    assert Prediction.pool([0.5, 0.5], opinions=folds).epistemic == 0.0
    mc = Prediction.pool([0.5, 0.5], opinions=folds, members=samples)
    assert mc.epistemic > 0.0 and mc.n_members == 3


def test_an_arm_without_the_extra_heads_reports_nothing_it_does_not_have():
    pred = Prediction.pool([0.7])
    assert pred.opinion is None and pred.defer_score is None
    assert pred.aleatoric is None and pred.epistemic is None


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #
def test_the_rule_ladder_degrades_in_the_stated_order():
    cal = Calibrator(class_names=CLASSES)
    full = Prediction.pool([0.6], opinions=[[0.2] * 5], defer_probs=[[0.1, 0.2, 0.7]])

    policy = _policy(["confidence_gap", "defer_head", "opinion_entropy",
                      "epistemic", "conformal"], [0, 0, 0, 0, 0])
    assert decide_or_defer(full, calibrator=cal, policy=policy).rule == "policy"
    assert decide_or_defer(full, calibrator=cal).rule == "defer_head"
    assert decide_or_defer(Prediction.pool([0.6]), calibrator=cal).rule == "confidence"


def test_the_defer_head_fallback_uses_the_supplied_threshold():
    cal = Calibrator(class_names=CLASSES)
    pred = Prediction.pool([0.6], opinions=[[0.2] * 5], defer_probs=[[0.1, 0.2, 0.7]])

    assert decide_or_defer(pred, calibrator=cal, threshold=0.5).verdict == "defer"
    decided = decide_or_defer(pred, calibrator=cal, threshold=0.8)
    assert decided.verdict == "decide" and decided.score == pytest.approx(0.7)


def test_an_empty_conformal_set_forces_a_deferral_the_rule_did_not_ask_for():
    tight = Calibrator(q_hat=0.05, class_names=CLASSES)
    pred = Prediction.pool([0.5], opinions=[[0.2] * 5], defer_probs=[[0.4, 0.4, 0.2]])

    d = decide_or_defer(pred, calibrator=tight, threshold=0.9)
    assert d.prediction_set == []
    assert d.verdict == "defer" and d.forced_by_empty_set
    assert d.score < d.threshold, "the rule itself wanted to decide"


def test_a_referral_the_rule_did_ask_for_is_not_flagged_as_forced():
    cal = Calibrator(q_hat=0.9, class_names=CLASSES)
    pred = Prediction.pool([0.5], opinions=[[0.2] * 5], defer_probs=[[0.1, 0.1, 0.8]])
    d = decide_or_defer(pred, calibrator=cal, threshold=0.5)
    assert d.verdict == "defer" and not d.forced_by_empty_set


def test_alpha_is_reported_only_when_a_calibration_stands_behind_it():
    pred = Prediction.pool([0.6], opinions=[[0.2] * 5], defer_probs=[[0.1, 0.2, 0.7]])
    assert decide_or_defer(pred, calibrator=Calibrator(alpha=0.1)).alpha is None
    assert decide_or_defer(
        pred, calibrator=Calibrator(q_hat=0.5, alpha=0.1)).alpha == 0.1


def test_the_signals_are_selected_and_ordered_to_match_the_fitted_policy():
    """A policy fitted on a subset, in its own order, must be scored on exactly
    that subset in exactly that order — positional luck would otherwise line the
    weights up against the wrong columns and still return a plausible number."""
    pred = Prediction.pool([0.6], opinions=[[0.2] * 5], defer_probs=[[0.1, 0.2, 0.7]])
    policy = _policy(["epistemic", "confidence_gap"], [0.0, 0.0])

    d = decide_or_defer(pred, calibrator=Calibrator(class_names=CLASSES), policy=policy)
    assert [s.name for s in d.signals] == ["epistemic", "confidence_gap"]
    assert d.signals[0].value == pytest.approx(pred.epistemic)
    assert d.signals[1].value == pytest.approx(1.0 - 2 * abs(0.6 - 0.5))


def test_a_policy_needing_a_head_the_arm_lacks_names_the_mismatch():
    bare = Prediction.pool([0.6])                    # no opinion head, no defer head
    policy = _policy(["confidence_gap", "defer_head"], [0.0, 0.0])
    with pytest.raises(ValueError, match="defer_head"):
        decide_or_defer(bare, calibrator=Calibrator(class_names=CLASSES), policy=policy)


def test_the_weights_travel_with_the_signals_as_the_why_readout():
    pred = Prediction.pool([0.6], opinions=[[0.2] * 5], defer_probs=[[0.1, 0.2, 0.7]])
    policy = _policy(["confidence_gap", "defer_head"], [1.5, -0.5])
    d = decide_or_defer(pred, calibrator=Calibrator(class_names=CLASSES), policy=policy)
    assert {s.name: s.weight for s in d.signals} == {"confidence_gap": 1.5,
                                                    "defer_head": -0.5}
    # Without a policy there are values but no weights: context, not reasons.
    plain = decide_or_defer(pred, calibrator=Calibrator(class_names=CLASSES))
    assert all(s.weight is None for s in plain.signals)


def test_the_decision_serialises_to_something_a_response_can_carry():
    pred = Prediction.pool([0.6], opinions=[[0.2] * 5], defer_probs=[[0.1, 0.2, 0.7]])
    body = decide_or_defer(pred, calibrator=Calibrator(class_names=CLASSES)).to_dict()
    assert body["verdict"] in ("decide", "defer")
    assert body["label"] == "positive"
    assert isinstance(body["signals"], list) and body["signals"][0]["name"]


# --------------------------------------------------------------------------- #
# The study's deferral rung: the predicted panel disagreement
# --------------------------------------------------------------------------- #
def test_without_a_disagreement_threshold_the_ladder_is_unchanged():
    """The rung is opt-in: every caller that passes no threshold (train_evaluate,
    every banked run) gets exactly the decision it got before."""
    for p in (0.1, 0.45, 0.55, 0.9):
        for defer in (None, np.array([0.2, 0.2, 0.6])):
            base = Prediction(prob=p, defer_probs=defer)
            with_disp = Prediction(prob=p, defer_probs=defer, dispersion=np.array([9.0, 9.0]))
            a, b = decide_or_defer(base), decide_or_defer(with_disp)
            assert (a.verdict, a.rule, a.score, a.threshold) == (
                b.verdict, b.rule, b.score, b.threshold)


def test_the_disagreement_rung_defers_on_the_mean_over_the_fields():
    cal = Calibrator(class_names=CLASSES)
    hi = Prediction(prob=0.95, dispersion=np.array([1.0, 0.5, 0.0]))   # mean 0.5
    lo = Prediction(prob=0.05, dispersion=np.array([-1.0, 0.0, 0.1]))
    d = decide_or_defer(hi, calibrator=cal, disp_threshold=0.4)
    assert d.rule == "panel_disp" and d.deferred and d.score == pytest.approx(0.5)
    assert not decide_or_defer(lo, calibrator=cal, disp_threshold=0.4).deferred
    # it outranks a 3-way head, and a fitted policy still outranks it
    both = Prediction(prob=0.95, defer_probs=np.array([0.9, 0.1, 0.0]),
                      dispersion=np.array([1.0]))
    assert decide_or_defer(both, disp_threshold=0.4).rule == "panel_disp"
    pol = _policy(["confidence_gap"], [1.0])
    assert decide_or_defer(both, policy=pol, disp_threshold=0.4).rule == "policy"


def test_pooling_averages_the_members_disagreement_per_field():
    pred = Prediction.pool([0.2, 0.4], dispersions=[[1.0, 3.0], [3.0, 5.0]])
    assert pred.dispersion.tolist() == [2.0, 4.0] and pred.panel_disp == pytest.approx(3.0)
    assert Prediction.pool([0.5]).panel_disp is None


def test_the_served_rule_vetoes_only_a_disputed_negative_call():
    """The study's served rule: the certainty cut, plus a veto on a non-obvious NEGATIVE
    call the panel is predicted to dispute. A sure positive call is fast-tracked and
    an obvious negative cleared even when disputed. Without `conf_threshold` the
    rung is the disagreement-only one."""
    cal = Calibrator(class_names=CLASSES)
    kw = dict(calibrator=cal, disp_threshold=0.5, conf_threshold=0.5)
    sure_neg_agreed = Prediction(prob=0.2, dispersion=np.array([0.1]))
    sure_neg_disputed = Prediction(prob=0.2, dispersion=np.array([0.9]))
    obvious_neg_disputed = Prediction(prob=0.03, dispersion=np.array([0.9]))
    sure_pos_disputed = Prediction(prob=0.97, dispersion=np.array([0.9]))
    unsure = Prediction(prob=0.55, dispersion=np.array([0.1]))
    d = decide_or_defer(sure_neg_agreed, **kw)
    assert d.rule == VETO_RULE and not d.deferred
    d = decide_or_defer(sure_neg_disputed, **kw)
    assert d.deferred
    assert any(s.name == "panel_disp" for s in d.signals)
    assert not decide_or_defer(sure_pos_disputed, **kw).deferred
    assert not decide_or_defer(obvious_neg_disputed, **kw).deferred   # obvious stays automatic
    # ... unless the dispute is extreme
    ext = dict(kw, extreme_threshold=0.8)
    d = decide_or_defer(obvious_neg_disputed, **ext)
    assert d.deferred and d.triggers == ("PANEL_WOULD_DISPUTE_NEGATIVE",)
    assert not decide_or_defer(Prediction(prob=0.03, dispersion=np.array([0.7])), **ext).deferred
    assert not decide_or_defer(sure_pos_disputed, **ext).deferred
    d = decide_or_defer(unsure, **kw)
    assert d.deferred and d.triggers == ("LOW_CONFIDENCE",)
    assert d.to_dict()["triggers"] == ["LOW_CONFIDENCE"]
    # disagreement alone is the older rung, unchanged
    only = decide_or_defer(sure_neg_disputed, calibrator=cal, disp_threshold=0.5)
    assert only.rule == "panel_disp" and only.deferred
    # a model without the disagreement head serves the certainty cut alone
    plain = decide_or_defer(Prediction(prob=0.55), calibrator=cal, conf_threshold=0.5)
    assert plain.rule == "confidence" and plain.deferred
