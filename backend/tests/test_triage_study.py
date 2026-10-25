"""opinion objectives, predictive uncertainty, and the deferral policy.

These pin the properties the contribution actually claims, not the arithmetic:

  * EMD must punish an ordinal miss harder than KL does, otherwise there is no
    reason to have two opinion objectives at all;
  * MC dropout must produce genuine spread, and must produce exactly none when
    the model has no dropout - a silent zero there would be reported as
    "the model is certain";
  * the learned policy must rank the model's OWN errors better than a
    disagreement threshold can, which is the whole argument for having it;
  * an empty conformal set must score as maximally ambiguous, not minimally.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from field_disagreement import metrics as M
from field_disagreement import triage


# --------------------------------------------------------------------------- #
# opinion objectives
# --------------------------------------------------------------------------- #
def _logits(dist):
    """Logits whose softmax is (approximately) `dist`."""
    return torch.log(torch.tensor([dist], dtype=torch.float32).clamp_min(1e-8))


def test_emd_sees_the_rating_order_and_kl_does_not():
    """The panel said 5. Predicting 4 is a near miss; predicting 1 is not. KL
    cannot tell those apart - both put the same mass off the true category - and
    that blindness is exactly why the ordinal arm exists."""
    q = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
    near = _logits([0.0, 0.0, 0.0, 1.0, 0.0])
    far = _logits([1.0, 0.0, 0.0, 0.0, 0.0])

    assert triage.opinion_emd_loss(far, q) > triage.opinion_emd_loss(near, q)
    assert triage.opinion_kl_loss(far, q) == pytest.approx(
        triage.opinion_kl_loss(near, q), rel=1e-4)


def test_opinion_loss_dispatch_and_no_op_arm():
    q = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1]])
    logits = torch.zeros(1, 5, requires_grad=True)
    for kind in ("kl", "emd"):
        assert triage.opinion_loss(kind, logits, q) >= 0
    # The "none" arm must still be differentiable, so every rung of the ablation
    # runs through the identical training step rather than a branched one.
    zero = triage.opinion_loss("none", logits, q)
    assert float(zero.detach()) == 0.0
    zero.backward()
    with pytest.raises(ValueError, match="unknown opinion loss"):
        triage.opinion_loss("wasserstein", logits, q)


def test_opinion_entropy_is_a_fraction_of_maximum():
    uniform = np.full((1, 5), 0.2)
    onehot = np.eye(5)[None, 2]
    assert triage.opinion_entropy(uniform)[0] == pytest.approx(1.0)
    assert triage.opinion_entropy(onehot)[0] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# predictive uncertainty
# --------------------------------------------------------------------------- #
def test_agreeing_members_have_no_epistemic_uncertainty():
    """Three members predicting the same divided panel: all the uncertainty is
    aleatoric. The panel is split, the model is not confused about that."""
    d = np.repeat(np.full((1, 1, 5), 0.2), 3, axis=0)
    u = triage.entropy_decomposition(d)
    assert u["aleatoric"][0] == pytest.approx(1.0)
    assert u["epistemic"][0] == pytest.approx(0.0, abs=1e-6)


def test_disagreeing_confident_members_are_purely_epistemic():
    """Each member is certain, and they are certain about different things. That
    is the case the opinion head alone cannot report and the whole reason a
    second uncertainty source is needed."""
    d = np.stack([np.eye(5)[None, 0], np.eye(5)[None, 4]])
    u = triage.entropy_decomposition(d)
    assert u["aleatoric"][0] == pytest.approx(0.0, abs=1e-6)
    assert u["epistemic"][0] > 0.4


class _Dropout(nn.Module):
    """Minimal stand-in for a head: one dropout layer on the decision path."""

    def __init__(self, p=0.5):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(4, 1)
        self.op = nn.Linear(4, 5)

    def forward(self, x, return_all=False):
        h = self.drop(x)
        return {"logit": self.fc(h).squeeze(1), "opinion": self.op(h)}


def test_mc_dropout_samples_differ_and_restore_eval_mode():
    model = _Dropout().eval()
    assert triage.has_dropout(model)
    s = triage.mc_dropout_predict(model, torch.ones(2, 4), n_passes=16, seed=0)
    assert s["probs"].shape == (16, 2)
    assert s["probs"].std(axis=0).max() > 0        # the passes actually differ
    # The context manager must hand the model back exactly as it found it - a
    # model left in train mode would silently corrupt the next deterministic pass.
    assert not model.training and not model.drop.training


def test_no_dropout_means_no_epistemic_term_and_is_detectable():
    """p=0 gives T identical passes. The number is a truthful 0, and `has_dropout`
    is what lets the caller say so instead of reporting false certainty."""
    model = _Dropout(p=0.0).eval()
    assert not triage.has_dropout(model)
    s = triage.mc_dropout_predict(model, torch.ones(3, 4), n_passes=8, seed=0)
    u = triage.summarise(s)
    assert np.allclose(u["opinion_epistemic"], 0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# conformal ambiguity
# --------------------------------------------------------------------------- #
def test_empty_conformal_set_is_maximally_ambiguous_not_minimally():
    """|C(x)| = 0 means NO label was plausible - the case is atypical. Passing the
    raw set size would rank it as the most confident outcome, which is backwards
    and would send the strangest cases straight through."""
    a = triage.conformal_ambiguity([0, 1, 2])
    assert a[1] == 0.0                     # singleton = the confident outcome
    assert a[0] == a[2] == 1.0             # empty and doubleton both defer


# --------------------------------------------------------------------------- #
# the learned deferral policy
# --------------------------------------------------------------------------- #
def _synthetic_split(n=400, seed=0):
    """A population where model error is driven by the model's own confusion, and
    panel disagreement is only loosely related to it - the situation the whole
    contribution is about."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    hard = rng.random(n) < 0.3                       # cases the model muddles
    p = np.where(hard, rng.uniform(0.4, 0.6, n),
                 np.where(y == 1, rng.uniform(0.8, 1.0, n), rng.uniform(0.0, 0.2, n)))
    # Panel disagreement overlaps with hardness but is not the same variable.
    disagreement = np.where(rng.random(n) < 0.5, hard, rng.random(n) < 0.3).astype(float)
    return y, p, hard, disagreement


def test_policy_ranks_the_models_own_errors_better_than_a_disagreement_threshold():
    """The claim the study has to defend: a policy fitted on predicted signals against
    the model's errors beats a head that imitates the annotation threshold."""
    y, p, hard, disagreement = _synthetic_split()
    signals, names = triage.build_signals(
        prob=p, defer_score=disagreement, opinion_entropy=disagreement,
        epistemic=hard * 0.5 + 0.1, set_size=np.where(hard, 2, 1))
    err = (p >= 0.5).astype(int) != y

    policy = triage.DeferralPolicy(signal_names=names).fit(signals, err)
    learned = M.rank_error_auc(y, p, policy.score(signals))
    threshold = M.rank_error_auc(y, p, disagreement)
    assert learned > threshold
    assert learned > 0.5


def test_policy_threshold_hits_the_requested_coverage():
    """The operating point is calibrated on val and frozen, so the deferral rate
    is a result rather than something tuned after seeing the test set."""
    y, p, hard, _ = _synthetic_split()
    signals, names = triage.build_signals(prob=p, epistemic=hard.astype(float))
    err = (p >= 0.5).astype(int) != y
    policy = triage.DeferralPolicy(signal_names=names).fit(signals, err)
    policy.calibrate_threshold(signals, coverage=0.8)
    assert policy.decide(signals).mean() == pytest.approx(0.2, abs=0.05)


def test_unfittable_policy_degrades_to_the_confidence_baseline():
    """A validation split with no errors has no ranking to learn. Falling back to
    confidence keeps the system safe; returning fitted noise would not."""
    signals, names = triage.build_signals(prob=np.linspace(0.9, 1.0, 20))
    policy = triage.DeferralPolicy(signal_names=names).fit(signals, np.zeros(20))
    assert policy.weights is None
    assert np.allclose(policy.score(signals), signals[:, 0])
    assert all(np.isnan(v) for v in policy.coefficients().values())


def test_signal_columns_all_point_the_same_way():
    """Every column must be defer-monotone, or the fitted coefficients cannot be
    read as 'this signal argued for referral'."""
    signals, names = triage.build_signals(
        prob=[0.5, 0.99], defer_score=[0.9, 0.1], opinion_entropy=[0.9, 0.1],
        epistemic=[0.9, 0.1], set_size=[2, 1])
    assert names == ["confidence_gap", "defer_head", "opinion_entropy",
                     "epistemic", "conformal"]
    assert (signals[0] > signals[1]).all()      # the contested case scores higher


# --------------------------------------------------------------------------- #
# selective-prediction reporting
# --------------------------------------------------------------------------- #
def test_compare_selectors_always_includes_the_confidence_control():
    """No caller can report a deferral mechanism without the baseline it has to
    beat - the omission that would turn the study back into a demonstration."""
    y, p, _, disagreement = _synthetic_split()
    table = M.compare_selectors(y, p, {"defer_head": -disagreement})
    assert "confidence" in table
    assert table["confidence"]["aurc"] <= table["defer_head"]["aurc"]


def test_error_concentration_reports_lift_over_random_referral():
    y, p, _, _ = _synthetic_split()
    perfect = M.error_concentration(y, p, (p >= 0.5).astype(int) != y)
    assert perfect["error_lift"] > 1.0
    assert perfect["kept_error_rate"] == 0.0


def test_deferring_every_positive_is_visible_in_the_sensitivity_report():
    """Accuracy at fixed coverage improves by dropping hard positives, which for a
    screening system is the failure mode. `deferred_malignant` is what exposes it."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5])
    conf = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)   # defer all positives
    s = M.sensitivity_at_coverage(y, p, conf, coverage=0.5)
    assert s["deferred_malignant"] == 1.0
    assert s["missed_malignant"] == 0.0                      # deferred != missed


def test_opinion_metrics_separate_placement_shape_and_spread():
    """A prediction can match the panel's entropy exactly and still put the mass
    in the wrong place. Reporting one number would hide that."""
    true = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
    wrong_place = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    m = M.opinion_metrics(wrong_place, true)
    assert m["opinion_emd"] > 0
    assert m["entropy_mae"] == pytest.approx(0.0, abs=1e-6)   # same spread...
    assert m["opinion_kl"] > 0                               # ...different shape


# --------------------------------------------------------------------------- #
# reporting: temperature, workload-at-sensitivity, opinion baselines, pairing
# --------------------------------------------------------------------------- #
def test_temperature_rescales_calibration_without_touching_the_ranking():
    """Temperature is applied to every arm, so it must be incapable of moving
    what the arms are COMPARED on. Dividing a logit by T > 0 preserves its sign
    and its order: decisions, errors and every rank-based metric are untouched,
    and only calibration moves."""
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    p = np.array([.02, .05, .30, .55, .95, .99, .80, .20])   # over-confident
    pt = M.apply_temperature(p, M.fit_temperature(y, p))

    assert M.roc_auc(y, pt) == pytest.approx(M.roc_auc(y, p))
    assert ((pt >= 0.5) == (p >= 0.5)).all()                 # identical error set
    assert M.nll(y, pt) <= M.nll(y, p)                       # and it did help
    assert M.apply_temperature(p, 1.0) == pytest.approx(p)


def test_coverage_at_sensitivity_trades_workload_against_recall():
    """The one false negative is the LEAST confident case, so full coverage
    cannot hold 100% sensitivity - and demanding more recall can never let you
    auto-report more cases."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([.01, .02, .03, .04, .90, .80, .70, .40])
    conf = np.abs(p - 0.5)
    assert M.coverage_at_sensitivity(y, p, conf, 1.0) == pytest.approx(7 / 8)
    assert M.coverage_at_sensitivity(y, p, conf, 0.7) == 1.0


def test_opinion_baselines_are_what_a_classifier_only_system_could_emit():
    """`prior` ignores the image (the floor); `class_conditional` interpolates
    the two class panels by p - which is why its entropy is welded to class
    uncertainty, the limit the opinion head exists to escape."""
    q = np.array([[1., 0, 0, 0, 0], [0, 0, 0, 0, 1.]])
    y = np.array([0, 1])
    fitted = M.fit_opinion_baselines(q, y)
    p = np.array([0.0, 1.0])

    prior = M.predict_opinion_baseline(p, fitted, "prior")
    cond = M.predict_opinion_baseline(p, fitted, "class_conditional")
    for Q in (prior, cond):
        assert Q.sum(axis=1) == pytest.approx(np.ones(2)) and (Q >= 0).all()
    assert prior[0] == pytest.approx(prior[1])        # same map for every case
    assert cond[0] == pytest.approx(q[0])             # p=0 -> the benign panel
    assert cond[1] == pytest.approx(q[1])             # p=1 -> the malignant one
    with pytest.raises(ValueError):
        M.predict_opinion_baseline(p, fitted, "nonsense")


def test_paired_fold_test_orients_both_metric_directions():
    """`delta` must always read as 'positive = arm A is better', whichever way
    the metric improves, or half the table silently inverts."""
    a = [{"m": v} for v in (0.10, 0.11, 0.09, 0.12, 0.10)]   # lower is better
    b = [{"m": v} for v in (0.20, 0.21, 0.19, 0.22, 0.20)]
    lower = M.paired_fold_test(a, b, "m", lower_is_better=True)
    higher = M.paired_fold_test(a, b, "m", lower_is_better=False)

    assert lower["delta"] > 0 and lower["wins"] == 5
    assert higher["delta"] == pytest.approx(-lower["delta"]) and higher["wins"] == 0
    # 5 paired folds cannot reach p < 0.05: the design's floor, not a bug.
    assert lower["p"] == pytest.approx(0.0625)


# --------------------------------------------------------------------------- #
# the policy's input columns - defined once, so every caller builds the same ones
# --------------------------------------------------------------------------- #
def test_a_policy_fitted_without_a_head_cannot_be_served_silently():
    """The study drops dead signal columns (`drop_constant`); a single case has
    N=1, where every column is trivially constant, so it must keep them all. A
    policy fitted on an arm that lacked a head therefore carries fewer weights than
    a single case supplies - and that has to fail by NAME, not as a numpy
    broadcasting error thrown from inside the arithmetic."""
    y, p, hard, disagreement = _synthetic_split(n=120)
    study, names = triage.build_signals(prob=p, defer_score=None,
                                        opinion_entropy=disagreement,
                                        epistemic=hard * 0.5 + 0.1,
                                        set_size=np.where(hard, 2, 1),
                                        drop_constant=True)
    assert "defer_head" not in names                  # the arm never built one
    policy = triage.DeferralPolicy(signal_names=names).fit(
        study, (p >= 0.5).astype(int) != y)

    served, served_names = triage.build_signals(
        prob=[0.7], defer_score=[0.3], opinion_entropy=[0.6], epistemic=[0.01],
        set_size=[2])
    assert served_names == ["confidence_gap", "defer_head", "opinion_entropy",
                            "epistemic", "conformal"]
    with pytest.raises(ValueError, match="fitted with 4 signals"):
        policy.score(served)


def test_the_column_set_is_a_property_of_the_arm_not_of_the_split():
    """A val fold whose conformal sets are ALL singletons must not fit a policy
    one column narrower than the test fold it is applied to.

    This crashed a real study run mid-ladder: `drop_constant` dropped `conformal` on
    val (no variance there), the test split had doubletons, and `score` was handed
    five columns against four weights. `conformal` is now never dropped for
    constancy, and the fitted policy's own `signal_names` pin what a later split
    builds."""
    y, p, hard, disagreement = _synthetic_split(n=120)
    val, names = triage.build_signals(
        prob=p, defer_score=disagreement, opinion_entropy=disagreement,
        epistemic=hard * 0.5 + 0.1, set_size=np.ones(len(p)),   # every set a singleton
        drop_constant=True)
    assert names[-1] == "conformal"                # kept, though it never moves

    policy = triage.DeferralPolicy(signal_names=names).fit(
        val, (p >= 0.5).astype(int) != y)
    # A dead column is inert rather than absent: the fit gives it exactly 0, so
    # keeping it cannot change what the other signals were fitted to say.
    assert policy.coefficients()["conformal"] == 0.0

    test, test_names = triage.build_signals(
        prob=p, defer_score=disagreement, opinion_entropy=disagreement,
        epistemic=hard * 0.5 + 0.1, set_size=np.where(hard, 2, 1),
        drop_constant=True, keep=policy.signal_names)
    assert test_names == names
    assert np.all(np.isfinite(policy.score(test)))


def test_keep_names_a_signal_the_arm_cannot_supply():
    """`keep` must fail by NAME when the arm being scored genuinely lacks a head
    the policy was fitted on - the silent alternative is a positional mismatch."""
    with pytest.raises(ValueError, match="defer_head"):
        triage.build_signals(prob=[0.7, 0.2], epistemic=[0.1, 0.9],
                             keep=["confidence_gap", "defer_head"])


def test_nested_fit_keeps_the_weight_a_confidence_only_fit_would_give():
    """Nesting fits confidence ALONE first and lets the rest explain only the
    residual - which is what makes the other coefficients readable as "what the
    panel machinery adds ON TOP OF confidence"."""
    y, p, hard, disagreement = _synthetic_split()
    err = (p >= 0.5).astype(int) != y
    sig, names = triage.build_signals(prob=p, defer_score=disagreement,
                                      opinion_entropy=disagreement,
                                      epistemic=hard * 0.5 + 0.1,
                                      set_size=np.where(hard, 2, 1))
    solo, _ = triage.build_signals(prob=p)

    only_conf = triage.DeferralPolicy().fit(solo, err, nest_confidence=False)
    nested = triage.DeferralPolicy(signal_names=names).fit(sig, err)
    assert nested.weights[0] == pytest.approx(only_conf.weights[0], rel=1e-6)


def test_candidate_rules_omit_signals_the_arm_never_built():
    """The ablation stays honest about what each rung had available, and the
    `confidence` control is never among them - compare_selectors inserts it."""
    _, p, _, disagreement = _synthetic_split(n=80)
    assert set(triage.candidate_rules(prob=p, q_hat=0.6)) == {"conformal"}

    rules = triage.candidate_rules(prob=p, q_hat=0.6, defer_score=disagreement,
                                   opinion_entropy=disagreement,
                                   epistemic=np.zeros_like(p))
    assert "epistemic" not in rules                   # no variance -> not a rule
    assert "confidence" not in rules                  # the control comes later
    # rules are CONFIDENCE-oriented (higher = decide first), so a defer score flips
    assert np.allclose(rules["defer_head"], -disagreement)


# --------------------------------------------------------------------------- #
# the deferral warm-up schedule
# --------------------------------------------------------------------------- #
def test_defer_lambda_ramps_over_a_fraction_of_training_and_then_holds():
    """The warm-up is a fraction of total training rather than a fixed epoch
    count, so it stretches with `epochs` instead of finishing early on a long
    run."""
    lam = [triage.defer_lambda(e, 1.0, 0.3, 100) for e in range(100)]
    assert lam[0] == pytest.approx(1 / 30)         # ramping from the first epoch
    assert lam[29] == pytest.approx(1.0)           # full weight at 30% of training
    assert lam[-1] == 1.0                          # and clamped thereafter
    assert all(b >= a for a, b in zip(lam, lam[1:]))

    # The fraction is of TOTAL training, so halving the run halves the ramp.
    assert triage.defer_lambda(14, 1.0, 0.3, 50) == pytest.approx(1.0)
    # A zero-length warm-up is full weight at once, not a division by zero.
    assert triage.defer_lambda(0, 2.0, 0.0, 100) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# the policy is one serialisation format, not two hand-written halves
# --------------------------------------------------------------------------- #
def _fitted_policy(seed=0, n=200):
    rng = np.random.default_rng(seed)
    prob = rng.random(n)
    labels = (rng.random(n) < prob).astype(int)
    signals, names = triage.build_signals(prob, defer_score=rng.random(n),
                                          epistemic=rng.random(n))
    return triage.fit_deferral_policy(signals, names, prob, labels), signals


def test_policy_round_trips_through_its_own_on_disk_form():
    """`to_dict` / `from_dict` are the policy's on-disk form; a lossy round trip
    would change the rule that is applied."""
    policy, signals = _fitted_policy()
    clone = triage.DeferralPolicy.from_dict(policy.to_dict(round_to=None))

    assert clone.signal_names == policy.signal_names
    assert np.allclose(clone.score(signals), policy.score(signals))
    assert np.array_equal(clone.decide(signals), policy.decide(signals))


def test_an_absent_policy_block_degrades_to_the_confidence_baseline():
    """An unfitted policy scores on confidence rather than on invented weights."""
    for block in (None, {}, {"signal_names": ["confidence_gap"], "weights": None}):
        p = triage.DeferralPolicy.from_dict(block)
        assert p.weights is None
        assert np.allclose(p.score(np.array([[0.7, 0.2]])), [0.7])


def test_averaging_folds_gives_the_one_rule_that_is_applied():
    a, signals = _fitted_policy(seed=1)
    b, _ = _fitted_policy(seed=2)
    mean = triage.DeferralPolicy.average([a, b])

    assert mean.signal_names == a.signal_names
    assert np.allclose(mean.weights, (a.weights + b.weights) / 2)
    assert len(mean.score(signals)) == len(signals)


def test_averaging_skips_folds_that_never_fitted():
    """A fold whose validation split had no errors holds no policy. Counting it
    as zeros would drag the mean toward a rule no fold ever held."""
    a, _ = _fitted_policy(seed=3)
    mean = triage.DeferralPolicy.average([a, triage.DeferralPolicy(), None])
    assert np.allclose(mean.weights, a.weights)
    assert triage.DeferralPolicy.average([]).weights is None


def test_averaging_refuses_folds_that_weighted_different_signals():
    """An arm without a head drops that column, so mismatched names mean the
    folds did not build the same heads - averaging them position by position
    would attribute weights to the wrong signals."""
    a, _ = _fitted_policy(seed=4)
    b = triage.DeferralPolicy(weights=np.ones(2), mean=np.zeros(2),
                              scale=np.ones(2), signal_names=["confidence_gap", "x"])
    with pytest.raises(ValueError, match="different signals"):
        triage.DeferralPolicy.average([a, b])


def test_the_safety_cut_clears_at_most_the_allowed_share_of_positives():
    from field_disagreement import triage
    rng = np.random.default_rng(0)
    n = 600
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    conf = triage.confidence_gap(p)
    for miss in (0.0, 0.05, 0.2):
        cut = triage.safety_cut(conf, p, y, miss)
        cleared = (conf < cut) & (p < 0.5) & (y == 1)
        assert cleared.sum() <= np.floor(miss * y.sum())
    # the loosest such cut: one step looser clears one positive too many
    cut = triage.safety_cut(conf, p, y, 0.05)
    k = int(np.floor(0.05 * y.sum()))
    assert ((conf <= cut) & (p < 0.5) & (y == 1)).sum() == k + 1
    # nothing it could clear breaks the limit -> no referral on certainty
    assert triage.safety_cut(conf, np.ones(n), y, 0.05) == float("inf")
    with pytest.raises(ValueError):
        triage.safety_cut(conf, p, np.zeros(n), 0.05)


def test_the_veto_reads_only_negative_calls_and_only_moves_cases_to_a_reader():
    from field_disagreement import triage
    rng = np.random.default_rng(1)
    n = 800
    p, disp = rng.uniform(0, 1, n), rng.normal(size=n)
    conf = triage.confidence_gap(p)
    dcut = triage.veto_cut(disp, p, 0.2)
    neg = p < 0.5
    assert np.mean(disp[neg] >= dcut) == pytest.approx(0.2, abs=0.01)
    assert triage.veto_cut(disp, p, 0.0) == float("inf")
    base = triage.served_refer(conf, p, conf_cut=0.6)
    both = triage.served_refer(conf, p, disp, conf_cut=0.6, disp_cut=dcut)
    assert (base <= both).all()                       # never un-refers a case
    assert not (both & ~base & ~neg).any()            # positive calls are never vetoed
    assert not (both & ~base & (p <= triage.OBVIOUS_NEGATIVE)).any()   # nor obvious negatives
    assert (both & ~base).any()
    # the extreme tier: an obvious negative goes only when its dispute is extreme
    ecut = triage.veto_cut(disp, p, 0.02)
    ext = triage.served_refer(conf, p, disp, conf_cut=0.6, disp_cut=dcut, extreme_cut=ecut)
    assert (both <= ext).all() and not (ext & ~both & ~neg).any()
    added = ext & ~both
    assert added.any() and (p[added] <= triage.OBVIOUS_NEGATIVE).all()
    assert (disp[added] >= ecut).all()


def test_served_outcomes_score_each_kind_of_case_over_the_whole_split():
    from field_disagreement import metrics as M
    auto = np.array([1, 1, 1, 0, 1, 0], bool)
    p = np.array([0.05, 0.3, 0.8, 0.4, 0.95, 0.6])
    y = np.array([0, 1, 0, 1, 1, 0])
    d = np.array([0.0, 3.0, 0.0, 3.0, 0.0, 3.0])
    o = M.served_outcomes(auto, p, y, d, 1.0)
    assert o["auto"] == pytest.approx(4 / 6)
    assert o["missed_pos"] == pytest.approx(1 / 3)          # 0.3 cleared, positive
    assert o["false_fast"] == pytest.approx(1 / 3)          # 0.8 fast-tracked, negative
    assert o["contested_neg"] == pytest.approx(1 / 3)       # of the three contested
    assert o["contested_auto"] == pytest.approx(1 / 3)
    assert o["obvious_pos_auto"] == 1.0 and o["obvious_neg_auto"] == 1.0
