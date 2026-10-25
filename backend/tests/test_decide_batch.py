"""`decide_batch`: the triage decision over a whole split of stored predictions."""
from __future__ import annotations

import numpy as np
import pytest

from field_disagreement import Calibrator, decide_batch
from field_disagreement.triage import fit_deferral_policy, prediction_signals

K = 5


@pytest.fixture
def split():
    """A `predict_full`-shaped result for one split, with every head present."""
    rng = np.random.default_rng(0)
    n = 60
    return {"probs": rng.uniform(0.05, 0.95, n),
            "labels": rng.integers(0, 2, n),
            "opinion": rng.dirichlet(np.ones(K), n),
            "defer": rng.dirichlet(np.ones(3), n),
            "aleatoric": rng.uniform(0.0, 1.0, n),
            "epistemic": rng.uniform(0.0, 0.3, n)}


def test_the_rule_degrades_to_what_the_arm_can_support(split):
    """A fitted policy is used when there is one, else the supervised head; a
    classifier-only arm has neither and still runs through the same call, and
    `rule` says which was used."""
    cal = Calibrator.fit(split["probs"], split["labels"], alpha=0.1)
    signals, names = prediction_signals(
        prob=split["probs"], q_hat=cal.q_hat, defer_score=split["defer"][:, 2],
        opinion_entropy=split["aleatoric"], epistemic=split["epistemic"],
        drop_constant=True)
    policy = fit_deferral_policy(signals, names, split["probs"],
                                 split["labels"], coverage=0.8)

    assert decide_batch(split, calibrator=cal, policy=policy)["rule"] == "policy"
    assert decide_batch(split, calibrator=cal)["rule"] == "defer_head"

    thin = {"probs": split["probs"]}
    out = decide_batch(thin)
    assert out["rule"] == "confidence" and out["n"] == len(thin["probs"])


def test_an_empty_conformal_set_forces_a_referral_over_a_split():
    out = decide_batch({"probs": np.array([0.5]),
                        "defer": np.array([[0.6, 0.3, 0.1]])},
                       calibrator=Calibrator(q_hat=0.1, alpha=0.1))
    assert out["set_size"][0] == 0
    assert out["deferred"][0] and out["forced_by_empty_set"][0]


def test_every_returned_column_is_aligned_with_the_split(split):
    """Results are pooled across folds downstream, so a length mismatch would
    silently misalign a per-case column against its labels."""
    out = decide_batch(split, calibrator=Calibrator.fit(
        split["probs"], split["labels"], alpha=0.1))
    for key in ("deferred", "score", "set_size", "forced_by_empty_set"):
        assert len(out[key]) == len(split["probs"]), key


def test_the_disagreement_rule_over_a_split():
    n = 50
    rng = np.random.default_rng(0)
    pred = {"probs": rng.uniform(0.05, 0.95, n), "dispersion": rng.normal(size=(n, 7))}
    cut = float(np.quantile(pred["dispersion"].mean(axis=1), 0.8))
    dep = decide_batch(pred, disp_threshold=cut)
    assert dep["rule"] == "panel_disp"
    assert dep["deferred"].tolist() == (pred["dispersion"].mean(axis=1) >= cut).tolist()
    assert decide_batch(pred)["rule"] == "confidence"


def test_the_triage_rule_over_a_split():
    """Certainty cut, plus the disagreement veto on negative calls above the
    extreme tier, plus the extreme-tier cut on every negative call."""
    n = 40
    rng = np.random.default_rng(1)
    pred = {"probs": rng.uniform(0.05, 0.95, n), "dispersion": rng.normal(size=(n, 7))}
    dcut, ccut, ecut = 0.3, 0.6, 0.5
    dep = decide_batch(pred, disp_threshold=dcut, conf_threshold=ccut, extreme_threshold=ecut)
    gap = 1 - 2 * np.abs(pred["probs"] - 0.5)
    m = pred["dispersion"].mean(axis=1)
    neg = pred["probs"] < 0.5
    want = (gap >= ccut) | (neg & (pred["probs"] > 0.1) & (m >= dcut)) | (neg & (m >= ecut))
    assert dep["rule"] == "confidence+negative_veto"
    assert dep["deferred"].tolist() == want.tolist()
