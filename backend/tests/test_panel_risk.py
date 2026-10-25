"""The study's two-risk machinery: panel-risk supervision, its read-out, and the
certificate over the auto-decided set.

These pin the properties the claim rests on, in the order the claim is made:

  1. the read-out measures PANEL SPREAD, which entropy provably cannot;
  2. the separation test is run at the cohort's DECLARED threshold, and the
     threshold-free correlation agrees with it;
  3. the certificate controls risk on the SELECTED set, which plain conformal
     does not, and its coverage is the thing a router competes on.
"""
import numpy as np
import pytest
import torch

from field_disagreement import metrics as M
from field_disagreement import triage
from field_disagreement.triage import risk_control as RC
from backend.notebooks.shared import panel as PANEL


# --------------------------------------------------------------------------- #
# 1. the read-out
# --------------------------------------------------------------------------- #
def test_spread_separates_panels_that_entropy_cannot():
    """1/1/5/5 and 2/2/4/4 have identical entropy and different spreads.

    This is the whole reason the panel-risk read-out is the spread: the
    agreed-vs-divided question is about WHERE the mass sits, and entropy is
    invariant to that. If this ever fails the contribution is being scored on a
    statistic that cannot see its own claim."""
    polarised = np.array([[0.5, 0.0, 0.0, 0.0, 0.5]])      # 1,1,5,5
    adjacent = np.array([[0.0, 0.5, 0.0, 0.5, 0.0]])       # 2,2,4,4
    h_pol = M.distribution_entropy(polarised)[0]
    h_adj = M.distribution_entropy(adjacent)[0]
    assert h_pol == pytest.approx(h_adj)                   # entropy: blind

    s_pol = triage.opinion_spread(polarised)[0]
    s_adj = triage.opinion_spread(adjacent)[0]
    assert s_pol == pytest.approx(2.0, abs=1e-3)
    assert s_adj == pytest.approx(1.0, abs=1e-3)


def test_spread_matches_the_cohorts_own_disagreement_statistic():
    """The cohort stores `disagreement` = s.d. of the ratings. The predicted
    spread has to be that same functional, or predicted-vs-true is a comparison
    of two different quantities dressed as an error."""
    ratings = [3, 4, 5, 3]
    hist = np.zeros((1, 5))
    for r in ratings:
        hist[0, r - 1] += 0.25
    assert triage.opinion_spread(hist)[0] == pytest.approx(np.std(ratings), abs=1e-6)


def test_spread_read_out_agrees_between_numpy_and_torch():
    p = np.random.default_rng(0).dirichlet(np.ones(5), 8)
    assert np.allclose(triage.opinion_spread(p),
                       triage.opinion_spread(torch.tensor(p)).numpy(), atol=1e-6)


def test_spread_error_reports_level_and_ranking_separately():
    rng = np.random.default_rng(1)
    d = rng.uniform(0, 2, 200)
    # a predictor that ranks perfectly but is biased high by 0.5
    P = np.stack([_hist_with_spread(x + 0.5) for x in d])
    out = M.spread_error(P, d)
    assert out["spread_corr"] > 0.9                       # ranking: right
    assert out["spread_bias"] > 0.2                       # level: wrong
    assert out["spread_mae"] > 0.2


def _hist_with_spread(s: float) -> np.ndarray:
    """A 5-bin histogram centred on 3 with (approximately) spread `s`."""
    s = float(np.clip(s, 0.0, 2.0))
    w = min(1.0, (s / 2.0) ** 2)
    return np.array([w / 2, 0.0, 1.0 - w, 0.0, w / 2])


# --------------------------------------------------------------------------- #
# 2. the separation test
# --------------------------------------------------------------------------- #
def test_declared_threshold_drops_the_near_agreed_middle():
    """`nonzero` calls a 3/3/3/4 panel divided, which is most of the borderline
    set and is what pulled every signal towards 0.5. The declared threshold
    keeps the two ends and is the contrast the design is about."""
    d = np.array([0.0] * 20 + [0.43] * 60 + [1.2] * 20)
    sig = {"s": np.r_[np.zeros(20), np.zeros(60), np.ones(20)]}
    loose = M.disagreement_separation(d, sig, split="nonzero")
    tight = M.disagreement_separation(d, sig, split="threshold", tau=1.0)
    assert loose["n"] == 100 and tight["n"] == 40          # middle dropped
    assert tight["signals"]["s"]["auc"] == pytest.approx(1.0)
    assert loose["signals"]["s"]["auc"] < tight["signals"]["s"]["auc"]


def test_threshold_split_demands_a_declared_tau():
    with pytest.raises(ValueError):
        M.disagreement_separation(np.zeros(10), {"s": np.zeros(10)},
                                  split="threshold")


def test_clustered_ci_is_wider_than_the_item_level_one():
    """One subject contributing several nodules makes an item bootstrap too
    narrow. The clustered interval is the one to report."""
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(40), 5)
    y = np.repeat(rng.integers(0, 2, 40), 5)
    d = np.where(y == 1, 1.5, 0.0)
    # The nodules of one patient look alike, so the signal carries a SUBJECT
    # -level offset. That shared component is exactly what an item bootstrap
    # averages away and a cluster bootstrap keeps.
    s = y + np.repeat(rng.normal(0, 1.0, 40), 5) + rng.normal(0, 0.2, len(y))
    item = M.disagreement_separation(d, {"s": s}, split="threshold", tau=1.0)
    clus = M.disagreement_separation(d, {"s": s}, split="threshold", tau=1.0,
                                     clusters=clusters)
    width = lambda b: b["ci95_hi"] - b["ci95_lo"]          # noqa: E731
    assert width(clus["signals"]["s"]) > width(item["signals"]["s"])


def test_rank_agreement_is_threshold_free_and_pairs_against_the_reference():
    rng = np.random.default_rng(4)
    d = rng.uniform(0, 2, 300)
    signals = {"confidence": rng.normal(size=300),          # reference: noise
               "panel": d + rng.normal(0, 0.3, 300)}
    out = M.rank_agreement(d, signals)
    assert out["signals"]["panel"]["rho"] > 0.8
    assert out["signals"]["panel"]["vs_confidence_lo"] > 0   # CI clear of zero
    assert "vs_confidence" not in out["signals"]["confidence"]


# --------------------------------------------------------------------------- #
# 3. the certificate
# --------------------------------------------------------------------------- #
def test_binomial_bound_is_above_the_rate_and_vacuous_without_data():
    assert RC.binom_upper(0, 0) == 1.0
    assert RC.binom_upper(5, 100, 0.05) > 0.05
    assert RC.binom_upper(5, 100, 0.05) < RC.binom_upper(5, 20, 0.05)


def test_certificate_controls_risk_on_the_SELECTED_set():
    """The property split conformal does not give: the bound is conditional on
    being auto-decided."""
    rng = np.random.default_rng(5)
    n = 4000
    score = rng.uniform(0, 1, n)                 # defer score
    loss = (rng.uniform(0, 1, n) < score ** 2).astype(float)   # risk rises with it
    cert = RC.calibrate(score, loss, alpha=0.05, delta=0.05)
    assert cert.feasible and 0.0 < cert.coverage < 1.0
    assert cert.bound["error"] <= 0.05
    keep = RC.apply_certificate(cert, {"error": score})
    assert loss[keep].mean() <= 0.05


def test_infeasible_certificate_defers_everything():
    """No threshold passes when every case is an error. The rule must refuse to
    auto-decide rather than quietly fall back to a coverage target."""
    score = np.linspace(0, 1, 200)
    cert = RC.calibrate(score, np.ones(200), alpha=0.01)
    assert not cert.feasible
    assert RC.apply_certificate(cert, {"error": score}).sum() == 0


def test_dual_risk_router_beats_a_single_score_at_equal_constraints():
    """The comparison the study is built to make. Two independent risks: a confidence
    -only router has to satisfy the panel constraint with a signal blind to it,
    and pays in coverage."""
    rng = np.random.default_rng(6)
    n = 3000
    r_err = rng.uniform(0, 1, n)
    r_pan = rng.uniform(0, 1, n)                  # independent of r_err
    err = (rng.uniform(0, 1, n) < r_err ** 3).astype(float)
    contested = (rng.uniform(0, 1, n) < r_pan ** 3).astype(float)
    alpha = {"error": 0.05, "panel": 0.10}

    both = RC.dual_calibrate({"error": r_err, "panel": r_pan},
                             {"error": err, "panel": contested}, alpha)
    only = RC.dual_calibrate({"error": r_err, "panel": r_err},
                             {"error": err, "panel": contested}, alpha)
    assert both.feasible and both.coverage > only.coverage
    for k in alpha:
        assert both.bound[k] <= alpha[k]


def test_leakage_and_enrichment_read_the_deployed_rule():
    d = np.r_[np.zeros(80), np.full(20, 1.5)]
    perfect = np.r_[np.zeros(80), np.ones(20)]            # defers every split panel
    out = M.contested_leakage(perfect, d, d0=1.0, coverage=0.8)
    assert out["leakage"] == pytest.approx(0.0)
    assert out["enrichment"] == pytest.approx(5.0)        # 1 / base rate

    blind = np.linspace(0, 1, 100)                        # unrelated to d
    rng = np.random.default_rng(7)
    rng.shuffle(d)
    out_blind = M.contested_leakage(blind, d, d0=1.0, coverage=0.8)
    assert out_blind["leakage"] > 0.0


# --------------------------------------------------------------------------- #
# 4. calibration of the panel's dispersion (the ECE the label cannot give)
# --------------------------------------------------------------------------- #
def test_dispersion_calibration_error_is_zero_only_when_calibrated():
    """DCE answers a question `ece` cannot: not "is the predicted probability of
    the LABEL right" but "when the model says this panel will disagree by one
    rating point, do those panels disagree by one rating point"."""
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 2, 600)
    calibrated = np.stack([_hist_with_spread(x) for x in d])
    biased = np.stack([_hist_with_spread(x + 0.6) for x in d])

    assert M.dispersion_calibration_error(calibrated, d) == pytest.approx(0.0, abs=1e-3)
    assert M.dispersion_calibration_error(biased, d) > 0.3


def test_dce_inherits_eces_blind_spot_and_must_not_be_read_alone():
    """A head with NO RESOLUTION - one number for every case - lands every item
    in one bin, where that bin's mean is compared against itself, and scores a
    low DCE while being useless. Exactly ECE's own failure mode, which is why a
    classifier is never reported on ECE alone either.

    Pinned as a property, not a bug: it is the reason `spread_error` returns
    calibration, ranking and error together, and the reason the tables print
    all three."""
    d = np.r_[np.full(300, 0.2), np.full(300, 1.6)]
    P = np.stack([_hist_with_spread(1.0) for _ in d])      # predicts the mean
    out = M.spread_error(P, d)
    assert out["spread_dce"] < 0.15                        # looks calibrated
    assert out["spread_mae"] > 0.5                         # and is not useful
    assert not np.isfinite(out["spread_corr"])             # no ranking at all


def test_dce_needs_the_ranking_beside_it():
    """An uninformative head scores about the same DCE as a biased but
    well-ordered one, so the pair (DCE, corr) is what identifies the failure -
    neither number alone does."""
    rng = np.random.default_rng(2)
    d = rng.uniform(0, 2, 600)
    noise = np.stack([_hist_with_spread(rng.uniform(0, 2)) for _ in d])
    out = M.spread_error(noise, d)
    assert out["spread_dce"] > 0.3 and abs(out["spread_corr"]) < 0.15


def test_reliability_bins_are_shared_across_arms():
    """Equal-width edges over [0, (K-1)/2], not this arm's own quantiles - two
    arms binned differently cannot be compared, which is what the table does."""
    d = np.random.default_rng(3).uniform(0, 2, 200)
    a = M.dispersion_reliability(np.stack([_hist_with_spread(x) for x in d]), d,
                                 n_bins=5)
    b = M.dispersion_reliability(np.stack([_hist_with_spread(0.3) for _ in d]), d,
                                 n_bins=5)
    assert set(zip(a["bin_lo"], a["bin_hi"])) >= set(zip(b["bin_lo"], b["bin_hi"]))
    assert max(a["bin_hi"]) == pytest.approx(2.0)          # (5 - 1) / 2

# --------------------------------------------------------------------------- #
# The certificate: a frontier over the panel constraint, an
# intersection-union test across the two risks, and a calibration split that is
# not the one the checkpoint was selected on.
# --------------------------------------------------------------------------- #

def _router_cohort(n=240, seed=0):
    """A cohort where the panel signal ranks dispersion better than confidence,
    which is the only situation the router comparison is about."""
    rng = np.random.default_rng(seed)
    d = rng.gamma(2.0, 0.33, n)
    z = (d - d.mean()) / d.std()
    conf = rng.normal(0.26 * z, 1.0, n)
    spread = rng.normal(0.95 * z, 1.0, n)
    zc = (conf - conf.mean()) / conf.std()
    err = (rng.random(n) < 1 / (1 + np.exp(-(-2.6 + 1.1 * zc)))).astype(float)
    return conf, spread, err, (d > 1.0).astype(float), d


def test_certified_coverage_cannot_fall_as_the_constraint_loosens():
    """A larger beta is a strictly weaker requirement, so the certified set can
    only grow. If this inverts, the search is not returning the best point it
    found and the frontier cannot be read as a trade-off curve."""
    conf, spread, err, pan, _ = _router_cohort()
    rows = RC.beta_frontier(conf, spread, err, pan, alpha=0.10,
                            betas=(0.05, 0.10, 0.15, 0.20, 0.30))
    cov = [r["coverage"] for r in rows]
    assert cov == sorted(cov), f"coverage fell as beta rose: {cov}"


def test_the_panel_router_never_certifies_less_than_the_baseline():
    """Both routers carry the SAME error constraint on the same signal and
    differ only in what carries the panel constraint. A signal that ranks panel
    division better cannot buy less automation under an identical contract."""
    conf, spread, err, pan, _ = _router_cohort()
    base = RC.beta_frontier(conf, conf, err, pan, alpha=0.10)
    ours = RC.beta_frontier(conf, spread, err, pan, alpha=0.10)
    for b, o in zip(base, ours):
        assert o["coverage"] >= b["coverage"] - 1e-9, (
            f"at beta={b['beta']} the panel router certified "
            f"{o['coverage']:.3f} against the baseline's {b['coverage']:.3f}")


def test_headline_beta_is_chosen_by_the_baseline_not_by_us():
    """The operating point is the SMALLEST beta the baseline can meet. Letting
    the baseline choose it is the whole point: it cannot then have been picked
    where our own router happens to look best."""
    rows = [{"beta": 0.05, "feasible": 0, "coverage": 0.0},
            {"beta": 0.10, "feasible": 0, "coverage": 0.0},
            {"beta": 0.15, "feasible": 1, "coverage": 0.3},
            {"beta": 0.20, "feasible": 1, "coverage": 0.6}]
    assert RC.headline_beta(rows) == 0.15
    assert RC.headline_beta([{**r, "feasible": 0, "coverage": 0.0}
                             for r in rows]) is None


def test_delta_is_not_split_between_the_two_risks():
    """Intersection-union: a certificate is issued only when BOTH pass, so each
    is tested at the full delta. The Bonferroni version was strictly tighter, so
    it can never certify more coverage than this one."""
    conf, spread, err, pan, _ = _router_cohort()
    cert = RC.dual_calibrate({"error": conf, "panel": spread},
                             {"error": err, "panel": pan},
                             {"error": 0.10, "panel": 0.15}, delta=0.05)
    tight = RC.dual_calibrate({"error": conf, "panel": spread},
                              {"error": err, "panel": pan},
                              {"error": 0.10, "panel": 0.15}, delta=0.025)
    assert cert.coverage >= tight.coverage - 1e-9


def test_calibration_split_is_patient_disjoint():
    """Two nodules from one patient on opposite sides of the cut would leak the
    calibration half into the half the guarantee is reported on.

    The split carries every per-item array through `take`, so a patient-id
    column rides along and the invariant can be checked on the OUTPUT rather
    than assumed: no patient on both sides, and every patient's items whole."""
    n = 120
    clusters = np.repeat(np.arange(40), 3)
    test = {"labels": np.zeros(n, int), "probs": np.full(n, 0.5),
            "disagreement": np.zeros(n), "patient": clusters.copy()}
    for seed in range(5):
        cal, ev = PANEL.split_for_calibration(test, clusters=clusters, seed=seed)
        cal_ids, ev_ids = set(cal["patient"].tolist()), set(ev["patient"].tolist())
        assert cal_ids and ev_ids
        assert cal_ids.isdisjoint(ev_ids), "a patient landed on both sides"
        assert cal_ids | ev_ids == set(clusters.tolist())
        for pid in cal_ids:
            assert (cal["patient"] == pid).sum() == 3, "a patient was split up"
        assert len(cal["labels"]) + len(ev["labels"]) == n


def test_dual_certificate_holds_at_its_level_under_repeated_calibration():
    """The guarantee is about the PROCEDURE: across fresh calibration draws the
    certified set must breach its alpha on a fresh population at most ~delta of
    the time. The old search kept the best of every grid pair at the full delta
    and is exactly what this test exists to catch.

    The risks are flat in the score here (5.5% error everywhere, alpha 5%), the
    setting where adaptive selection over many thresholds produces false
    certificates: every certificate this procedure issues is a false one, so the
    issue rate itself is the FWER."""
    rng = np.random.default_rng(11)
    issued, trials = 0, 200
    for _ in range(trials):
        n = 400
        s_err, s_pan = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
        err = (rng.uniform(0, 1, n) < 0.055).astype(float)
        pan = (rng.uniform(0, 1, n) < 0.02).astype(float)
        cert = RC.dual_calibrate({"error": s_err, "panel": s_pan},
                                 {"error": err, "panel": pan},
                                 {"error": 0.05, "panel": 0.10},
                                 delta=0.10, n_grid=10)
        issued += cert.feasible
    assert issued / trials <= 0.10 + 0.05, f"false certificates {issued}/{trials}"


def test_a_mismatched_cluster_list_raises_instead_of_silently_downgrading():
    """`panel_battery`'s own `clusters` argument is aligned to the BORDERLINE
    subset, so passing it here is a live mistake, not a hypothetical. Falling
    back to an item-level split would weaken the guarantee while still printing
    one."""
    n = 100
    test = {"labels": np.zeros(n, int), "probs": np.full(n, 0.5),
            "disagreement": np.zeros(n)}
    with pytest.raises(ValueError, match="cut by patient"):
        PANEL.split_for_calibration(test, clusters=np.arange(40))


def test_a_panel_signal_that_is_really_confidence_residualises_to_zero():
    """The control that replaces the permuted-dispersion arm. A signal equal to
    confidence carries no panel information beyond it and must score 0; a signal
    that genuinely ranks dispersion must survive."""
    conf, spread, err, pan, d = _router_cohort(n=300, seed=3)
    out = PANEL.residual_panel_signal(
        {"confidence": conf, "fake": conf.copy(), "real": spread}, d,
        n_boot=200, seed=0)
    assert abs(out["sp_fake"]) < 1e-6, out["sp_fake"]
    assert out["sp_real"] > 0.2, out["sp_real"]
    assert out["sp_real_lo"] > 0.0, "a real panel signal should clear zero"


def test_the_dispersion_head_survives_the_mc_dropout_path():
    """END TO END, because the pieces each passed on their own and the chain
    still broke.

    `D_reg`'s head was built, trained and checkpointed, but `mc_dropout_predict`
    did not stack `dispersion`, so `summarise` never produced it,
    `predict_with_uncertainty` returned None for it, and `panel_signals` dropped
    `panel_disp`. Every table then fell back to `confidence` and the whole
    LIDC 5-fold run read as a negative result for a head nothing had looked at.

    Unit tests of each stage cannot catch that; only the composition can.
    """
    import torch
    from field_disagreement.registry import build_model
    from field_disagreement.triage import mc_dropout_predict, summarise

    torch.manual_seed(0)
    model = build_model("cnn", pretrained=False, opinion_head=True,
                        n_ratings=5, defer_head=False, dispersion_head=True)
    model.eval()
    x = torch.randn(4, 3, 64, 64)

    stacks = mc_dropout_predict(model, x, n_passes=3, seed=0)
    assert "dispersion" in stacks, "the MC loop dropped the head"
    assert stacks["dispersion"].shape == (3, 4), stacks["dispersion"].shape

    u = summarise(stacks)
    assert "dispersion" in u and u["dispersion"].shape == (4,)

    sig = PANEL.panel_signals({"probs": np.full(4, 0.5), "dispersion": u["dispersion"]})
    assert "panel_disp" in sig, "the contribution arm would ship `confidence`"


def test_an_arm_without_the_head_does_not_get_a_constant_panel_disp_column():
    """The other half: absence must stay ABSENT. A column of zeros would read as
    'measured, no effect' on a ladder rung that never had the head."""
    import torch
    from field_disagreement.registry import build_model
    from field_disagreement.triage import mc_dropout_predict

    model = build_model("cnn", pretrained=False, opinion_head=True, n_ratings=5)
    model.eval()
    stacks = mc_dropout_predict(model, torch.randn(4, 3, 64, 64), n_passes=2, seed=0)
    assert "dispersion" not in stacks
