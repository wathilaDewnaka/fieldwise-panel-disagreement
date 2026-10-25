"""The study's claim `H_fld` (the capped per-field disagreement term plus the field
histograms) and its control `H_fldp` (field_disagreement/triage/ladder.py)."""
import numpy as np
import pytest
import torch

from field_disagreement.config import DeferralConfig, ExperimentConfig, InvalidConfig
from field_disagreement.objectives import Objective
from field_disagreement.registry import build_model
from field_disagreement.triage import ladder as L

P, K, B = 7, 5, 4


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.rand(B, P, K, generator=g)
    return {"image": torch.randn(B, 1, 64, 64, generator=g),
            "label": torch.tensor([0, 1, 0, 1]),
            "opinion_dist": torch.softmax(torch.randn(B, K, generator=g), -1),
            "disagreement": torch.rand(B, generator=g),
            "d_profile": torch.rand(B, P, generator=g),
            "opinion_fields": h / h.sum(-1, keepdim=True)}


def _exp():
    e = ExperimentConfig()
    return e


def _objective(arm):
    cfg = L.ladder_config(_exp(), arm)
    ref = np.sort(np.random.default_rng(0).random((50, P)), axis=0).astype(np.float32)
    return cfg, Objective(cfg, cls_loss=torch.nn.BCEWithLogitsLoss(), r_ref=ref)


def test_the_claim_and_its_control_share_the_one_weight_and_the_control_shuffles():
    h, hp = (L.ladder_config(ExperimentConfig(), a) for a in ("H_fld", "H_fldp"))
    assert h.deferral.weight == hp.deferral.weight == ExperimentConfig().deferral.weight == 0.3
    assert h.deferral.aux_fields and hp.deferral.aux_fields
    assert h.opinion.panel_shuffle == "none" and hp.opinion.panel_shuffle == "all"
    assert L.CLAIM == "H_fld" and L.CONTROL == "H_fldp"


def test_aux_fields_needs_the_per_field_target():
    with pytest.raises(InvalidConfig):
        DeferralConfig(aux_fields=True, regress_target="disagreement")


def test_h_fld_trains_both_parts_of_its_head():
    cfg, obj = _objective("H_fld")
    m = build_model("cnn", in_ch=1, opinion_head=True, n_ratings=K,
                    dispersion_head=P + P * K)
    b = _batch()
    out = m(b["image"], return_all=True)
    loss, _ = obj(m, b, out, epoch=cfg.epochs, device="cpu",
                  rng=np.random.default_rng(0))
    loss.backward()
    g = m.dispersion.weight.grad
    assert g[:P].abs().sum() > 0 and g[P:].abs().sum() > 0


def test_evaluation_keeps_only_the_per_field_dispersions():
    import types
    from backend.notebooks.shared.study_arms import _drop_aux
    pred = {"dispersion": np.arange(3 * (P + P * K), dtype=float).reshape(3, -1)}
    _drop_aux(pred, "H_fld", types.SimpleNamespace(n_ratings=K))
    assert pred["dispersion"].shape == (3, P)


def test_two_sided_triage_automates_the_declared_share():
    from backend.notebooks.shared import study_results as R
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    r = {"test_probs": (0.6 * y + 0.4 * rng.random(n)).tolist(), "test_labels": y.tolist(),
         "test_disagreement": rng.random(n).tolist(), "test_confidence": rng.random(n).tolist()}
    t = R.triage_fold(r, "A_cls", 0.8)
    assert abs(t["automated"] - (R.TRIAGE_CLEAR + R.TRIAGE_FLAG)) < 0.01
    assert t["automated"] > 0.5


def test_h_fldp_shuffles_every_panel_target_including_the_histograms():
    """H_fldp is H_fld's information control: one permutation moves the opinion
    histogram, the per-field dispersions AND the auxiliary field histograms
    together, so the image keeps none of its own panel."""
    b = _batch(2)
    torch.manual_seed(0)
    m = build_model("cnn", in_ch=1, opinion_head=True, n_ratings=K,
                    dispersion_head=P + P * K)
    out = m(b["image"], return_all=True)
    cfg, o = _objective("H_fldp")
    _, oh = _objective("H_fld")
    e = cfg.epochs
    same = oh(m, b, out, e, "cpu", rng=np.random.default_rng(3))[0]
    shuf = o(m, b, out, e, "cpu", rng=np.random.default_rng(3))[0]
    assert not torch.isclose(same, shuf)
    # with the histograms alone permuted back, the loss would differ again
    src = __import__("inspect").getsource(type(o).__call__)
    assert "h_tgt = h_tgt[panel_perm]" in src and "d = d[panel_perm]" in src


def test_the_declared_criteria_and_their_pooling():
    from backend.notebooks.shared import study_results as R
    rule = R.study_rule()                     # configs/notebooks/study.yaml
    assert (rule.veto, rule.safety_net, rule.miss_margin) == (0.50, 0.30, 5.0)

    def sv(auto, miss, hi):
        return {"served": {"auto": auto, "missed_per_1000": miss, "high_neg": hi}}
    ours = sv([.5] * 5, [5, 6, 5, 6, 5], [.10, .11, .10, .12, .10])
    sp = {"claim_v4": ours, "claim": sv([.5] * 5, [99] * 5, [.9] * 5),
          "signal": {"C_emd": sv([.55] * 5, [5, 6, 6, 5, 6], [.15, .16, .15, .17, .16]),
                     "H_fldp": sv([.5] * 5, [0] * 5, [0] * 5)}}
    c = R.triage_contrasts(sp)
    assert set(c) == {"C_emd"}                         # prior art only, never the control
    assert c["C_emd"]["disputed"][3] and c["C_emd"]["disputed"][4] == 5
    assert c["C_emd"]["missed"][3]                     # judged on claim_v4, not the learned row
    p = R.pool_served_priority([sp, sp])
    assert len(p["claim_v4"]["served"]["auto"]) == 10
    assert R.triage_contrasts(p)["C_emd"]["n"] == 10


def test_every_result_carries_its_training_history():
    """The per-epoch record is what lets an over-fitted checkpoint be seen after
    the session is gone; the report's overfitting check reads it."""
    import inspect
    from field_disagreement.engine import runner as ER
    from backend.notebooks.shared import study_results as R
    from backend.notebooks.shared import runner as SR
    assert "history" in {f.name for f in __import__("dataclasses").fields(ER.Fold)}
    assert '"history"' in inspect.getsource(SR)
    res = {k: {"H_fld": {"auc": 0.90, "best_val_auc": 0.93, "selection": {"epoch": 14},
                         "history": [{"epoch": e, "loss": 1.0, "val_auc": v,
                                      "val_auc_smooth": v} for e, v in
                                     ((11, 0.91), (12, 0.93), (13, 0.92))]}}
           for k in range(2)}
    oc = R.overfit_check(res)["H_fld"]
    assert oc["kept"] == [14, 14] and oc["peak"] == [12, 12]
    assert np.allclose(oc["gap"], 0.03)


@pytest.mark.parametrize("arm", ["H_fld", "P_emd", "C_emd", "D_reg", "A_cls"])
def test_the_val_monitor_logs_each_arms_signal_and_leaves_the_rng_alone(arm):
    """The per-epoch disagreement log runs on every kind of head, and does not
    move the training draw."""
    import types
    from backend.notebooks.shared import study_arms as SARM
    spec = L.TRAINED_ARMS[arm]
    width = (P if spec.target == "d_profile" else P * K if spec.target == "opinion_fields"
             else int(spec.regress)) + (P * K if spec.aux_fields else 0)
    torch.manual_seed(0)
    m = build_model("cnn", in_ch=1, opinion_head=spec.opinion != "none", n_ratings=K,
                    dispersion_head=width)
    g = torch.Generator().manual_seed(1)
    val = [{"image": torch.randn(8, 1, 64, 64, generator=g),
            "label": torch.randint(0, 2, (8,), generator=g).float(),
            "d_profile": torch.rand(8, P, generator=g),
            "disagreement": torch.rand(8, generator=g),
            "file": [f"f{i}_{j}" for j in range(8)]} for i in range(3)]
    ctx = types.SimpleNamespace(model=m, val=val, device=torch.device("cpu"))
    cfg = types.SimpleNamespace(n_ratings=K, seed=0)
    before = torch.get_rng_state()
    out = SARM.val_monitor(arm, cfg, torch.device("cpu"))(ctx)
    assert torch.equal(before, torch.get_rng_state())
    assert set(out) == {"val_rho_d", "val_rho_fields"}
    assert all(np.isfinite(v) for v in out.values()), out


def test_the_served_rule_is_one_framework_object_set_from_config():
    from field_disagreement.triage import ServedRule, served_refer, veto_cut
    d = ServedRule()                                   # the deployed rule
    assert (d.veto, d.safety_net, d.miss) == (0.40, 0.0, 0.04)
    r = ServedRule.from_config({"veto": 0.5, "safety_net": 0.3})
    assert (r.veto, r.safety_net, r.extreme) == (0.5, 0.3, d.extreme)
    with pytest.raises(ValueError):
        ServedRule.from_config({"safty_net": 0.3})     # a typo never falls back
    with pytest.raises(ValueError):
        ServedRule(veto=1.2)
    # the safety net refers negative calls only, by the net signal
    p = np.array([0.05, 0.2, 0.3, 0.8])
    conf = 1 - 2 * np.abs(p - 0.5)
    net = np.array([0.9, 0.1, 0.5, 0.99])
    out = served_refer(conf, p, conf_cut=2.0, net=net, net_cut=veto_cut(net, p, 0.34))
    assert out.tolist() == [True, False, False, False]


def test_the_panel_weight_and_cohort_facts_come_from_config_and_registry():
    from field_disagreement.config import ExperimentConfig
    from backend.config import load
    from backend.domains.datasets import DATASETS
    from backend.notebooks.shared import study_results as R
    e = ExperimentConfig.from_namespace(load("notebooks/study"))
    assert L.ladder_config(e, "H_fld").deferral.weight == e.deferral.weight == 0.3
    assert L.ladder_config(e, "D_reg").deferral.weight == e.deferral.weight
    for s in DATASETS.values():
        if s.report_level:
            assert len(s.field_names) == s.n_fields and s.title, s.name
    assert R.FIELD_NAMES[7][-1] == "malignancy" and R.FIELD_NAMES[10][1] == "VCDR"


def test_the_floor_check_passes_a_learned_head_and_raises_a_late_one():
    from backend.notebooks.shared import study_results as R

    def unit(kept, rhos):
        return {"selection": {"epoch": kept},
                "history": [{"epoch": e, "val_rho_fields": r} for e, r in enumerate(rhos, 1)]}
    flat = [0.1, 0.3, 0.4, 0.42, 0.41, 0.42, 0.42, 0.41, 0.42, 0.42, 0.42, 0.41, 0.42]
    late = [0.0] * 10 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6]
    ok = R.floor_check({0: {"H_fld": unit(11, flat), "A_cls": unit(11, late)}},
                       floor=10, share=0.9, max_floor=30, window=1)
    assert ok["ok"] and ok["floor"] == 10 and "A_cls" not in ok["units"]   # no panel head
    bad = R.floor_check({0: {"H_fld": unit(11, flat), "D_reg": unit(11, late)}},
                        floor=10, share=0.9, max_floor=30, window=1)
    assert not bad["ok"] and bad["floor"] == 16          # D_reg reached 90% at epoch 16
    assert bad["underfit"] == [("D_reg", 0)]
    # learned early, then fell: kept AFTER the peak - drift, which a floor cannot fix
    fell = [0.1, 0.5, 0.6, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    d = R.floor_check({0: {"B_kl": unit(11, fell)}}, floor=10, share=0.9,
                      max_floor=30, window=1)
    assert d["ok"] and d["drifted"] == [("B_kl", 0)] and d["floor"] == 10
    # the shuffled control has no signal to learn: never in the check
    c = R.floor_check({0: {"H_fldp": unit(11, late)}}, floor=10, share=0.9,
                      max_floor=30, window=1)
    assert c["ok"] and not c["units"]
