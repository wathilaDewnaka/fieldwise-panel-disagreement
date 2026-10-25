"""The scorer (`backend/notebooks/shared/study_results.py`) on a synthetic run
folder: which files it reads, the per-field primary contrast (folds won and
pooled), the report-level secondary and the parity table."""
from __future__ import annotations

import json

import numpy as np
import pytest

from backend.notebooks.shared import study_results as SRES


def _write_run(tmp_path, strength: dict, n=120, folds=5, seed=0, fields=3,
               per_field=()):
    """Every arm's shipped signal = d * strength + noise, on shared items; d is
    the mean of `fields` per-field disagreements. Arms in `per_field` also store
    one signal per field (field_j * strength + noise)."""
    rng = np.random.default_rng(seed)
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    for k in range(folds):
        prof = rng.gamma(2.0, 0.5, (n, fields))
        d = prof.mean(1)
        clusters = [f"p{i // 2}" for i in range(n)]
        for arm, s in strength.items():
            sig = s * (d - d.mean()) / d.std() + rng.normal(size=n)
            res = {"auc": 0.9 + 0.001 * k, "test_disagreement": d.tolist(),
                   "test_clusters": clusters,
                   f"test_{SRES.ships(arm)}": sig.tolist(),
                   f"leak_{SRES.ships(arm)}": 0.15, "leak_base": 0.2}
            if arm in per_field:
                z = (prof - prof.mean(0)) / prof.std(0)
                res["test_panel_disp_fields"] = (s * z + rng.normal(size=z.shape)).tolist()
            if arm == next(iter(strength)):     # one arm per fold carries the profile
                res["test_d_profile"] = prof.tolist()
            if SRES.ships(arm) != "confidence":   # never stored for confidence
                res[f"sp_{SRES.ships(arm)}"] = 0.1
            (ck / f"cnn_{arm}_fold{k}.result.json").write_text(json.dumps(res))
    return tmp_path


def test_load_run_reads_every_arm_and_skips_seed_replications(tmp_path):
    run = _write_run(tmp_path, {"A_cls": 0.0, "D_reg": 1.0}, folds=2)
    (run / "checkpoints" / "cnn_D_reg_s7_fold0.result.json").write_text("{}")
    res = SRES.load_run(run)
    assert sorted(res) == [0, 1]
    assert set(res[0]) == {"A_cls", "D_reg"}


def test_a_strong_claim_is_separated_pooled_and_a_tie_is_not(tmp_path):
    run = _write_run(tmp_path, {"A_cls": 0.0, "B_kl": 0.95, "D_reg": 1.0})
    prim = SRES.primary(SRES.load_run(run), "D_reg", rivals=("A_cls", "B_kl"), n_boot=300)
    won = {r: sum(t["delta"] > 0 for t in per) for r, per in prim["contrast"].items()}
    assert won["A_cls"] == 5
    pool = SRES.pooled(prim)
    assert pool["A_cls"]["lo"] > 0
    assert pool["B_kl"]["lo"] < pool["B_kl"]["delta"] < pool["B_kl"]["hi"]


def test_the_primary_is_paired_on_the_same_items(tmp_path):
    run = _write_run(tmp_path, {"A_cls": 0.0, "D_reg": 1.0}, folds=1)
    ck = run / "checkpoints" / "cnn_A_cls_fold0.result.json"
    r = json.loads(ck.read_text())
    for key in ("test_confidence", "test_disagreement", "test_clusters"):
        r[key] = r[key][:-1]
    ck.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="different items"):
        SRES.primary(SRES.load_run(run), "D_reg", rivals=("A_cls",), n_boot=10)


def test_a_signal_that_is_confidence_has_a_zero_semi_partial(tmp_path):
    run = _write_run(tmp_path, {"A_cls": 0.0, "D_reg": 1.0}, folds=2)
    tab = SRES.arm_table(SRES.load_run(run), d0=1.0)
    assert tab["rows"]["A_cls"]["semi_partial"] == [0.0, 0.0]
    assert tab["rows"]["D_reg"]["semi_partial"] == [0.1, 0.1]


def test_a_per_field_head_beats_one_signal_broadcast_to_every_field(tmp_path):
    # the report-level signal sees only the mean; a head per field sees each field
    run = _write_run(tmp_path, {"D_reg": 1.0, "H_fld": 1.0}, per_field=("H_fld",))
    res = SRES.load_run(run)
    assert "test_d_profile" not in res[0]["H_fld"]      # borrowed from D_reg
    prim = SRES.primary(res, rivals=("D_reg",), n_boot=300)
    assert sum(t["delta"] > 0 for t in prim["contrast"]["D_reg"]) == 5
    assert SRES.pooled(prim)["D_reg"]["lo"] > 0
    rep = SRES.report_level(res, rivals=("D_reg",), n_boot=50)
    assert rep["target"] == "report" and prim["target"] == "macro"


def test_a_scalar_arm_is_broadcast_to_every_field(tmp_path):
    run = _write_run(tmp_path, {"A_cls": 0.0, "D_reg": 1.0}, folds=1, fields=4)
    res = SRES.load_run(run)
    S = SRES._fields(res[0]["D_reg"], "D_reg", 4)
    assert S.shape == (120, 4) and (S == S[:, :1]).all()


def test_report_prints_every_section(tmp_path, capsys):
    run = _write_run(tmp_path, {a: 0.5 for a in SRES.ARMS} | {"H_fld": 1.0},
                     fields=7, per_field=("H_fld", "H_fldp", "P_emd"))
    out = SRES.report(run, d0=1.0, n_boot=200)
    text = capsys.readouterr().out
    for head in ("PRIMARY", "won", "SECONDARY 1 ", "PER FIELD", "malignancy",
                 "SECONDARY 2-5", "PRECONDITION"):
        assert head in text, head
    assert set(out["won"]) == set(out["report_won"]) == set(SRES.RIVALS)
    assert "bar" not in text.lower().split()


def test_the_served_rule_table_is_rebuilt_from_the_banked_columns():
    """The certainty cut plus the negative-call veto, cut on half of test and read
    on the other half. Needs no weights and no validation outputs."""
    from backend.notebooks.shared import study_results as R
    rng = np.random.default_rng(0)
    n = 160
    res = {}
    for k in range(2):
        p = rng.uniform(0, 1, n)
        r = {"test_probs": p.tolist(),
             "test_labels": (rng.uniform(0, 1, n) < p).astype(float).tolist(),
             "test_disagreement": rng.normal(size=n).tolist(),
             "test_confidence": (1 - 2 * np.abs(p - 0.5)).tolist(),
             "test_panel_disp": rng.normal(size=n).tolist(),
             "test_clusters": [str(i // 2) for i in range(n)], "conf_qhat": 0.9}
        res[k] = {"A_cls": dict(r), "P_emd": dict(r)}
    out = R.served(res, 1.0, arms=("A_cls", "P_emd"), n_splits=3)
    assert set(out["own"]) == set(out["signal"]) == {"A_cls", "P_emd"}
    f = out["own"]["P_emd"]
    assert set(f) == {"served", "sota", "samecut", "matched", "randveto"}
    assert len(f["served"]["missed_pos"]) == 2
    # the veto only removes auto-decisions: never more positives cleared than the same cut
    assert all(a <= b + 1e-12 for a, b in zip(f["served"]["missed_pos"],
                                              f["samecut"]["missed_pos"]))
    # A_cls ships confidence: its served rule IS the certainty cut, with no veto
    a = out["own"]["A_cls"]
    assert "randveto" not in a and a["served"]["auto"] == a["samecut"]["auto"]
