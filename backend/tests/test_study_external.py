"""The external test: LIDC-trained networks scored, unchanged, on an external cohort.

Small random networks stand in for the trained arms (saved as best.pt in the run
folder's layout), and synthetic files in the lidc_report format stand in for
LNDb, so the whole path - architecture rebuild, weight load, prediction, the
result files and the shared-draw report - runs on CPU in seconds."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from backend.config import load
from backend.notebooks.shared import study_external as EXT
from backend.notebooks.shared import study_summary as CS

ARMS = ("H_fld", "A_cls", "D_reg", "U_edl")


def _cohort(tmp_path, n=12, seed=0):
    rng = np.random.default_rng(seed)
    d = tmp_path / "lndb_report"
    d.mkdir()
    rows = []
    for i in range(n):
        mask = np.zeros((64, 64), np.uint8)
        mask[28:36, 28:36] = 1
        prof = rng.gamma(2.0, 0.5, 7).astype(np.float32)
        lab = [0, 1, -1][i % 3]
        f = f"LNDb-{i // 2:04d}_n{i % 2:02d}.npz"
        np.savez_compressed(
            d / f, image=rng.random((3, 64, 64)).astype(np.float32),
            agreement_map=mask * 2, agreement_frac=mask.astype(np.float32),
            binary_mask=mask, malignancy=np.array([2, 4], np.int8),
            opinion_dist=np.array([0, .5, 0, .5, 0], np.float32),
            disagreement=np.float32(prof.mean()), n_readers=np.int8(2),
            label=np.int8(lab), d_profile=prof, d_single=np.float32(prof[-1]),
            patient_id=f"LNDb-{i // 2:04d}")
        rows.append({"file": f, "patient_id": f"LNDb-{i // 2:04d}", "label": lab})
    pd.DataFrame(rows).to_csv(d / "index.csv", index=False)
    return d


def _cfg():
    cfg = load("notebooks/study")
    cfg.model, cfg.n_ratings, cfg.mc_passes, cfg.pretrained = "cnn", 5, 0, False
    cfg.num_workers = 0
    return cfg


def _run(tmp_path, cfg):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    for arm in ARMS:
        for k in range(2):
            torch.manual_seed(k)
            net = EXT._network(cfg, arm, 7)
            torch.save(net.state_dict(), EXT._weights(run, cfg, arm, k, 42))
            (run / "checkpoints" / f"cnn_{arm}_fold{k}.result.json").write_text(
                json.dumps({"conf_qhat": 0.5}))
    return run


def test_score_writes_one_result_per_network_on_the_labelled_items(tmp_path):
    cfg = _cfg()
    ext = _cohort(tmp_path)
    out = EXT.score(cfg, _run(tmp_path, cfg), ext, "cpu", arms=ARMS, folds=range(2))
    res = EXT.load(out)
    assert set(res) == set(ARMS) and all(sorted(v) == [(42, 0), (42, 1)] for v in res.values())
    r = res["H_fld"][(42, 0)]
    assert r["n_items"] == 8                               # the 4 median-3 items left out
    assert np.asarray(r["test_panel_disp_fields"]).shape == (8, 7)
    assert set(r["test_labels"]) <= {0.0, 1.0}
    assert "test_epistemic" in res["U_edl"][(42, 0)]             # vacuity, as on LIDC
    assert r["conf_qhat_from_lidc"] == 0.5                 # LIDC's threshold, not refitted


def test_include_ambiguous_scores_every_panel(tmp_path):
    cfg = _cfg()
    out = EXT.score(cfg, _run(tmp_path, cfg), _cohort(tmp_path), "cpu", arms=("H_fld",),
                    folds=range(1), include_ambiguous=True)
    assert out.name.endswith("_all") and EXT.load(out)["H_fld"][(42, 0)]["n_items"] == 12


def test_missing_weights_are_skipped_not_fatal(tmp_path, capsys):
    cfg = _cfg()
    EXT.score(cfg, _run(tmp_path, cfg), _cohort(tmp_path), "cpu", arms=("H_fld",),
              folds=range(3))
    assert "no LIDC weights for 1 networks" in capsys.readouterr().out


def test_report_pools_networks_on_shared_draws_and_feeds_the_page(tmp_path):
    cfg = _cfg()
    run = _run(tmp_path, cfg)
    EXT.score(cfg, run, _cohort(tmp_path), "cpu", arms=ARMS, folds=range(2))
    rep = EXT.report(run, rivals=("A_cls", "D_reg", "U_edl"), n_boot=50, verbose=False)
    assert rep["nets"] == [(42, 0), (42, 1)] and rep["n_items"] == 8 and rep["n_patients"] == 6
    for a in ("A_cls", "D_reg", "U_edl"):
        c = rep["contrast"][a]
        assert c["lo"] <= c["delta"] <= c["hi"] and c["n_nets"] == 2
    assert np.isclose(rep["rho"]["H_fld"], np.mean(list(rep["per_net"]["H_fld"].values())))
    page = CS.build(None, "lidc", 42, external_report=rep)
    assert "## 6. External test: LNDb" in page and "Networks won" in page
