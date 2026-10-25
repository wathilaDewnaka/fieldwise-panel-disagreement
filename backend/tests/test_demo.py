"""The notebook demo figures: they read processed items and run folders, draw one
figure each, and never train or change a result."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from backend.notebooks.shared import demo

FIELDS = ("subtlety", "sphericity", "margin")


def _cohort(tmp_path, n=6, ct=True):
    rng = np.random.default_rng(0)
    files = []
    for i in range(n):
        if ct:
            base = rng.uniform(0.3, 0.6, size=(32, 32))
            img = np.stack([base + rng.normal(0, 0.05, base.shape) for _ in range(3)])
        else:
            img = np.stack([np.full((32, 32), 0.7), np.full((32, 32), 0.4),
                            np.full((32, 32), 0.1)])
        mask = np.zeros((32, 32), np.float32)
        mask[10:20, 10:20] = 1
        agree = mask * rng.uniform(0.2, 1.0, size=mask.shape)
        f = f"item{i}.npz"
        np.savez(tmp_path / f, image=img.astype(np.float32), binary_mask=mask,
                 agreement_frac=agree.astype(np.float32), label=np.int8(i % 2),
                 disagreement=np.float32(i / 2), d_profile=rng.uniform(0, 2, 3).astype(np.float32))
        files.append(f)
    df = pd.DataFrame({"file": files, "patient_id": [f"p{i // 2}" for i in range(n)],
                       "label": [i % 2 for i in range(n)], "n_readers": 4})
    return df


def test_ct_and_photographs_are_told_apart(tmp_path):
    _cohort(tmp_path)
    ct = np.load(tmp_path / "item0.npz")["image"]
    assert demo.is_ct(ct)
    assert not demo.is_ct(np.stack([np.full((8, 8), 0.7), np.full((8, 8), 0.4),
                                    np.full((8, 8), 0.1)]))


def test_items_are_picked_from_least_to_most_disputed(tmp_path):
    df = _cohort(tmp_path)
    picked = demo.pick_by_disagreement(tmp_path, 3, files=df["file"])
    assert len(picked) == 3 and picked[0] == "item0.npz" and picked[-1] == "item5.npz"


def test_the_data_figures_draw(tmp_path, capsys):
    df = _cohort(tmp_path)
    demo.describe_cohort(df, tmp_path, FIELDS)
    out = capsys.readouterr().out
    assert "items 6" in out and "report fields (3)" in out and "binary_mask" in out
    files = list(df["file"][:3])
    assert demo.panel_gallery(tmp_path, files, FIELDS).axes


def _run(n=40, folds=2):
    rng = np.random.default_rng(1)
    res = {}
    for k in range(folds):
        d = rng.uniform(0, 3, n)
        D = np.stack([d + rng.normal(0, 0.3, n) for _ in FIELDS], 1)
        res[k] = {
            "good": {"test_files": [f"item{i % 6}.npz" for i in range(n)],
                     "test_labels": rng.integers(0, 2, n), "test_disagreement": d,
                     "test_d_profile": D, "test_panel_disp": d,
                     "test_panel_disp_fields": D + rng.normal(0, 0.1, D.shape)},
            "noise": {"test_files": [f"item{i % 6}.npz" for i in range(n)],
                      "test_labels": rng.integers(0, 2, n), "test_disagreement": d,
                      "test_confidence": rng.uniform(size=n)}}
    return res


def _signal(arm):
    return "panel_disp" if arm == "good" else "confidence"


def test_a_signal_that_tracks_the_panel_catches_more_contested_cases():
    res = _run()
    budgets = [0.1, 0.3, 0.5]
    good = demo.referral_capture(res, "good", _signal, 2.0, budgets)
    noise = demo.referral_capture(res, "noise", _signal, 2.0, budgets)
    assert (good > noise).all() and np.all(np.diff(good) >= 0)


def test_the_study_figures_draw(tmp_path):
    _cohort(tmp_path)
    res = _run()
    arms = {"Ours": "good", "Confidence": "noise"}
    assert demo.case_gallery(res[0], tmp_path, FIELDS, arms, _signal, n_cases=2).axes
    assert demo.referral_curve(res, arms, _signal, 2.0, [0.1, 0.3]).axes
