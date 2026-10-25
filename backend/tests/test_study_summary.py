"""The plain-words results page reads the scorers' numbers."""
from backend.notebooks.shared import study_summary as CS


def _q1(lo_dreg=0.1, parity_hi=0.01):
    rho = {"H_fld": [0.4] * 5, "A_cls": [0.05] * 5, "D_reg": [0.1] * 5}
    return {"primary": {"rho": rho},
            "pooled": {"A_cls": {"delta": .35, "lo": .3, "hi": .4},
                       "D_reg": {"delta": .3, "lo": lo_dreg, "hi": .4}},
            "won": {"A_cls": 5, "D_reg": 4},
            "parity": {"A_cls": {"delta": -.001, "ci95_lo": -.01, "ci95_hi": parity_hi}}}


def test_all_pass_reads_as_pass():
    text = CS.build(_q1(), "lidc", 42)
    assert "✅ 2/2 rivals beaten" in text and "beats 2 of 2" in text
    assert "DUP: predicts one overall disagreement" in text      # plain names, not codes


def test_a_ci_touching_zero_is_not_a_win():
    assert "❌ 1/2 rivals beaten" in CS.build(_q1(lo_dreg=-0.001), "chaksu")


def test_auc_loss_is_reported_with_its_size():
    assert "significantly below Classifier confidence (-0.0010 AUC)" in CS.build(_q1(parity_hi=-0.001), "lidc")


def test_write_puts_the_page_in_the_run_folder(tmp_path):
    path = CS.write(_q1(), "lidc", tmp_path, seed=42)
    assert path.name == "SUMMARY_lidc.md" and "seed 42" in path.read_text(encoding="utf-8")


def test_seeds_trained_in_separate_folders_pool(tmp_path):
    """A seed run in its own session lands in its own dated folder; listing the
    folders pools them, and a file present in both is read from the first folder."""
    import json
    from backend.notebooks.shared import study_results as R
    a, b = tmp_path / "a" / "checkpoints", tmp_path / "b" / "checkpoints"
    a.mkdir(parents=True); b.mkdir(parents=True)
    (a / "resnet50_H_fld_fold0.result.json").write_text(json.dumps({"auc": 0.9}))
    (b / "resnet50_H_fld_s7_fold0.result.json").write_text(json.dumps({"auc": 0.8}))
    dirs = [tmp_path / "a", tmp_path / "b"]
    assert R.seeds_in(dirs) == [42, 7]
    assert R.load_run(dirs, 7)[0]["H_fld"]["auc"] == 0.8
    (b / "resnet50_H_fld_fold0.result.json").write_text(json.dumps({"auc": 0.1}))
    # the same file in two folders: the earlier folder (a rerun) is the one read
    assert R.load_run(dirs, 42)[0]["H_fld"]["auc"] == 0.9
    assert R.load_run(dirs[::-1], 42)[0]["H_fld"]["auc"] == 0.1
