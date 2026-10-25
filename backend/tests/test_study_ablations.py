"""One panel-term weight for every arm (`lam_def`), the optional ablation
`H_disp` (the claim without its per-field histograms), and a rerun of some arms
read in place of the study's own results for those arms."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from field_disagreement.config import ExperimentConfig
from field_disagreement.triage import ladder as L
from backend.config import load
from backend.notebooks.shared import study_results as SRES
from backend.notebooks.shared import screen_arms
from backend.tests.test_study_results import _write_run


def _deferral(arm, experiment=None):
    return L.ladder_config(experiment or ExperimentConfig(), arm).deferral


def test_every_arm_uses_the_one_configured_weight():
    cfg = load("notebooks/study")
    e = ExperimentConfig.from_namespace(cfg)
    assert e.deferral.weight == cfg.lam_def == ExperimentConfig().deferral.weight == 0.3
    for a in L.TRAINED_ARMS:
        assert _deferral(a, e).weight == cfg.lam_def, a
    one = replace(e, deferral=replace(e.deferral, weight=1.0))
    assert {_deferral(a, one).weight for a in L.TRAINED_ARMS} == {1.0}


def test_the_ablation_is_the_claim_without_its_histograms():
    h, hd = _deferral("H_fld"), _deferral("H_disp")
    assert not hd.aux_fields and replace(hd, aux_fields=True) == h
    assert L.ABLATIONS == ("H_disp",) and not set(L.ABLATIONS) & set(L.ALL_ARMS)
    assert "H_disp" in L.DISPLAY_NAMES


def test_the_ablation_trains_only_when_named():
    assert screen_arms(SimpleNamespace(screen_arms=None), L.ALL_ARMS,
                       optional=L.ABLATIONS) == list(L.ALL_ARMS)
    assert screen_arms(SimpleNamespace(screen_arms=["P_emd", "D_reg", "H_disp"]), L.ALL_ARMS,
                       optional=L.ABLATIONS) == ["P_emd", "D_reg", "H_disp"]
    with pytest.raises(ValueError):
        screen_arms(SimpleNamespace(screen_arms=["H_fldx"]), L.ALL_ARMS, optional=L.ABLATIONS)
    with pytest.raises(ValueError):
        screen_arms(SimpleNamespace(screen_arms=["H_disp"]), L.ALL_ARMS)


def test_a_rerun_replaces_the_studys_result_for_its_arms(tmp_path, capsys):
    (tmp_path / "study").mkdir()
    (tmp_path / "rerun").mkdir()
    study = _write_run(tmp_path / "study", {"H_fld": 1.0, "P_emd": 0.8, "A_cls": 0.0},
                       folds=2, per_field=("H_fld",))
    rerun = _write_run(tmp_path / "rerun", {"P_emd": 0.3}, folds=2, seed=5)
    res = SRES.load_run([rerun, study])
    for k in (0, 1):
        new = json.loads((rerun / "checkpoints" / f"cnn_P_emd_fold{k}.result.json").read_text())
        assert res[k]["P_emd"] == new and "H_fld" in res[k] and "A_cls" in res[k]
    SRES.result_files([rerun, study], verbose=True)
    assert "2 result file(s) of later folders replaced" in capsys.readouterr().out


def test_the_ablation_report_reads_the_study_and_the_ablation_folders(tmp_path, capsys):
    (tmp_path / "study").mkdir()
    (tmp_path / "abl").mkdir()
    study = _write_run(tmp_path / "study", {"H_fld": 1.0, "P_emd": 0.8, "A_cls": 0.0},
                       folds=2, per_field=("H_fld", "P_emd"))
    abl = _write_run(tmp_path / "abl", {"H_disp": 0.5}, folds=2, per_field=("H_disp",))
    rep = SRES.ablation_report([abl, study], n_boot=100)
    out = capsys.readouterr().out
    assert "ABLATION" in out
    assert list(rep["arms"]) == ["H_fld", "P_emd", "H_disp"]
    assert set(rep["per_field"]["pooled"]) == set(rep["auc"]) == {"P_emd", "H_disp"}
    assert rep["arms"]["H_fld"]["units"] == 2
    # the study's own tables never read the ablation arm
    assert SRES.RIVALS == L.PRIOR_ART + (L.CONTROL,) and not set(L.ABLATIONS) & set(SRES.ARMS)


def test_no_ablation_result_means_no_ablation_table(tmp_path, capsys):
    run = _write_run(tmp_path, {"H_fld": 1.0, "P_emd": 0.8}, folds=2, per_field=("H_fld",))
    assert SRES.ablation_report(run, n_boot=50) is None
    assert "nothing to report" in capsys.readouterr().out
