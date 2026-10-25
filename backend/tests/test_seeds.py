"""`seed:` may be one seed or a list (base.yaml): the config normalisation, the
per-seed result files of a study run, and the across-seed report."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from backend.config import _normalise_seeds, load
from backend.notebooks.shared import study_results as SRES
from backend.tests.test_study_results import _write_run


# ------------------------------------------------------------------ config
def test_one_seed_stays_one_seed():
    cfg = {"seed": 42}
    _normalise_seeds(cfg)
    assert cfg["seed"] == 42 and cfg["seeds"] == [42] and cfg["split_seed"] == 42


def test_a_list_trains_every_seed_and_reads_the_first():
    cfg = {"seed": [43, 44, 45], "split_seed": 42}
    _normalise_seeds(cfg)
    assert cfg["seed"] == 43 and cfg["seeds"] == [43, 44, 45]
    assert cfg["split_seed"] == 42          # the frozen split never follows the seeds


@pytest.mark.parametrize("bad", [[], [42, 42], [42, "43"], [True]])
def test_a_malformed_seed_list_is_refused(bad):
    with pytest.raises(ValueError):
        _normalise_seeds({"seed": bad})


def test_the_study_config_exposes_seeds_and_a_fixed_split_seed():
    cfg = load("notebooks/study")
    assert cfg.seeds[0] == cfg.seed == 42 and cfg.split_seed == 42
    cfg = load("notebooks/study", seed=[42, 43])
    assert cfg.seed == 42 and cfg.seeds == [42, 43] and cfg.split_seed == 42


# ------------------------------------------------------------------ report
def _add_seed(run, seed: int, src_seed: int):
    """Copy a second synthetic seed's result files in under `_s<seed>` names."""
    (run.parent / f"src{seed}").mkdir()
    other = _write_run(run.parent / f"src{seed}", {"A_cls": 0.0, "D_reg": 0.2, "H_fld": 1.0},
                       folds=2, seed=src_seed, per_field=("H_fld",))
    for f in (other / "checkpoints").glob("*.result.json"):
        arm, fold = SRES._TAG.search(f.name).group(1, 3)
        shutil.copy(f, run / "checkpoints" / f"cnn_{arm}_s{seed}_fold{fold}.result.json")


def test_each_seed_is_read_alone(tmp_path):
    (tmp_path / "run").mkdir()
    run = _write_run(tmp_path / "run", {"A_cls": 0.0, "D_reg": 0.2, "H_fld": 1.0},
                     folds=2, per_field=("H_fld",))
    _add_seed(run, 43, src_seed=1)
    assert SRES.seeds_in(run) == [42, 43]
    r42, r43 = SRES.load_run(run), SRES.load_run(run, 43)
    assert set(r42) == set(r43) == {0, 1}
    a = r42[0]["H_fld"]["test_panel_disp_fields"]
    b = r43[0]["H_fld"]["test_panel_disp_fields"]
    assert a != b                           # two different seeds, not one read twice


def test_the_across_seed_report_pools_paired_draws(tmp_path, capsys):
    (tmp_path / "run").mkdir()
    run = _write_run(tmp_path / "run", {"A_cls": 0.0, "D_reg": 0.2, "H_fld": 1.0},
                     folds=2, per_field=("H_fld",))
    _add_seed(run, 43, src_seed=1)
    rep = SRES.report_seeds(run, n_boot=200)
    out = capsys.readouterr().out
    assert "PRIMARY ACROSS SEEDS" in out and rep["seeds"] == [42, 43]
    t = rep["pooled"]["A_cls"]
    assert t["n_seeds"] == 2 and t["n_folds"] == 4
    assert np.isclose(t["delta"], np.mean(list(t["per_seed"].values())))
    assert t["lo"] > 0                      # a strong claim survives pooling


def test_the_notebook_loops_over_the_seeds():
    nb = json.load(open("backend/notebooks/train_evaluate.ipynb", encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "for SEED in SEEDS" in src and "RUNNER.experiment = exp" in src
    assert "SRES.report_seeds(READ, DONE)" in src
