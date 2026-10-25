"""The plumbing the notebook imports: the names it imports are there, the
statistics behave, the Colab stager stages the cohort the config names, and the
checkpoint store keeps what a re-score needs.
"""
import ast
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


from field_disagreement import metrics as M
from backend.notebooks import shared
from backend.notebooks.shared import colab

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# what the notebook imports
# --------------------------------------------------------------------------- #


def test_the_shared_package_exports_everything_the_notebooks_import():
    """Every `from backend.notebooks.shared import ...` in a notebook resolves.

    A missing name here is a NameError several cells into a run that has already
    spent a GPU hour, so it is worth catching statically."""
    wanted = set()
    for nb_path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if "backend.notebooks.shared" not in src:
                continue
            for node in ast.walk(ast.parse(src)):
                if (isinstance(node, ast.ImportFrom)
                        and node.module == "backend.notebooks.shared"):
                    wanted |= {a.name for a in node.names}
    assert wanted, "no notebook imports from the shared package - wrong glob?"
    missing = [n for n in sorted(wanted) if not hasattr(shared, n)]
    assert not missing, f"notebooks import names the package does not export: {missing}"


def test_no_notebook_still_pastes_the_staging_or_stats_code():
    """The point of the move is that the copies are gone, not merely unused.

    A notebook that keeps its own `patient_cluster_bootstrap` or mounts Drive with
    130 lines of inline staging will drift again the moment one of them is fixed,
    which is how copies drift apart."""
    # Bodies, not names: what must be gone is a notebook's own re-implementation
    # of something the shared plumbing or the framework provides.
    banned = ("def patient_cluster_bootstrap",
              "def paired_delta(arm_a", "shutil.copytree(src, REPO",
              "def predict_full",
              # ...and the notebook does not compose the OBJECTIVE itself. The
              # losses live in `field_disagreement` and `field_disagreement.Objective`
              # assembles them from the arm's config, so an arm differs from its
              # control by the config and by nothing that was retyped.
              "def train_one_epoch",
              "triage.opinion_loss(", "triage.deferral_ce_loss(")
    for nb_path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        src = nb_path.read_text(encoding="utf-8")
        for phrase in banned:
            assert phrase not in src, f"{nb_path.name} still carries {phrase!r}"


# --------------------------------------------------------------------------- #
# paired + clustered statistics
# --------------------------------------------------------------------------- #
def test_paired_auc_bootstrap_brackets_zero_for_identical_scores():
    """The parity check has to say 'no difference' when there is none."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    s = rng.random(60)
    r = M.paired_auc_bootstrap(y, s, s, np.arange(60), n_boot=200, seed=0)
    assert r["delta"] == pytest.approx(0.0)
    assert r["lo"] <= 0 <= r["hi"]


# --------------------------------------------------------------------------- #
# the evaluation passes
# --------------------------------------------------------------------------- #
def _batch(n=4, size=8):
    """One batch in the loader's contract: image, mask, agreement surface,
    disagreement and file id."""
    rng = np.random.default_rng(0)
    frac = np.zeros((n, size, size), dtype=np.float32)
    frac[:, 2:6, 2:6] = rng.uniform(0.2, 1.0, (n, 4, 4))
    mask = (frac > 0).astype(np.float32)
    return {"image": torch.zeros(n, 1, size, size),
            "binary_mask": torch.tensor(mask),
            "agreement_frac": torch.tensor(frac),
            "agreement_map": torch.tensor(frac),
            "disagreement": torch.tensor([0.0, 0.9, 0.1, 0.8]),
            "label": torch.tensor([0, 1, 0, 1]),
            "opinion_dist": torch.tensor(np.full((n, 5), 0.2, dtype=np.float32)),
            "file": [f"s{i}" for i in range(n)]}


def test_panel_stats_needs_no_forward_pass():
    b = _batch()
    q, y = shared.panel_stats([b])
    assert q.shape == (4, 5) and y.tolist() == [0, 1, 0, 1]


# --------------------------------------------------------------------------- #
# checkpoint retention
# --------------------------------------------------------------------------- #
def test_a_finished_unit_keeps_what_a_re_score_needs_when_asked(tmp_path):
    """`Runner.fit` returns early - without training an epoch - only on a unit
    whose CHECKPOINT says it already reached the epoch cap, and `load_best`
    needs the .pt. Dropping both on completion is therefore not a disk
    optimisation, it silently turns every re-score into a full retrain."""
    store = shared.CheckpointStore(tmp_path, "cnn", "cpu", use_drive=False,
                                   keep_best=True, keep_ckpt=True)
    for kind in ("ckpt", "best.pt"):
        store.path("H_fld_fold0", kind).write_text("x")

    store.cache_result("H_fld_fold0", {"auc": 0.9})

    assert store.path("H_fld_fold0", "result.json").exists()
    assert store.path("H_fld_fold0", "ckpt").exists()       # fit() resumes off this
    assert store.path("H_fld_fold0", "best.pt").exists()    # load_best reads this


def test_a_finished_unit_still_drops_its_resume_state_by_default(tmp_path):
    """The default is unchanged: every other notebook shares this store and the
    files are what eat a Drive quota."""
    store = shared.CheckpointStore(tmp_path, "cnn", "cpu", use_drive=False)
    for kind in ("ckpt", "best.pt"):
        store.path("H_fld_fold0", kind).write_text("x")

    store.cache_result("H_fld_fold0", {"auc": 0.9})

    assert store.path("H_fld_fold0", "result.json").exists()
    assert not store.path("H_fld_fold0", "ckpt").exists()
    assert not store.path("H_fld_fold0", "best.pt").exists()


def test_keep_best_alone_does_not_buy_a_re_score(tmp_path):
    """Guards the half-fix. `keep_best` was already available and would look
    like enough - but with the checkpoint gone `fit` starts at epoch 0 and
    retrains, then overwrites the very weights that were kept."""
    store = shared.CheckpointStore(tmp_path, "cnn", "cpu", use_drive=False,
                                   keep_best=True)
    for kind in ("ckpt", "best.pt"):
        store.path("H_fld_fold0", kind).write_text("x")

    store.cache_result("H_fld_fold0", {"auc": 0.9})

    assert store.path("H_fld_fold0", "best.pt").exists()
    assert not store.path("H_fld_fold0", "ckpt").exists()


# --------------------------------------------------------------------------- #
# Colab staging
# --------------------------------------------------------------------------- #
def _fake_drive(tmp_path, dataset="lidc", stem="lidc-idri", zipped=True):
    """A minimal uploaded-to-Drive project: enough real config for `stage` to
    resolve a cohort, plus one zipped processed folder."""
    src = tmp_path / "field-disagreement-prediction"
    (src / "configs" / "notebooks").mkdir(parents=True)
    (src / "data" / "processed").mkdir(parents=True)
    (src / "pyproject.toml").write_text("[project]\nname='x'\n")
    (src / "configs" / "notebooks" / "base.yaml").write_text(f"dataset: {dataset}\n")

    samples = tmp_path / "build" / stem
    samples.mkdir(parents=True)
    (samples / "index.csv").write_text("file\na.npz\n")
    (samples / "a.npz").write_bytes(b"\x00")
    if zipped:
        with zipfile.ZipFile(src / "data" / "processed" / f"{stem}.zip", "w") as z:
            z.write(samples / "index.csv", "index.csv")
            z.write(samples / "a.npz", "a.npz")
    else:
        import shutil
        shutil.copytree(samples, src / "data" / "processed" / stem)
    return src


def _paths_for(src, key, value):
    return {"lidc_raw": str(src), "processed": str(value), key: str(value),
            "runs": str(src / "data" / "runs")}


def test_stage_unpacks_the_cohort_the_config_names(tmp_path, monkeypatch):
    """Staging must follow `dataset:`, not whichever archive happens to sort first. Two cohorts are on
    Drive here and only the named one may be unpacked."""
    src = _fake_drive(tmp_path, dataset="chaksu", stem="chaksu")
    with zipfile.ZipFile(src / "data" / "processed" / "lidc-idri.zip", "w") as z:
        z.writestr("index.csv", "file\n")          # sorts first; must be ignored

    paths_file = tmp_path / "paths.yaml"
    paths_file.write_text(yaml.safe_dump(
        _paths_for(src, "chaksu_processed", tmp_path / "elsewhere" / "chaksu")))
    monkeypatch.setenv("FDP_PATHS", str(paths_file))
    monkeypatch.chdir(tmp_path)

    repo, data_root = tmp_path / "repo", tmp_path / "local"
    info = colab.stage(src, repo, data_root, activate=False, quiet=True)

    assert info.dataset == "chaksu"
    assert info.data_dir.parts[-1] == "chaksu"
    assert (data_root / "chaksu").exists()
    assert not (data_root / "lidc-idri").exists()


def test_stage_writes_a_paths_file_pointing_at_local_samples_and_drive_runs(
        tmp_path, monkeypatch):
    """The one machine-specific file. Samples must be read off local disk (the
    Drive mount is far too slow to train from) and results written BACK to Drive
    (the local disk is wiped on disconnect)."""
    src = _fake_drive(tmp_path)
    paths_file = tmp_path / "paths.yaml"
    paths_file.write_text(yaml.safe_dump(
        _paths_for(src, "processed", tmp_path / "elsewhere" / "lidc-idri")))
    monkeypatch.setenv("FDP_PATHS", str(paths_file))
    monkeypatch.chdir(tmp_path)

    repo, data_root = tmp_path / "repo", tmp_path / "local"
    colab.stage(src, repo, data_root, activate=False, quiet=True)

    written = yaml.safe_load((repo / "configs" / "paths.yaml").read_text())
    assert Path(written["processed"]).is_relative_to(data_root)
    assert Path(written["runs"]).is_relative_to(src)


def test_stage_copies_the_code_but_never_the_data_directory(tmp_path, monkeypatch):
    """data/ is staged separately, so copying it here would move the archive
    twice - once as a zip and once unpacked."""
    src = _fake_drive(tmp_path)
    (src / ".venv").mkdir()
    (src / ".venv" / "huge").write_text("x" * 100)
    paths_file = tmp_path / "paths.yaml"
    paths_file.write_text(yaml.safe_dump(
        _paths_for(src, "processed", tmp_path / "elsewhere" / "lidc-idri")))
    monkeypatch.setenv("FDP_PATHS", str(paths_file))
    monkeypatch.chdir(tmp_path)

    repo = tmp_path / "repo"
    colab.stage(src, repo, tmp_path / "local", activate=False, quiet=True)

    assert (repo / "configs" / "notebooks" / "base.yaml").exists()
    assert not (repo / "data").exists()
    assert not (repo / ".venv").exists()


def test_activate_puts_the_staged_copy_ahead_of_the_drive_copy(tmp_path, monkeypatch):
    """The step that makes the staged copy authoritative.

    Without it every later `import backend...` keeps resolving to Drive - whose
    configs/paths.yaml is the laptop's, so training would read sample paths that
    do not exist on the VM while the freshly written VM paths.yaml sat unused."""
    drive, repo = tmp_path / "drive", tmp_path / "repo"
    repo.mkdir()
    monkeypatch.syspath_prepend(str(drive))
    monkeypatch.setattr(colab.sys, "modules", dict(colab.sys.modules))
    monkeypatch.chdir(tmp_path)

    colab._activate(repo, drive)

    assert colab.sys.path[0] == str(repo)
    assert str(drive) not in colab.sys.path
    assert not [m for m in colab.sys.modules
                if m.split(".")[0] in ("backend", "field_disagreement")]
    assert Path.cwd() == repo.resolve()


def test_stage_says_which_folder_it_wanted_when_drive_root_is_wrong(tmp_path):
    (tmp_path / "Something Else").mkdir()
    with pytest.raises(FileNotFoundError, match="pyproject.toml"):
        colab.stage(tmp_path / "field-disagreement-prediction", tmp_path / "repo", tmp_path / "local")


def test_stage_names_the_missing_archive_and_the_config_switch(tmp_path, monkeypatch):
    """The failure a user actually hits: the cohort is selected but not uploaded.
    The message has to name both the file to add and the line to change."""
    src = _fake_drive(tmp_path, dataset="chaksu", stem="chaksu")
    (src / "data" / "processed" / "chaksu.zip").unlink()
    paths_file = tmp_path / "paths.yaml"
    paths_file.write_text(yaml.safe_dump(
        _paths_for(src, "chaksu_processed", tmp_path / "elsewhere" / "chaksu")))
    monkeypatch.setenv("FDP_PATHS", str(paths_file))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as e:
        colab.stage(src, tmp_path / "repo", tmp_path / "local", activate=False, quiet=True)
    assert "chaksu.zip" in str(e.value)
    assert "configs/notebooks/base.yaml" in str(e.value)


# --------------------------------------------------------------------------- #
# the coverage guard: items missing from one arm must not silently bias a delta
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# united-vs-divided separation
# --------------------------------------------------------------------------- #
def test_disagreement_separation_splits_united_from_divided_by_default():
    """`nonzero` is "every annotator said the same" vs "they did not"; a median
    split asks a weaker question."""
    d = np.array([0.0] * 20 + [0.5] * 20)
    sep = M.disagreement_separation(d, {"perfect": d.copy()})
    assert (sep["n_united"], sep["n_divided"]) == (20, 20)
    assert sep["signals"]["perfect"]["auc"] == pytest.approx(1.0)

    mixed = np.array([0.0] * 10 + [0.1] * 20 + [0.9] * 10)
    assert M.disagreement_separation(mixed, {})["n_united"] == 10
    assert M.disagreement_separation(mixed, {}, split="median")["n_united"] == 30


def test_an_unscoreable_signal_is_left_nan_rather_than_reported_as_chance():
    """A constant signal means the arm never built that head. Printing 0.5 would
    read as "measured, no effect" instead of "not available here"."""
    d = np.array([0.0] * 10 + [1.0] * 10)
    assert np.isnan(M.disagreement_separation(d, {"flat": np.ones(20)})
                    ["signals"]["flat"]["auc"])
    # ... and so does a cohort with only one side of the contrast.
    assert np.isnan(M.disagreement_separation(np.zeros(20), {"s": np.arange(20.0)})
                    ["signals"]["s"]["auc"])


# --------------------------------------------------------------------------- #
# pooling per-fold results, and formatting one aggregate cell
# --------------------------------------------------------------------------- #
def _by_arm():
    return {
        "H_fld": [{"fold": 0, "per": {"a": 0.8, "b": 0.6}},
                   {"fold": 1, "per": {"c": 0.7}}],
        "none":   [{"fold": 0, "per": {"a": 0.5, "b": 0.5}},
                   {"fold": 1, "per": {"c": 0.4}}],
    }


def test_pool_predictions_reads_the_stored_per_item_records():
    folds = [{"test_preds": [{"file": "a", "prob": 0.9, "label": 1}]},
             {"test_preds": [{"file": "b", "prob": 0.2, "label": 0}]}]
    assert shared.pool_predictions(folds) == {"a": (0.9, 1), "b": (0.2, 0)}


def test_flatten_moves_a_nested_metric_to_the_top_level():
    """`aggregate_folds` walks the top level only, so a nested metric is
    invisible to it unless it is flattened where it is recorded."""
    assert shared.flatten("policy", {"aurc": 0.1, "name": "skip"}) == {"policy.aurc": 0.1}


def test_checkpoint_write_survives_a_transient_mount_failure(tmp_path, monkeypatch):
    """Colab/Drive: a `ckpt.tmp` write died with `open file failed
    ... No such file or directory` seconds after `best.pt` wrote into the same
    folder. One flaky write must be retried, not end the run - and must leave no
    `.tmp` behind and a checkpoint that loads."""
    import shutil as _shutil

    from backend.notebooks.shared import checkpoints as C

    store = shared.CheckpointStore(tmp_path / "ck", "resnet50", "cpu", use_drive=False)
    real, calls = _shutil.copyfile, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(2, "No such file or directory")
        return real(src, dst)

    monkeypatch.setattr(C.shutil, "copyfile", flaky)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    model = torch.nn.Linear(3, 1)
    store.save_best("fold0", model)
    assert calls["n"] == 2
    assert not list((tmp_path / "ck").glob("*.tmp"))
    fresh = torch.nn.Linear(3, 1)
    store.load_best("fold0", fresh)
    assert torch.equal(fresh.weight, model.weight)
