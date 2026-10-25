"""Running on Kaggle: staging from read-only inputs, restoring earlier runs, the
ImageNet weights, the session time budget and its clean stop, and the notebook's
setup cell.

Everything runs against a fake `/kaggle` tree in tmp_path; nothing needs Kaggle.
"""
from __future__ import annotations

import ast
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.notebooks.shared import kaggle, session

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ["train_evaluate"]


@pytest.fixture(autouse=True)
def _no_budget(monkeypatch):
    """Every test starts AND ends with no session deadline in the environment.

    `start_budget` writes os.environ directly, which monkeypatch does not track -
    without the teardown a deadline set here would leak into every later test
    and stop the runner tests with SessionBudgetReached."""
    import os
    monkeypatch.delenv(session.DEADLINE_ENV, raising=False)
    yield
    os.environ.pop(session.DEADLINE_ENV, None)


def _samples(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.csv").write_text("file\na.npz\n")
    (folder / "a.npz").write_bytes(b"\x00")
    return folder


def _kaggle_tree(tmp_path, monkeypatch, *, dataset="chaksu", stem="chaksu"):
    """/kaggle/input with a code dataset, two cohorts, an earlier output and weights."""
    inp = tmp_path / "input"
    code = inp / "fdp-code"
    (code / "configs" / "notebooks").mkdir(parents=True)
    (code / "backend" / "notebooks" / "shared").mkdir(parents=True)
    (code / "pyproject.toml").write_text("[project]\nname='x'\n")
    (code / "configs" / "notebooks" / "base.yaml").write_text(f"dataset: {dataset}\n")
    # a laptop's runs/ inside the code upload must never be restored
    (code / "data" / "runs" / "laptop").mkdir(parents=True)
    (code / "data" / "runs" / "laptop" / "stale.result.json").write_text("{}")

    _samples(inp / "fdp-chaksu" / stem)
    _samples(inp / "fdp-lidc" / "lidc-idri")           # another cohort: must be ignored

    prev = inp / "my-notebook" / "runs" / "study_run" / "results"
    prev.mkdir(parents=True)
    (prev / "m_H_fld_fold0.result.json").write_text('{"auc": 0.9}')

    (inp / "torch-weights").mkdir()
    for name in kaggle.weight_files():
        (inp / "torch-weights" / name).write_bytes(b"w")

    paths_file = tmp_path / "paths.yaml"
    paths_file.write_text(yaml.safe_dump({
        "lidc_raw": str(code), "processed": str(tmp_path / "laptop" / stem),
        f"{dataset}_processed": str(tmp_path / "laptop" / stem),
        "runs": str(tmp_path / "laptop" / "runs")}))
    monkeypatch.setenv("FDP_PATHS", str(paths_file))
    monkeypatch.setattr(kaggle, "_online", lambda *a, **k: True)
    return inp, code


def _stage(tmp_path, inp, **kw):
    return kaggle.stage_kaggle(
        kw.pop("code_root", None), input_root=inp, working=tmp_path / "working",
        scratch=tmp_path / "scratch", torch_hub=tmp_path / "hub",
        budget_hours=kw.pop("budget_hours", None), activate=False, quiet=True, **kw)


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #
def test_stage_finds_everything_by_content_and_reads_data_in_place(tmp_path, monkeypatch):
    inp, code = _kaggle_tree(tmp_path, monkeypatch)
    info = _stage(tmp_path, inp)

    assert info.dataset == "chaksu"
    assert info.data_dir == inp / "fdp-chaksu" / "chaksu"      # not the lidc cohort
    assert info.n_samples == 1
    # the code is a writable copy with this machine's paths.yaml, and no data
    assert info.repo == tmp_path / "scratch" / "field_disagreement"
    written = yaml.safe_load((info.repo / "configs" / "paths.yaml").read_text())
    assert Path(written["processed"]) == info.data_dir
    assert Path(written["chaksu_processed"]) == info.data_dir
    assert Path(written["runs"]) == tmp_path / "working" / "runs"
    assert not (info.repo / "data").exists()
    # the earlier version's output is back; the laptop's runs inside the code are not
    runs = tmp_path / "working" / "runs"
    assert json.loads((runs / "study_run" / "results" / "m_H_fld_fold0.result.json")
                      .read_text()) == {"auc": 0.9}
    assert not (runs / "laptop").exists() and info.restored == 1
    # weights are where torchvision looks
    for name in kaggle.weight_files():
        assert (tmp_path / "hub" / "checkpoints" / name).exists()
    assert info.missing_weights == [] and info.deadline is None


def test_restoring_runs_never_overwrites_work_done_in_this_session(tmp_path, monkeypatch):
    inp, _ = _kaggle_tree(tmp_path, monkeypatch)
    mine = tmp_path / "working" / "runs" / "study_run" / "results" / "m_H_fld_fold0.result.json"
    mine.parent.mkdir(parents=True)
    mine.write_text('{"auc": 0.95}')
    info = _stage(tmp_path, inp)
    assert json.loads(mine.read_text()) == {"auc": 0.95} and info.restored == 0


def test_a_zipped_cohort_is_unpacked_to_scratch(tmp_path, monkeypatch):
    inp, _ = _kaggle_tree(tmp_path, monkeypatch)
    import shutil
    shutil.rmtree(inp / "fdp-chaksu")
    src = _samples(tmp_path / "build" / "chaksu")
    (inp / "zipped").mkdir()
    with zipfile.ZipFile(inp / "zipped" / "chaksu.zip", "w") as z:
        z.write(src / "index.csv", "chaksu/index.csv")
        z.write(src / "a.npz", "chaksu/a.npz")
    info = _stage(tmp_path, inp)
    assert info.data_dir == tmp_path / "scratch" / "data" / "chaksu" / "chaksu"


def test_an_unnamed_cohort_is_used_only_when_it_is_the_only_one(tmp_path, monkeypatch):
    inp, _ = _kaggle_tree(tmp_path, monkeypatch)
    import shutil
    shutil.rmtree(inp / "fdp-chaksu")
    _samples(inp / "upload-2")
    # two sample folders (lidc-idri, upload-2), neither named for the cohort:
    # refuse to guess
    with pytest.raises(FileNotFoundError, match="'chaksu' cohort"):
        _stage(tmp_path, inp)
    shutil.rmtree(inp / "fdp-lidc")
    shutil.rmtree(inp / "upload-2")
    _samples(inp / "upload-3")
    assert _stage(tmp_path, inp).data_dir == inp / "upload-3"


def test_missing_code_names_the_fix(tmp_path):
    (tmp_path / "input" / "something").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Add Input"):
        kaggle.find_code(tmp_path / "input")


def test_missing_weights_are_reported_not_hidden(tmp_path, monkeypatch):
    inp, _ = _kaggle_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(kaggle, "_online", lambda *a, **k: False)
    missing = kaggle.install_torch_weights(tmp_path / "empty", hub_dir=tmp_path / "hub2",
                                           quiet=True)
    assert missing == kaggle.weight_files()


def test_the_staged_weights_are_the_ones_the_backbones_build_with():
    src = (ROOT / "field_disagreement" / "backbones" / "builtin.py").read_text(encoding="utf-8")
    assert "ResNet50_Weights.IMAGENET1K_V2" in src
    assert kaggle.weight_files() == ["resnet50-11ad3fa6.pth"]


def test_bundle_builds_the_three_uploads(tmp_path, monkeypatch):
    inp, code = _kaggle_tree(tmp_path, monkeypatch)
    laptop = _samples(tmp_path / "laptop" / "chaksu")
    (code / "kaggle_upload").mkdir()                  # an old bundle must not nest
    hub = tmp_path / "hub"
    (hub / "checkpoints").mkdir(parents=True)
    for name in kaggle.weight_files():
        (hub / "checkpoints" / name).write_bytes(b"w")
    import torch
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub))

    out = kaggle.bundle(tmp_path / "out", repo=code, quiet=True)
    # exactly two uploads: small code, large data (cohort + weights)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["fdp-code", "fdp-data"]
    assert out.archive.parent == out.data and out.weights.parent == out.data
    assert (out.code / "pyproject.toml").exists()
    assert not (out.code / "data").exists() and not (out.code / "kaggle_upload").exists()
    with zipfile.ZipFile(out.archive) as z:
        assert sorted(z.namelist()) == ["chaksu/a.npz", "chaksu/index.csv"]
    assert sorted(p.name for p in out.weights.iterdir()) == kaggle.weight_files()
    assert laptop.exists()


# --------------------------------------------------------------------------- #
# the session budget
# --------------------------------------------------------------------------- #
def _store(tmp_path):
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(result_dir=tmp_path / "results")


def test_no_deadline_means_no_check_and_no_file(tmp_path):
    store = _store(tmp_path)
    session.check_budget(store, "a_fold0")
    assert not (store.result_dir / session.TIMES_FILE).exists()


def test_the_deadline_is_set_once_per_session(monkeypatch):
    first = session.start_budget(11, now=1000.0)
    assert first == 1000.0 + 11 * 3600
    assert session.start_budget(20, now=5000.0) == first     # re-running setup buys nothing


def test_a_unit_starts_only_if_the_longest_unit_fits(tmp_path):
    store = _store(tmp_path)
    session.start_budget(1.0, now=0.0)                        # deadline = 3600 s
    session.check_budget(store, "first", now=100.0)           # nothing timed yet: go
    session.record_unit_time(store, "a_fold0", 1200.0)
    session.record_unit_time(store, "b_fold0", 1800.0)
    assert session.load_unit_times(store) == {"a_fold0": 1200.0, "b_fold0": 1800.0}
    session.check_budget(store, "c_fold0", now=1700.0)        # 1900 s left >= 1800
    with pytest.raises(session.SessionBudgetReached, match="c_fold0 not started"):
        session.check_budget(store, "c_fold0", now=1900.0)    # 1700 s left < 1800
    with pytest.raises(session.SessionBudgetReached):
        session.check_budget(_store(tmp_path / "fresh"), "x", now=3601.0)


def test_the_runner_refuses_to_start_a_unit_past_the_budget(tmp_path):
    from backend.notebooks.shared import FoldRunner
    runner = FoldRunner.__new__(FoldRunner)
    runner.schema = None
    runner.store = SimpleNamespace(cached_result=lambda tag, schema=None: None,
                                   result_dir=_store(tmp_path).result_dir)
    session.start_budget(1.0, now=0.0)                        # long past
    built = []
    with pytest.raises(session.SessionBudgetReached):
        runner.run_fold(0, "a_fold0", build=lambda ctx: built.append(1),
                        train_step=None, evaluate=None)
    assert built == []
    # a CACHED unit still loads: the budget only guards new training
    runner.store.cached_result = lambda tag, schema=None: {"auc": 1.0}
    assert runner.run_fold(0, "a_fold0", build=None, train_step=None,
                           evaluate=None) == {"auc": 1.0}


def test_platform_detection(monkeypatch):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert session.platform() == "kaggle"


# --------------------------------------------------------------------------- #
# the clean stop, in a real IPython shell
# --------------------------------------------------------------------------- #
def test_a_budget_stop_ends_the_notebook_cleanly_and_nothing_else_changes():
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    before = list(shell.ast_transformers)
    try:
        assert session.install_budget_stop(shell)
        assert session.install_budget_stop(shell)              # idempotent
        assert len(shell.ast_transformers) == len(before) + 1

        # ordinary cells behave as before: definitions, display, real errors
        assert shell.run_cell("def f():\n    return 7\nx = f()\nx + 1").result == 8
        assert shell.user_ns["x"] == 7
        err = shell.run_cell("1 / 0")
        assert isinstance(err.error_in_exec, ZeroDivisionError)
        assert shell.run_cell("x * 2").result == 14

        # the stop: the cell succeeds, and every later cell is skipped
        stop = shell.run_cell(
            "class SessionBudgetReached(RuntimeError):\n    pass\n"
            "y = 5\nraise SessionBudgetReached('out of time')")
        assert stop.success and shell.user_ns["y"] == 5
        later = shell.run_cell("z = 3\nz")
        assert later.success and later.result is None and "z" not in shell.user_ns
        assert shell.run_cell("x").result is None
    finally:
        shell.ast_transformers[:] = before
        shell.user_ns.pop("__field_disagreement_session_guard__", None)
        for name in ("f", "x", "y", "SessionBudgetReached"):
            shell.user_ns.pop(name, None)


# --------------------------------------------------------------------------- #
# the notebook
# --------------------------------------------------------------------------- #
def _setup_cell(name):
    nb = json.loads((ROOT / "backend" / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
    return next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_notebook_opens_with_the_three_platform_setup(name):
    src = _setup_cell(name)
    ast.parse(src)
    # a study that changes the cohort passes its config, so the stager copies it
    assert "stage_kaggle(CODE" in src and "stage(DRIVE_ROOT" in src
    assert "local run - nothing to stage" in src
    assert 'drive.mount' in src.split("elif")[1]               # Drive only on Colab


def test_the_setup_cell_is_a_no_op_locally(monkeypatch, capsys):
    if Path("/kaggle/input").is_dir():                         # pragma: no cover
        pytest.skip("running on Kaggle")
    for var in ("KAGGLE_KERNEL_RUN_TYPE", "COLAB_RELEASE_TAG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "modules", {k: v for k, v in sys.modules.items()
                                         if not k.startswith("google.colab")})
    ns = {}
    exec(_setup_cell("train_evaluate"), ns)
    assert ns["ENV"] is None
    assert "local run" in capsys.readouterr().out


def test_bundle_cohort_switches_only_the_bundled_base_yaml(tmp_path):
    import shutil
    from backend import config
    from backend.notebooks.shared import kaggle as K
    code = tmp_path / "fdp-code"
    shutil.copytree(config.ROOT / "configs", code / "configs")
    before = (config.ROOT / "configs" / "notebooks" / "base.yaml").read_text(encoding="utf-8")
    K.set_bundle_cohort(code, "chaksu")
    text = (code / "configs" / "notebooks" / "base.yaml").read_text(encoding="utf-8")
    assert "\nextends: cohorts/chaksu" in text and "extends: cohorts/lidc" not in text
    assert (config.ROOT / "configs" / "notebooks" / "base.yaml").read_text(encoding="utf-8") == before
    from backend.notebooks.shared import colab
    dataset, key, stem = colab._resolve_cohort(code, "notebooks/study")
    assert dataset == "chaksu_report"
    import pytest
    with pytest.raises(FileNotFoundError):
        K.set_bundle_cohort(code, "nope")


def test_the_bundle_sets_the_seed_line_only_in_the_copy(tmp_path):
    base = tmp_path / "configs" / "notebooks" / "base.yaml"
    base.parent.mkdir(parents=True)
    base.write_text("extends: cohorts/lidc\nseed: 42\nsplit_seed: 42\n")
    kaggle.set_bundle_seed(tmp_path, [42, 43, 44])
    assert base.read_text() == "extends: cohorts/lidc\nseed: [42, 43, 44]\nsplit_seed: 42\n"
    kaggle.set_bundle_seed(tmp_path, 43)
    assert "seed: 43\nsplit_seed: 42" in base.read_text()
