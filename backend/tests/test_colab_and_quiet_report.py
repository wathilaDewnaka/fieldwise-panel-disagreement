"""Colab staging copies only what training needs, Colab is detected without
importing google.colab, and the report can run silently for the notebook."""
from __future__ import annotations

import fnmatch
import json

from backend.notebooks.shared import colab, session
from backend.notebooks.shared import study_results as SRES
from backend.tests.test_study_results import _write_run


def _skipped(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in colab.SKIP)


def test_colab_does_not_copy_bundles_writeups_or_virtualenvs():
    for heavy in ("kaggle_upload", "reports", "paper", ".venv", ".venvs", "data",
                  "field_disagreement.egg-info", "runs_lidc.zip"):
        assert _skipped(heavy), heavy
    for needed in ("field_disagreement", "backend", "configs", "pyproject.toml"):
        assert not _skipped(needed), needed


def test_a_dropped_drive_read_is_retried(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.txt", tmp_path / "b.txt"
    src.write_text("x")
    calls = {"n": 0}
    real = colab.shutil.copy2

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(5, "Input/output error")
        return real(a, b)
    monkeypatch.setattr(colab.shutil, "copy2", flaky)
    colab._copy_with_retry(src, dst, wait=0)
    assert dst.read_text() == "x" and calls["n"] == 2


def test_colab_is_detected_from_its_environment(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.setenv("COLAB_GPU", "1")
    if not session.Path("/kaggle/input").is_dir():
        assert session.platform() == "colab"


def test_the_notebook_platform_is_set_by_hand():
    """The setup cell names the platform with flags, and the runtime release is
    switched by its own flag; nothing in the notebook guesses the platform."""
    nb = json.load(open("backend/notebooks/train_evaluate.ipynb", encoding="utf-8"))
    setup = "".join(nb["cells"][0]["source"])
    last = "".join(nb["cells"][-1]["source"])
    for flag in ("RUN_ON_KAGGLE = False", "RUN_ON_COLAB = False", "RELEASE_RUNTIME_AT_END"):
        assert flag in setup, flag
    assert 'os.environ["FDP_PLATFORM"]' in setup
    assert "COLAB_GPU" not in setup and "KAGGLE_KERNEL_RUN_TYPE" not in setup
    assert "if RELEASE_RUNTIME_AT_END:" in last and "release_runtime()" in last


def test_the_platform_flag_overrides_detection(monkeypatch):
    monkeypatch.delenv("COLAB_GPU", raising=False)
    for where in ("colab", "kaggle", "local"):
        monkeypatch.setenv("FDP_PLATFORM", where)
        assert session.platform() == where
    monkeypatch.setenv("FDP_PLATFORM", "")
    monkeypatch.setenv("COLAB_GPU", "1")
    if not session.Path("/kaggle/input").is_dir():
        assert session.platform() == "colab"            # unset flag: detection as before


def test_the_report_can_run_silently(tmp_path, capsys):
    run = _write_run(tmp_path, {a: 0.5 for a in SRES.ARMS} | {"H_fld": 1.0},
                     folds=2, per_field=("H_fld",))
    rep = SRES.report(run, d0=1.0, n_boot=50, verbose=False)
    assert rep is not None and "primary" in rep
    assert capsys.readouterr().out == ""


def test_a_replaced_zip_makes_the_unpacked_folder_stale(tmp_path):
    """A runtime that unpacked an old zip must not keep training on it after the
    zip on Drive is replaced."""
    import os
    import zipfile
    from backend.notebooks.shared import colab
    processed, dest = tmp_path / "processed", tmp_path / "data" / "cohort"
    processed.mkdir()
    archive = processed / "cohort.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("cohort/index.csv", "file\n")
    colab._unpack(processed, "cohort", dest, "cohort", quiet=True)
    assert not colab._stale(processed, "cohort", dest)
    with zipfile.ZipFile(archive, "w") as z:                 # a new upload
        z.writestr("cohort/index.csv", "file\nnew.npz\n")
    os.utime(archive, (1, 1))
    assert colab._stale(processed, "cohort", dest)
    (dest / colab.STAGED_STAMP).unlink()                     # unpacked before stamps existed
    assert colab._stale(processed, "cohort", dest)
