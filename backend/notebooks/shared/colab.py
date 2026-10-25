"""Stage the repo and its data onto a Colab VM's local disk.

Before it can train on Colab, the notebook has to mount Drive, copy the code to
local disk, work out which cohort this run wants, unpack that cohort's samples,
rewrite the one machine-specific config, and check that a GPU is attached. This
module does all of it, so the notebook's setup cell only says where the folder is
on Drive:

    from google.colab import drive
    drive.mount('/content/drive')

    DRIVE_ROOT = '/content/drive/MyDrive/field-disagreement-prediction'
    import sys; sys.path.insert(0, DRIVE_ROOT)
    from backend.notebooks.shared.colab import stage
    stage(DRIVE_ROOT)

WHICH COHORT is never stated in the notebook. It is read from the configs that
travelled with the code - `dataset:` in configs/notebooks/base.yaml, then that
cohort's `*_processed` folder name from configs/paths.yaml, via the registry in
backend/domains/datasets.py. So the cohort is named in exactly one place, the
same place the training cells read it from, and staging cannot disagree with
training.

configs/paths.yaml is the ONE file that differs between machines and is rewritten
for the VM. Everything else - the preprocessing contract, the frozen folds, the
model and training settings - travels with the folder, so a Colab session and a
local run read byte-identical configuration.

Nothing in here imports torch or google.colab at module scope, so it can be
imported (and tested) off Colab.
"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

# Never copied to the VM: history and virtualenvs are large and useless there,
# `data` is staged separately below (so the big files are not copied twice), and
# `paper` (the analysis scripts and stored predictions) is not part of a training
# run. Copying large folders over the Drive mount makes the setup cell slow and
# can fail with a Drive I/O error.
SKIP = (".git", ".venv", ".venvs", "venv", "data", "kaggle_upload", "reports", "paper",
        ".pytest_cache", ".ipynb_checkpoints", "*.egg-info", "__pycache__", "*.pyc", "*.zip")

REPO_DEFAULT = "/content/field_disagreement"
DATA_DEFAULT = "/content/data"


def release_runtime(delay: float = 5.0) -> None:            # pragma: no cover
    """Hand the GPU back. Run it LAST, once every result is saved.

    Kept here because the notebook imports it from here; it is platform-aware
    (`session.release_runtime`): Colab disconnects, Kaggle and local only say so."""
    from .session import release_runtime as _release

    _release(delay)


def stage(drive_root, repo=REPO_DEFAULT, data_root=DATA_DEFAULT, *,
          activate=True, quiet=False, config_name=None):
    """Copy code + this run's cohort to local disk and point the repo at them.

    Returns a namespace with `dataset`, `repo`, `data_dir`, `runs` and
    `n_samples`. With `activate` (the default), `repo` is also the current
    working directory and the front of `sys.path` on return, so every later
    `import backend...` / `import field_disagreement` resolves to the staged copy whose
    configs/paths.yaml describes THIS VM.

    `activate=False` stages the files and leaves the interpreter alone - which is
    what a test wants, since becoming a different checkout of yourself mid-process
    is not something a test can undo.
    """
    src = _check_drive_root(drive_root)
    repo, data_root = Path(repo), Path(data_root)

    # CODE -> local disk, re-staged FRESH every run. It is a few MB, and the
    # alternative is a reconnect silently reusing what was uploaded three
    # sessions ago.
    shutil.rmtree(repo, ignore_errors=True)
    shutil.copytree(src, repo, ignore=shutil.ignore_patterns(*SKIP),
                    copy_function=_copy_with_retry)

    dataset, processed_key, stem = _resolve_cohort(repo, config_name)
    _say(quiet, f"dataset: {dataset}  ->  staging {stem}"
               + (f"   (from {config_name or _notebook_config_from_env()})"
                  if (config_name or _notebook_config_from_env())
                  else "   (from configs/notebooks/base.yaml)"))

    # DATA -> local disk too: training reads it thousands of times per fold and
    # the Drive mount is far too slow. Unlike the code it is staged ONLY IF
    # ABSENT, because it is large and /content survives a reconnect.
    #
    # The unpack folder is named after the COHORT on purpose: staging is skipped
    # when it exists, so a shared name would let a second cohort train on
    # whichever one the VM unpacked first.
    #
    # "Absent" alone is not enough: a runtime that unpacked an OLD zip would keep
    # training on it after the zip on Drive was replaced. The unpacked
    # folder records which archive it came from, and a different archive on Drive
    # replaces it.
    dest = data_root / stem
    if dest.exists() and _stale(src / "data" / "processed", stem, dest):
        _say(quiet, f"{stem}.zip on Drive changed since it was unpacked - re-unpacking")
        shutil.rmtree(dest)
    if not dest.exists():
        _unpack(src / "data" / "processed", stem, dest, dataset, quiet)
    data_dir = _find_samples(dest)

    runs = f"{src}/data/runs"
    _write_paths(repo, src, processed_key, data_dir, runs)

    if activate:
        _activate(repo, src)

    n = len(list(data_dir.glob("*.npz")))
    _say(quiet, f"code  {src}  ->  {repo}")
    _say(quiet, f"data  {n} samples ({dataset})  ->  {data_dir}")
    _say(quiet, f"runs  {runs}  (survives a disconnect)")
    _say(quiet, "GPU   " + gpu_name())
    return SimpleNamespace(dataset=dataset, repo=repo, data_dir=data_dir,
                           runs=Path(runs), n_samples=n)


def _copy_with_retry(src, dst, *, tries: int = 4, wait: float = 2.0):
    """shutil.copy2, retried: the Drive mount drops a read now and then
    (`OSError: [Errno 5] Input/output error`), and one dropped file used to
    abort the whole setup cell."""
    import time
    for i in range(tries):
        try:
            return shutil.copy2(src, dst)
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def _activate(repo, drive_root):
    """Make the staged copy THE repo for the rest of the session.

    `stage` was imported from the Drive copy - it had to be, the local one did
    not exist yet - so `backend.*` is currently bound to Drive, whose
    configs/paths.yaml is the laptop's and whose files are read over a slow
    mount. Drop those bindings and let the next import re-resolve against the
    copy whose paths.yaml was just written for this VM."""
    for name in [m for m in sys.modules if m.split(".")[0] in ("backend", "field_disagreement")]:
        del sys.modules[name]
    sys.path[:] = [p for p in sys.path if Path(p or ".") != Path(drive_root)]
    sys.path.insert(0, str(repo))
    os.chdir(repo)


def gpu_name() -> str:
    """The attached accelerator, or the fix for there not being one."""
    try:
        import torch
    except ImportError:                                       # pragma: no cover
        return "unknown (torch is not installed)"
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "NONE - fix via Runtime > Change runtime type > GPU"


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def _check_drive_root(drive_root) -> Path:
    """Fail with the folder listing rather than a FileNotFoundError deep in a copy."""
    src = Path(drive_root)
    if (src / "pyproject.toml").exists():
        return src
    here = src.parent
    have = sorted(p.name for p in here.glob("*") if p.is_dir()) if here.is_dir() else []
    raise FileNotFoundError(
        f"{src}/pyproject.toml not found, so DRIVE_ROOT is not the project folder.\n"
        f"Folders in {here}: {have}\n"
        f"Set DRIVE_ROOT to whichever one contains pyproject.toml.")


def _notebook_config_from_env():
    """The notebook config THIS run will actually use, if the run said so.

    A notebook run from a terminal names its config with e.g.
    `NB_CONFIG=notebooks/study`, and that config can change the cohort
    (the study trains on base.yaml's cohort's report-level condition). Staging read base.yaml
    only, so it copied the DEFAULT cohort and wrote a paths.yaml holding only the
    default cohort's key - and the run then died in `DS.bind` with
    "configs/paths.yaml has no `chaksu_processed` entry", pointing at the config
    file rather than at the stager that never staged it."""
    import os
    v = os.environ.get("NB_CONFIG", "").strip()
    return v or None


def _resolve_cohort(repo, config_name=None):
    """(dataset, paths.yaml key, folder stem) for the cohort this run trains on.

    Asked of the CONFIG, not of the caller: `load` resolves the `extends:` chain
    and `resolve` maps the cohort to the paths.yaml key holding its processed
    folder, so the folder name - and therefore the zip name - comes from the same
    two files the training cells read.

    Read out of the STAGED copy explicitly. `stage` is imported off Drive (the
    local copy does not exist yet), so `backend.config` is bound to Drive's
    configs and would answer from there. It is a fresh copy so the answer is the
    same today, but pointing at `repo` means the cohort is read from the exact
    files the training cells will read rather than from whichever copy this
    module happened to be imported from."""
    from backend import config
    from backend.domains.datasets import resolve, run_dataset

    was = config.CONFIG_DIR
    config.CONFIG_DIR = Path(repo) / "configs"
    try:
        name = config_name or _notebook_config_from_env() or "notebooks/base"
        dataset = run_dataset(config.load(name))
        spec = resolve(dataset)
        root = getattr(config.paths(), spec.processed_key, None)
    finally:
        config.CONFIG_DIR = was
    if root is None:
        raise KeyError(
            f"configs/paths.yaml (the copy you uploaded) has no "
            f"`{spec.processed_key}` entry, which `dataset: {dataset}` needs. Add "
            f"it locally, pointing at the folder your preprocessing wrote, and "
            f"re-upload that one file.")
    return dataset, spec.processed_key, Path(root).name


STAGED_STAMP = ".staged_from"     # in an unpacked folder: which zip it came from


def _fingerprint(archive: Path) -> str:
    st = archive.stat()
    return f"{archive.name} {st.st_size} {int(st.st_mtime)}"


def _stale(processed: Path, stem: str, dest: Path) -> bool:
    """True when `dest` was unpacked from a zip other than the one on Drive now.
    A folder with no stamp (unpacked before stamps existed, or copied loose) is
    stale only when a zip is there to replace it."""
    archive = processed / f"{stem}.zip"
    if not archive.exists():
        return False
    stamp = dest / STAGED_STAMP
    return not stamp.exists() or stamp.read_text().strip() != _fingerprint(archive)


def _unpack(processed, stem, dest, dataset, quiet):
    """`<stem>.zip` if it is there, else the loose `<stem>/` folder."""
    archive = processed / f"{stem}.zip"
    if archive.exists():
        _say(quiet, f"unpacking {archive.name} ...")
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
        (dest / STAGED_STAMP).write_text(_fingerprint(archive))
        return
    # Not zipped - copy the folder across instead. Works, but zip it next time:
    # Drive is slow at thousands of small files.
    loose = processed / stem
    if not (loose / "index.csv").exists():
        have = sorted(p.name for p in processed.glob("*")) if processed.is_dir() else []
        raise FileNotFoundError(
            f"`dataset: {dataset}` needs {stem}.zip or {stem}/index.csv under "
            f"{processed}/.\nFound: {have}\nEither upload it, or change "
            f"`dataset:` in configs/notebooks/base.yaml to a cohort you have.")
    _say(quiet, f"copying {loose.name}/ from Drive (zip it next time - much faster) ...")
    shutil.copytree(loose, dest)


def _find_samples(dest) -> Path:
    """Locate the samples by index.csv rather than by folder name, so it does not
    matter what the zip tool wrapped them in."""
    data_dir = next((p.parent for p in sorted(Path(dest).rglob("index.csv"))), None)
    if data_dir is None:
        raise FileNotFoundError(f"no index.csv under {dest} - wrong archive?")
    if not any(data_dir.glob("*.npz")):
        raise FileNotFoundError(
            f"{data_dir} has an index.csv but no .npz files beside it.")
    return data_dir


def _write_paths(repo, src, processed_key, data_dir, runs):
    """The one machine-specific file, rewritten for this VM.

    runs/ points BACK at Drive so checkpoints and results survive a disconnect;
    everything read hot is on local disk. `processed` is written because
    backend.config.paths() requires it, and the cohort's own key is written
    beside it (the same key when the cohort is LIDC). The *_raw keys are only
    read when re-running preprocessing, so they need not exist here."""
    entries = {
        "lidc_raw": f"{src}/data/raw/lidc-idri/lidc-idri",
        "chaksu_raw": f"{src}/data/raw/chaksu",
        "processed": str(data_dir),
        processed_key: str(data_dir),
        "runs": runs,
    }
    Path(repo, "configs", "paths.yaml").write_text(
        "".join(f"{k}: {v}\n" for k, v in entries.items()))


def _say(quiet, msg):
    if not quiet:
        print(msg)
