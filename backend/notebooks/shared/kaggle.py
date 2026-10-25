"""Run the study notebooks on Kaggle.

Kaggle differs from Colab in four ways that matter here, and this module handles
each one so the notebook cell stays a single call:

    Colab                                  Kaggle
    ------------------------------------   --------------------------------------
    code + data uploaded to Drive          code, cohort and weights uploaded as
                                           private Datasets, mounted READ-ONLY
                                           under /kaggle/input
    runs/ on Drive, survives a disconnect  runs/ in /kaggle/working, saved as the
                                           version's OUTPUT; the next version gets
                                           it back by attaching that output
    internet on                            internet OFF unless the account is
                                           phone-verified: the ImageNet weights
                                           have to be supplied as an input
    session killed at the limit            12 h per session, and a version killed
                                           by the limit loses its output

In the notebook (the setup cell already does this):

    from backend.notebooks.shared.kaggle import stage_kaggle
    stage_kaggle(CODE)       # CODE = the uploaded repo under /kaggle/input

On your own machine, to build the two folders to upload:

    python -m backend.notebooks.shared.kaggle bundle

WHICH COHORT is read from `dataset:` in configs/notebooks/base.yaml, exactly as the
Colab stager reads it, so staging cannot disagree with training. The frozen split's
manifest fingerprint is checked again when the folds load.

Nothing here imports torch at module scope, so it is importable (and tested) off
Kaggle.
"""
from __future__ import annotations

import argparse
import shutil
import socket
import zipfile
from pathlib import Path
from types import SimpleNamespace

from . import colab, session

__all__ = ["stage_kaggle", "find_code", "find_samples", "restore_runs",
           "install_torch_weights", "weight_files", "gpu_report", "bundle",
           "INPUT", "WORKING", "DEFAULT_BUDGET_HOURS"]

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
# Kaggle stops a GPU session at 12 h. One hour is kept back for staging, the
# report cells and writing the output; `check_budget` additionally refuses a unit
# longer than the time left.
DEFAULT_BUDGET_HOURS = 11.0
_SEARCH_DEPTH = 5
# Never part of the uploaded code: history, virtualenvs, data (uploaded as its own
# dataset), the paper's analysis folder and this bundle's own output.
BUNDLE_SKIP = colab.SKIP + ("kaggle_upload", "reports", ".pytest_cache",
                            "*.egg-info")


# --------------------------------------------------------------------------- #
# the session
# --------------------------------------------------------------------------- #
def stage_kaggle(code_root=None, *, input_root=INPUT, working=WORKING, scratch=None,
                 data_dir=None, budget_hours: float | None = DEFAULT_BUDGET_HOURS,
                 torch_hub=None, activate: bool = True, quiet: bool = False,
                 config_name=None):
    """Stage code, cohort, earlier runs and weights for this Kaggle session.

      1. copy the uploaded code to writable scratch (inputs are read-only, and
         configs/paths.yaml has to be rewritten for this machine)
      2. find the cohort `dataset:` names under the inputs - a folder, or its zip
      3. restore every `runs/` folder attached as an input (an earlier version's
         output, or results carried over from Colab) into /kaggle/working/runs,
         never overwriting a file already there
      4. write paths.yaml: data read in place, runs/ in /kaggle/working
      5. put the ImageNet weights where torchvision looks for them
      6. start the session budget and, in a notebook, the clean stop
      7. make the staged copy THE repo for the rest of the session

    Returns (dataset, repo, data_dir, runs, n_samples, restored, missing_weights,
    deadline)."""
    input_root, working = Path(input_root), Path(working)
    src = Path(code_root) if code_root is not None else find_code(input_root)
    colab._check_drive_root(src)
    scratch = Path(scratch) if scratch is not None else _scratch_root()
    repo = scratch / "field_disagreement"

    shutil.rmtree(repo, ignore_errors=True)
    shutil.copytree(src, repo, ignore=shutil.ignore_patterns(*colab.SKIP))

    dataset, processed_key, stem = colab._resolve_cohort(repo, config_name)
    _say(quiet, f"dataset: {dataset}  ->  looking for {stem} under {input_root}")
    if data_dir is None:
        data_dir = find_samples(input_root, stem, scratch=scratch, quiet=quiet)
    data_dir = colab._find_samples(data_dir)

    runs = working / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    restored = restore_runs(input_root, runs, exclude=(src,), quiet=quiet)

    colab._write_paths(repo, src, processed_key, data_dir, str(runs))
    missing = install_torch_weights(input_root, hub_dir=torch_hub, quiet=quiet)

    end = session.start_budget(budget_hours) if budget_hours else None

    if activate:
        colab._activate(repo, src)
        # The session module of the STAGED copy, so the guard lives beside the
        # runner that raises the stop.
        import importlib
        importlib.import_module("backend.notebooks.shared.session").install_budget_stop()

    n = len(list(data_dir.glob("*.npz")))
    _say(quiet, f"code  {src}  ->  {repo}")
    _say(quiet, f"data  {n} samples ({dataset})  <-  {data_dir}  (read in place)")
    _say(quiet, f"runs  {runs}  (saved as this version's output; "
                f"{restored} file(s) restored from inputs)")
    if end is not None:
        _say(quiet, f"time  budget {budget_hours:g} h - no new unit starts that "
                    f"cannot finish inside it")
    _say(quiet, "GPU   " + gpu_report())
    return SimpleNamespace(dataset=dataset, repo=repo, data_dir=data_dir, runs=runs,
                           n_samples=n, restored=restored, missing_weights=missing,
                           deadline=end)


def find_code(input_root=INPUT) -> Path:
    """The uploaded repo: a folder under the inputs holding pyproject.toml and
    backend/notebooks/shared. Found by content, so the dataset's name is free."""
    input_root = Path(input_root)
    for p in _walk(input_root, "pyproject.toml"):
        if (p.parent / "backend" / "notebooks" / "shared").is_dir():
            return p.parent
    raise FileNotFoundError(
        f"no uploaded field-disagreement-prediction code under {input_root} (looked for a folder holding "
        f"pyproject.toml and backend/notebooks/shared). Add your code dataset: "
        f"Add Input > Your Datasets. Inputs present: {_listing(input_root)}")


def find_samples(input_root, stem: str, *, scratch, quiet: bool = False) -> Path:
    """The cohort folder (index.csv + .npz) for `stem` under the inputs.

    Kaggle unpacks an uploaded zip, so this is normally a folder. Preference:
      1. a sample folder named `stem`, or one inside a folder named `stem`
      2. `<stem>.zip`, unpacked to scratch once
      3. the only sample folder present, whatever it is called
    Several unnamed candidates is an error rather than a guess: training one cohort
    on another's samples is the silent failure this ordering exists to prevent."""
    input_root, scratch = Path(input_root), Path(scratch)
    candidates = [p.parent for p in _walk(input_root, "index.csv")
                  if next(p.parent.glob("*.npz"), None) is not None]
    named = [c for c in candidates
             if stem in c.relative_to(input_root).parts]
    if named:
        return named[0]

    archive = next(iter(_walk(input_root, f"{stem}.zip")), None)
    if archive is not None:
        dest = scratch / "data" / stem
        if not dest.exists():
            _say(quiet, f"unpacking {archive} ...")
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        return colab._find_samples(dest)

    if len(candidates) == 1:
        _say(quiet, f"note: using {candidates[0]} - no folder is named '{stem}'; "
                    f"the split's manifest fingerprint is checked when folds load")
        return candidates[0]
    raise FileNotFoundError(
        f"`dataset:` needs the '{stem}' cohort (a folder with index.csv and .npz "
        f"files, or {stem}.zip) under {input_root}. Sample folders found: "
        f"{[str(c) for c in candidates] or 'none'}. Attach the cohort dataset "
        f"(Add Input), or change `dataset:` in configs/notebooks/base.yaml.")


def restore_runs(input_root, runs, *, exclude=(), quiet: bool = False) -> int:
    """Copy every attached `runs/` folder into `runs`, never overwriting.

    An earlier version's output attached as an input appears read-only as
    `<input>/<notebook>/runs/...`; copying it into /kaggle/working is what makes
    cached units load and interrupted units resume. Existing files win, so a
    re-run of the setup cell cannot roll back work done in this session. Folders
    inside `exclude` (the code dataset) are ignored. Returns files copied."""
    input_root, runs = Path(input_root), Path(runs)
    excluded = [Path(e) for e in exclude]
    copied = 0
    for src in _walk(input_root, "runs"):
        if not src.is_dir() or src.resolve() == runs.resolve():
            continue
        if any(_inside(src, e) for e in excluded):
            continue
        for f in sorted(src.rglob("*")):
            if not f.is_file():
                continue
            dest = runs / f.relative_to(src)
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            copied += 1
        _say(quiet, f"restored runs from {src}")
    return copied


def weight_files() -> list[str]:
    """File names torchvision caches the pretrained backbones under.

    The SAME weight enums `field_disagreement/backbones/builtin.py` builds with, so the file
    a model build asks for is the file staged."""
    from torchvision.models import ResNet50_Weights
    return [w.url.rsplit("/", 1)[-1] for w in (ResNet50_Weights.IMAGENET1K_V2,)]


def install_torch_weights(input_root=INPUT, *, hub_dir=None, names=None,
                          quiet: bool = False) -> list[str]:
    """Copy any attached ImageNet weight file into torch's hub cache.

    Returns the names still missing afterwards. Missing weights only matter when
    `pretrained: true` and the session has no internet; that case is named here
    rather than surfacing as a URLError inside the first model build."""
    if hub_dir is None:
        import torch
        hub_dir = torch.hub.get_dir()
    cache = Path(hub_dir) / "checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    names = weight_files() if names is None else list(names)
    missing = []
    for name in names:
        if (cache / name).exists():
            continue
        found = next(iter(_walk(Path(input_root), name)), None)
        if found is not None:
            shutil.copy2(found, cache / name)
            _say(quiet, f"weights  {name}  <-  {found}")
        else:
            missing.append(name)
    if missing and not _online():
        _say(quiet, f"!! weights {missing} are not attached and this session has no "
                    f"internet - a pretrained backbone cannot be built. Attach your "
                    f"fdp-data dataset, or turn Internet on in the notebook "
                    f"settings (needs a phone-verified account).")
    return missing


def gpu_report() -> str:
    """The attached GPU, or what to change in the notebook settings."""
    try:
        import torch
    except ImportError:                                       # pragma: no cover
        return "unknown (torch is not installed)"
    if not torch.cuda.is_available():
        return "NONE - Settings > Accelerator > GPU T4 x2, then restart the session"
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    if archs and f"sm_{major}{minor}" not in archs:
        return (f"{name} - NOT SUPPORTED by this PyTorch build (sm_{major}{minor}); "
                f"switch Settings > Accelerator to GPU T4 x2")
    return name


# --------------------------------------------------------------------------- #
# on your machine: the two folders to upload
# --------------------------------------------------------------------------- #
def bundle(out="kaggle_upload", *, repo=None, include_data: bool = True, config_name=None,
           cohort: str | None = None, seed=None, quiet: bool = False) -> SimpleNamespace:
    """Build the two upload folders from this checkout.

      <out>/fdp-code/                     the repo without data, history or the
                                          paper folder. Small; upload a NEW VERSION of
                                          this dataset whenever code or yaml change.
      <out>/fdp-data/<cohort>.zip         the processed cohort `dataset:` names,
                                          its folder name at the top (Kaggle unpacks
                                          it on upload)
      <out>/fdp-data/torch-weights/*.pth  the two ImageNet weight files, from the
                                          local torch cache (downloaded once if not
                                          cached). fdp-data is large; upload it once.

    `cohort` (e.g. "chaksu") switches base.yaml's `extends: cohorts/<name>` in the
    BUNDLED copy only, so Kaggle can train one cohort while this checkout (and
    Colab) keeps another. The cohort is then read back from that copy, so the zip
    and the training cells cannot disagree. `seed` (an int or a list) does the
    same for base.yaml's `seed:` line.
    """
    from backend import config

    repo = Path(repo) if repo is not None else config.ROOT
    out = Path(out)
    if not out.is_absolute():
        out = repo / out
    out.mkdir(parents=True, exist_ok=True)

    code = out / "fdp-code"
    shutil.rmtree(code, ignore_errors=True)
    shutil.copytree(repo, code, ignore=shutil.ignore_patterns(*BUNDLE_SKIP))
    _say(quiet, f"code     -> {code}")
    if cohort is not None:
        set_bundle_cohort(code, cohort)
        _say(quiet, f"cohort   base.yaml in the bundle now extends cohorts/{cohort} "
                    f"(this checkout unchanged)")
    if seed is not None:
        set_bundle_seed(code, seed)
        _say(quiet, f"seed     base.yaml in the bundle now trains seed: {seed} "
                    f"(this checkout unchanged)")

    dataset, key, stem = colab._resolve_cohort(code if cohort is not None else repo,
                                               config_name)
    data = out / "fdp-data"
    data.mkdir(exist_ok=True)
    archive = None
    if include_data:
        was = config.CONFIG_DIR
        config.CONFIG_DIR = Path(repo) / "configs"
        try:
            processed = Path(getattr(config.paths(), key))
        finally:
            config.CONFIG_DIR = was
        colab._find_samples(processed)
        archive = data / f"{stem}.zip"
        _say(quiet, f"cohort   {dataset}: zipping {processed} -> {archive} ...")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
            for f in sorted(processed.rglob("*")):
                if f.is_file():
                    z.write(f, Path(stem) / f.relative_to(processed))

    import torch
    weights = data / "torch-weights"
    weights.mkdir(exist_ok=True)
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    from torchvision.models import ResNet50_Weights
    for w in (ResNet50_Weights.IMAGENET1K_V2,):
        name = w.url.rsplit("/", 1)[-1]
        if not (cache / name).exists():
            torch.hub.load_state_dict_from_url(w.url, progress=not quiet)
        shutil.copy2(cache / name, weights / name)
    _say(quiet, f"weights  -> {weights}  ({', '.join(weight_files())})")
    _say(quiet, f"\nUpload TWO private Kaggle datasets: {code} as 'fdp-code' and "
                f"{data} as 'fdp-data', and attach both to the notebook (Add Input).")
    return SimpleNamespace(code=code, data=data, archive=archive, weights=weights,
                           dataset=dataset)


def bundle_external(out="kaggle_upload", *, name: str = "lndb_report", repo=None,
                    quiet: bool = False) -> Path:
    """<out>/fdp-lndb/<name>.zip: the processed EXTERNAL test cohort, uploaded as its
    own dataset and attached beside fdp-code, fdp-data and the run being tested
    (`study_external.find` looks for `<name>/index.csv` under /kaggle/input).

    LNDb is CC BY-NC-ND 4.0: keep this dataset PRIVATE, never public."""
    from backend import config

    repo = Path(repo) if repo is not None else config.ROOT
    out = Path(out)
    if not out.is_absolute():
        out = repo / out
    processed = Path(getattr(config.paths(), f"{name}_processed"))
    if not (processed / "index.csv").exists():
        raise FileNotFoundError(f"{processed} has no index.csv: run "
                                f"`python -m backend.domains.lndb.prepare` first")
    dest = out / "fdp-lndb"
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(processed.rglob("*")):
            if f.is_file():
                z.write(f, Path(name) / f.relative_to(processed))
    _say(quiet, f"external -> {archive}\nUpload it as a PRIVATE Kaggle dataset "
                f"'fdp-lndb' (LNDb's licence forbids redistribution).")
    return archive


def set_bundle_cohort(code: Path, cohort: str) -> Path:
    """Point the bundled base.yaml at configs/cohorts/<cohort>.yaml."""
    import re
    code = Path(code)
    if not (code / "configs" / "cohorts" / f"{cohort}.yaml").exists():
        raise FileNotFoundError(f"no configs/cohorts/{cohort}.yaml to point base.yaml at")
    base = code / "configs" / "notebooks" / "base.yaml"
    text = base.read_text(encoding="utf-8")
    new, n = re.subn(r"(?m)^extends:\s*cohorts/\S+", f"extends: cohorts/{cohort}", text)
    if n != 1:
        raise ValueError(f"{base}: expected exactly one `extends: cohorts/...` line, found {n}")
    base.write_text(new, encoding="utf-8")
    return base


def set_bundle_seed(code: Path, seed) -> Path:
    """Set the bundled base.yaml's `seed:` line to one seed or a list."""
    import re
    from backend.config import _normalise_seeds
    _normalise_seeds({"seed": seed})                          # refuse a malformed list
    value = f"[{', '.join(str(s) for s in seed)}]" if isinstance(seed, list) else str(seed)
    base = Path(code) / "configs" / "notebooks" / "base.yaml"
    text = base.read_text(encoding="utf-8")
    new, n = re.subn(r"(?m)^seed:[^\n#]*", f"seed: {value}", text)
    if n != 1:
        raise ValueError(f"{base}: expected exactly one `seed:` line, found {n}")
    base.write_text(new, encoding="utf-8")
    return base


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _walk(root: Path, name: str, depth: int = _SEARCH_DEPTH):
    """Paths called `name` at most `depth` levels below `root`, shallowest first."""
    if not Path(root).is_dir():
        return []
    found = []
    for level in range(1, depth + 1):
        found.extend(sorted(Path(root).glob("/".join(["*"] * level) + "/" + name)))
    return found


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _listing(root: Path):
    return sorted(p.name for p in root.iterdir()) if Path(root).is_dir() else []


def _scratch_root() -> Path:
    """Writable local disk outside /kaggle/working, so staging is never uploaded
    as output."""
    for cand in (Path("/kaggle/temp"), Path("/tmp")):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".field_disagreement_write_probe"
            probe.write_text("ok")
            probe.unlink()
            return cand
        except OSError:
            continue
    raise OSError("no writable scratch directory (/kaggle/temp or /tmp)")


def _online(host: str = "download.pytorch.org", timeout: float = 3.0) -> bool:
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def _say(quiet, msg):
    if not quiet:
        print(msg)


def main(argv=None):                                          # pragma: no cover
    parser = argparse.ArgumentParser(prog="python -m backend.notebooks.shared.kaggle")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle", help="build the folders to upload to Kaggle")
    b.add_argument("--out", default="kaggle_upload")
    b.add_argument("--no-data", action="store_true",
                   help="skip zipping the cohort (code and weights only)")
    b.add_argument("--config", default=None,
                   help="the notebook config whose cohort to zip, e.g. notebooks/study")
    b.add_argument("--cohort", default=None,
                   help="train this cohort on Kaggle (e.g. chaksu); edits the BUNDLED "
                        "base.yaml only, never this checkout")
    b.add_argument("--seed", default=None,
                   help="train these seeds on Kaggle, e.g. 43 or 42,43,44; edits the "
                        "BUNDLED base.yaml only, never this checkout")
    e = sub.add_parser("external", help="zip the processed external test cohort (LNDb)")
    e.add_argument("--out", default="kaggle_upload")
    e.add_argument("--name", default="lndb_report")
    args = parser.parse_args(argv)
    if args.cmd == "external":
        bundle_external(args.out, name=args.name)
    if args.cmd == "bundle":
        seed = None
        if args.seed is not None:
            seed = [int(s) for s in args.seed.split(",")]
            seed = seed[0] if len(seed) == 1 else seed
        bundle(args.out, include_data=not args.no_data, config_name=args.config,
               cohort=args.cohort, seed=seed)


if __name__ == "__main__":                                    # pragma: no cover
    main()
