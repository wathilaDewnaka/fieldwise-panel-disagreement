"""Configuration loading — the one place that knows where things live.

    from backend.config import paths, load

    P   = paths()                     # P.processed, P.lidc_raw, P.runs (absolute Paths)
    cfg = load("notebooks/study") # cfg.model, cfg.epochs, ... (mutable namespace)
    pp  = load("preprocessing/preprocess_lidc")

`configs/notebooks/study.yaml` is the study's config: it says `extends: base`
and overrides only what the study owns. Everything the arms share is in
`configs/notebooks/base.yaml`, once.

`configs/paths.yaml` is the only config that is machine-specific; everything else
is the protocol and is identical everywhere. On Colab, set
`FDP_PATHS=/content/paths.yaml` to use a different paths file without
editing the one in the repo.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

_PATH_KEYS = ("lidc_raw", "processed", "runs")

# Roots only some runs need. Kept OPTIONAL rather than added to _PATH_KEYS
# because paths.yaml is the one machine-specific config, and a machine that only
# trains on one cohort never needs the others. `paths()` exposes whichever are
# present; the scripts that need one say so themselves.
_OPTIONAL_PATH_KEYS = ("lidc_xml",
                       "chaksu_raw", "chaksu_processed", "chaksu_vcdr_processed",
                       # report-level cohorts (backend.domains.report_condition)
                       "lidc_multi_processed", "lidc_report_processed",
                       "chaksu_report_processed",
                       # LNDb, the external test cohort (backend.domains.lndb)
                       "lndb_raw", "lndb_report_processed")


def _read(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge(name: str, chain: tuple[str, ...] = ()) -> dict:
    """`configs/<name>.yaml` with its `extends:` parent merged in underneath.

    A BARE parent name is a sibling: the notebook configs say `extends: base`
    and get `configs/notebooks/base.yaml`. A parent name CONTAINING A SLASH is
    resolved from `configs/` instead, so `extends: cohorts/lidc` reaches
    `configs/cohorts/lidc.yaml` from anywhere. That second form exists because
    the cohort contracts are not notebook settings - the notebook inherits
    one, and freeze_folds and the reports read them too - so they live in their
    own folder rather than beside the studies.

    The file's own keys win. The merge is one level deep: a mapping value
    (e.g. `served_rule:`) is replaced wholesale rather than merged key by key, so a
    file that touches one of those states the whole block and there is no
    half-inherited version of it to reason about."""
    if name in chain:
        raise ValueError(f"circular extends: {' -> '.join(chain + (name,))}")
    raw = _read(CONFIG_DIR / f"{name}.yaml")

    parent = raw.pop("extends", None)
    if parent is None:
        return raw

    parent = str(parent)
    resolved = (parent if "/" in parent
                else str(PurePosixPath(name).parent / parent))
    cfg = _merge(resolved, chain + (name,))
    cfg.update(raw)
    return cfg


def load(name: str, **overrides) -> SimpleNamespace:
    """Load `configs/<name>.yaml` as a namespace; keyword args override fields.

    The namespace is plain and mutable, so a notebook can still do
    `cfg.epochs = 2` for a smoke run without inventing a config file."""
    cfg = _merge(name)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"{name}.yaml has no field(s) {sorted(unknown)}")
    cfg.update(overrides)
    _normalise_seeds(cfg)
    return SimpleNamespace(**cfg)


def _normalise_seeds(cfg: dict) -> None:
    """`seed` may be one int or a list of ints. Afterwards `seeds` is always the
    list and `seed` its first element, so every consumer that reads `cfg.seed`
    keeps getting an int; a notebook that runs several seeds loops over
    `cfg.seeds` and sets `cfg.seed` itself. `split_seed` (the frozen fold split)
    defaults to the first seed for a config that predates it."""
    if "seed" not in cfg:
        return
    raw = cfg["seed"]
    seeds = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if not seeds:
        raise ValueError("seed: an empty list - give one seed or a list of seeds")
    if any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ValueError(f"seed: every seed must be an integer, got {raw!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seed: the list repeats a seed: {raw!r}")
    cfg["seeds"] = [int(s) for s in seeds]
    cfg["seed"] = cfg["seeds"][0]
    cfg.setdefault("split_seed", cfg["seed"])


def paths() -> SimpleNamespace:
    """Absolute dataset roots from `configs/paths.yaml` (or `$FDP_PATHS`).

    Relative entries resolve against the repo root. Nothing is created and
    nothing has to exist — a script that needs a particular root checks it with
    `require`, so this stays usable for the ones that only need `runs`."""
    env = os.environ.get("FDP_PATHS")
    path = Path(env) if env else CONFIG_DIR / "paths.yaml"
    raw = _read(path)

    missing = [k for k in _PATH_KEYS if k not in raw]
    if missing:
        raise KeyError(f"{path} is missing {missing}")

    keys = _PATH_KEYS + tuple(k for k in _OPTIONAL_PATH_KEYS if k in raw)
    return SimpleNamespace(**{
        k: (Path(raw[k]) if Path(raw[k]).is_absolute() else ROOT / raw[k])
        for k in keys})


def require(root: Path, what: str) -> Path:
    """Fail with an actionable message instead of an empty result set."""
    if not Path(root).is_dir():
        raise FileNotFoundError(
            f"{what} not found at {root}. Fix the entry in configs/paths.yaml.")
    return Path(root)
