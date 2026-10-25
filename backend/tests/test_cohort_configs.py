"""The cohort is selected by naming a frozen file, and a pinned study refuses
any other cohort.

Each cohort's thresholds live in `configs/cohorts/<name>.yaml` - their own
folder, not beside the study config, because `freeze_folds` and the reports read
them too. base.yaml extends one of them, and a study can pin the cohort it is a
result on.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from backend.config import CONFIG_DIR, load
from backend.domains import datasets as DS
from backend.domains.datasets import DATASETS

COHORT_FILES = sorted(CONFIG_DIR.glob("cohorts/*.yaml"))
STUDIES = ("study",)

# Keys that are on a cohort's own measurement scale and must therefore live in
# the cohort file, never in the shared protocol.
COHORT_OWNED = ("dataset", "folds", "contested_d")
# Everything a cohort file may say. The quantile switches (`contested_q`,
# `defer_d_q`) and the second copy of the cut-off (`defer_d_thresh`) were
# removed: nothing read them, or they only mirrored `contested_d`.
COHORT_KEYS = {*COHORT_OWNED, "neutral_rating", "decentring", "shift_seed", "data_version"}


def _top(name: str) -> dict:
    return yaml.safe_load(
        (CONFIG_DIR / "notebooks" / f"{name}.yaml").read_text(encoding="utf-8"))


def test_there_is_one_cohort_file_per_registered_cohort():
    named = {p.stem for p in COHORT_FILES}
    # a condition of a cohort (lidc_report) shares its base's file and decentring
    cohorts = {n for n, s in DATASETS.items() if s.condition_of is None}
    assert named == cohorts, (
        f"cohort files {sorted(named)} vs registry {sorted(DATASETS)}")


def test_base_states_no_cohort_of_its_own():
    """The protocol must not be able to answer "which cohort" by itself, or the
    cohort file stops being the single place that decides."""
    base = _top("base")
    leaked = [k for k in COHORT_OWNED if k in base]
    assert not leaked, (
        f"{leaked} are back in base.yaml; they are on one cohort's scale and "
        f"belong in configs/cohorts/<name>.yaml")
    assert base.get("extends", "").startswith("cohorts/"), base.get("extends")


@pytest.mark.parametrize("path", COHORT_FILES, ids=[p.stem for p in COHORT_FILES])
def test_each_cohort_file_is_self_consistent(path: Path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = path.stem
    spec = DATASETS[name]

    assert cfg["dataset"] == name, "the file names a different cohort than itself"
    assert cfg["folds"] == spec.folds, (
        f"{path.name} names {cfg['folds']}, the registry says {spec.folds}")
    assert set(cfg) <= COHORT_KEYS, f"{path.name}: unknown keys {set(cfg) - COHORT_KEYS}"
    assert cfg["contested_d"] == pytest.approx(spec.contested_d), (
        f"{path.name} and DatasetSpec disagree about the contested threshold")


def test_switching_the_cohort_file_switches_every_study():
    """The point of the arrangement: one named file, and every study follows."""
    base = load("notebooks/base")
    active = _top("base")["extends"].split("/", 1)[1]
    cohort = yaml.safe_load(
        (CONFIG_DIR / "cohorts" / f"{active}.yaml").read_text(encoding="utf-8"))
    assert base.dataset == active
    assert base.contested_d == cohort["contested_d"]
    for study in STUDIES:
        assert load(f"notebooks/{study}").dataset == active


def test_a_pinned_study_refuses_the_wrong_cohort():
    """A study pinned to a cohort must not run on another one because one line
    of base.yaml moved."""
    right = load("notebooks/study")
    right.require_dataset, right.dataset = "lidc", "lidc"
    assert DS.bind(right, require_data=False).name == "lidc_report"    # the right cohort binds, at report level

    wrong = load("notebooks/study")
    wrong.require_dataset, wrong.dataset = "lidc", "chaksu"
    with pytest.raises(ValueError, match="pinned to"):
        DS.bind(wrong, require_data=False)


def test_an_unpinned_study_is_unaffected():
    """The study runs on two cohorts, so it is not pinned; the guard must be
    inert when `require_dataset` is absent."""
    cfg = load("notebooks/study")
    assert getattr(cfg, "require_dataset", None) is None
    cfg.dataset = "chaksu"
    spec = DS.bind(cfg, require_data=False)
    assert spec.name == "chaksu_report"
    # and the cohort still cannot collide with LIDC's cached folds
    assert "_chaksu" in cfg.run_dir, cfg.run_dir


def test_extends_resolves_both_a_sibling_and_a_root_relative_path():
    """`extends: base` is a sibling; `extends: cohorts/lidc` is resolved from
    `configs/`. The second form is what lets the cohort contracts live outside
    `configs/notebooks/` without every study file spelling a `../` path."""
    assert _top("base")["extends"].startswith("cohorts/")
    assert _top("study")["extends"] == "base"
    # both forms land in one namespace
    cfg = load("notebooks/study")
    assert cfg.dataset == _top("base")["extends"].split("/", 1)[1]   # the active cohort file
    assert cfg.lam_def == 0.3                     # from configs/notebooks/base.yaml
    assert cfg.report_level is True               # the study's own


def test_the_cohort_contracts_are_not_notebook_configs():
    """They sit in their own folder: `freeze_folds`, the dataset report and the
    tests read them too, so filing them beside the study config said they were
    a study's setting when they are the data protocol."""
    assert not list(CONFIG_DIR.glob("notebooks/cohort_*.yaml")), (
        "a cohort contract is back in configs/notebooks/")
    assert (CONFIG_DIR / "cohorts").is_dir()
