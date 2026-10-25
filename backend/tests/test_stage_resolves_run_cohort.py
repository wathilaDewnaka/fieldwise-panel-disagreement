"""Staging must copy the cohort THIS RUN trains on, not base.yaml's default.

THE FAILURE THIS PINS. `_resolve_cohort` once read `dataset:` from
configs/notebooks/base.yaml only, so a run whose config changed the cohort had
the default one staged, and died later in `DS.bind` with

    KeyError: configs/paths.yaml has no `chaksu_processed` entry

which blames the config file rather than the staging that never happened. The study
trains on base.yaml's cohort's REPORT-LEVEL condition, so staging asks the same
`run_dataset` that `bind` does.
"""
import os
import pytest

from backend.notebooks.shared import colab


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.get(k)
             for k in ("NB_CONFIG",)}
    for k in saved:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _base():
    """base.yaml's cohort (the one switch - lidc or chaksu) and its report-level
    condition, so these tests hold whichever cohort is selected."""
    from backend.config import load
    from backend.domains.datasets import DATASETS
    base = load("notebooks/base").dataset
    report = next(n for n, s in DATASETS.items() if s.condition_of == base and s.report_level)
    return base, report


def test_without_a_config_it_uses_the_base_default():
    base, _ = _base()
    dataset, key, stem = colab._resolve_cohort(REPO)
    assert dataset == base


def test_the_study_config_stages_the_report_level_condition():
    _, report = _base()
    dataset, key, stem = colab._resolve_cohort(REPO, "notebooks/study")
    assert dataset == report, "staging must follow the run's config"
    assert key == f"{report}_processed"
    assert stem == report, "the zip/folder staged is named after the cohort"


def test_an_explicit_argument_beats_the_environment():
    os.environ["NB_CONFIG"] = "notebooks/study"
    dataset, _, _ = colab._resolve_cohort(REPO, "notebooks/base")
    assert dataset == _base()[0]


def test_a_terminal_run_names_its_config_in_the_environment():
    os.environ["NB_CONFIG"] = "notebooks/study"
    dataset, _, _ = colab._resolve_cohort(REPO)
    assert dataset == _base()[1]


def test_paths_yaml_would_carry_the_cohort_key(tmp_path):
    """`_write_paths` must emit the SELECTED cohort's key, which is the entry
    `DS.bind` looks up and the one that was missing."""
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    colab._write_paths(repo, "/drive/field-disagreement-prediction", "chaksu_processed",
                       "/content/data/chaksu", "/drive/field-disagreement-prediction/data/runs")
    text = (repo / "configs" / "paths.yaml").read_text()
    assert "chaksu_processed: /content/data/chaksu" in text
    assert "processed: /content/data/chaksu" in text
