"""The dataset registry — the three things that differ between cohorts.

Everything downstream of preprocessing is dataset-agnostic, so these are the
only per-domain facts in the codebase, and each of them fails SILENTLY if wrong:
a mismatched K trains a head against a target of another length, and a
disagreement threshold on the wrong scale selects an empty subgroup that reads
as "no effect" rather than as an error.
"""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from field_disagreement import build_model
from backend.config import ROOT, load
from backend.domains.datasets import DATASETS, UnsupportedExperiment, resolve


def test_every_COHORT_declares_a_distinct_frozen_split():
    """One file per cohort: a shared split file would be silently wrong for
    whichever domain wrote it second.

    A spec with `condition_of` set is EXEMPT by design — it is a second build
    on an existing cohort and has to share that cohort's split or the two are not
    paired. `test_a_condition_shares_its_base_cohorts_split` pins the other half."""
    folds = [s.folds for s in DATASETS.values() if s.condition_of is None]
    assert len(set(folds)) == len(folds)


def test_frozen_splits_are_named_uniformly_under_configs_folds():
    """`configs/folds/folds_<cohort>.json`, with no cohort exempt.

    LIDC's used to be `configs/folds.json` — unprefixed, because it was the only
    one. Reading the naming off `spec.name` is what stops the next cohort landing
    somewhere else again. A condition is named after its BASE, for the same
    reason it shares the file."""
    for name, spec in DATASETS.items():
        stem = spec.condition_of or name
        assert spec.folds == f"configs/folds/folds_{stem}.json", name


def test_a_condition_shares_its_base_cohorts_split():
    """The other half of the exemption above: a condition must point at a REAL
    base, that base must not itself be a condition, and the two must share the
    split file — otherwise `lidc_report` and `lidc` are not paired item for item
    and no comparison between them means anything."""
    for name, spec in DATASETS.items():
        if spec.condition_of is None:
            continue
        base = resolve(spec.condition_of)
        assert base.condition_of is None, f"{name} points at another condition"
        assert spec.folds == base.folds, f"{name} does not share {base.name}'s split"
        assert spec.processed_key != base.processed_key,             f"{name} would overwrite {base.name}'s processed folder"


def test_resolve_lists_the_options_instead_of_raising_keyerror():
    with pytest.raises(ValueError, match="known:"):
        resolve("lung")


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_the_paths_config_knows_every_declared_dataset(name):
    """`processed_key` must name a real entry in configs/paths.yaml, or the
    failure only shows up once someone tries to train."""
    import yaml
    raw = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))
    assert DATASETS[name].processed_key in raw


@pytest.mark.parametrize("name,expected_k",
                         [("lidc", 5), ("chaksu", 2), ("lndb", 5)])
def test_opinion_head_width_matches_what_the_domain_supplies(name, expected_k):
    """K is a property of the annotation protocol, not of the model. Chákṣu's
    panel only says normal / glaucoma-suspect, so its opinion head is 2 wide —
    building it at LIDC's 5 would train against a 2-wide target."""
    spec = resolve(name)
    assert spec.n_ratings == expected_k

    model = build_model("cnn", pretrained=False, opinion_head=True,
                        n_ratings=spec.n_ratings)
    out = model(torch.zeros(2, 3, 64, 64), return_all=True)
    assert out["opinion"].shape == (2, expected_k)


def test_the_default_opinion_width_is_unchanged():
    """A caller that never passes n_ratings must build exactly the network it
    built before, so the measured LIDC results stay reproducible."""
    model = build_model("cnn", pretrained=False, opinion_head=True)
    assert model(torch.zeros(1, 3, 64, 64), return_all=True)["opinion"].shape == (1, 5)


def test_an_opinion_head_narrower_than_two_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        build_model("cnn", pretrained=False, opinion_head=True, n_ratings=1)


def test_every_cohort_classifies_and_has_a_varying_panel():
    """Both capabilities the study needs are declared per cohort, and every
    cohort in the registry has both."""
    for name, spec in DATASETS.items():
        assert spec.classification is True, name
        assert spec.opinion_distribution is True, name


# --------------------------------------------------------------------------- #
# the capability guard — the traps above, turned into an error before training
# --------------------------------------------------------------------------- #


def _constant_panel_cohort():
    """A cohort whose panel never disagrees: it classifies, but an opinion head
    trained on it would be trained against a constant."""
    from dataclasses import replace
    return replace(resolve("chaksu"), name="constant_panel",
                   opinion_distribution=False,
                   note="every reader gives the same opinion")


def test_a_cohort_that_cannot_support_the_study_says_so_before_training():
    """The whole point: an error at the config cell, not a null result a GPU day
    later. The cohort can still classify; it is only the opinion head it cannot
    train, and the message has to distinguish the two."""
    spec = _constant_panel_cohort()
    assert spec.require_capabilities("classification") is spec

    with pytest.raises(UnsupportedExperiment, match="opinion_distribution"):
        spec.require_capabilities("classification", "opinion_distribution")


def test_the_error_names_the_cohort_and_repeats_its_note():
    """A reader hitting this has picked the wrong cohort for the study and needs
    to know which one and why, without opening the registry."""
    with pytest.raises(UnsupportedExperiment) as excinfo:
        _constant_panel_cohort().require_capabilities("opinion_distribution")
    message = str(excinfo.value)
    assert "constant_panel" in message
    assert "opinion_distribution" in message
    assert "same opinion" in message                # its own note, carried through


def test_a_misspelt_capability_is_rejected_rather_than_silently_passing():
    """`require_capabilities("opinion")` must not quietly succeed — a guard that
    checks nothing is worse than no guard, because it reads as protection."""
    with pytest.raises(ValueError, match="unknown capability"):
        resolve("lidc").require_capabilities("opinion")


# --------------------------------------------------------------------------- #
# the disagreement-scale trap
# --------------------------------------------------------------------------- #
def test_one_constant_threshold_cannot_serve_two_scales():
    """The heart of it. `contested_d: 1.0` is on LIDC's rating-s.d. scale. A
    bounded overlap (1 - unanimous/union) cannot exceed 1, so the constant
    selects NOTHING there — and an empty contested subgroup reads as "no effect"
    downstream, never as a misconfiguration. Hence one cut-off per cohort."""
    lidc_like = pd.DataFrame({"disagreement": [0.0, 0.5, 0.5, 1.2, 1.8]})     # s.d.
    bounded_like = pd.DataFrame({"disagreement": [0.07, 0.2, 0.4, 0.5, 1.0]})  # 0..1 overlap
    flat = 1.0
    assert (lidc_like.disagreement > flat).sum() == 2      # a real subgroup
    assert (bounded_like.disagreement > flat).sum() == 0   # ...and none at all


def test_bind_points_a_config_at_a_cohort_and_rescales_its_threshold(tmp_path,
                                                                    monkeypatch):
    """One call in the notebook does all of it: paths, split, K, and — the part
    that matters — a `contested_d` on the cohort's OWN scale instead of LIDC's."""
    import types

    from backend.domains import datasets as DS

    monkeypatch.setattr(DS, "paths",
                        lambda: types.SimpleNamespace(chaksu_processed=tmp_path))

    cfg = load("notebooks/base")
    cfg.dataset = "chaksu"
    cfg.contested_d = 1.0                           # LIDC's ruler
    spec = DS.bind(cfg)

    assert spec.name == "chaksu"
    assert cfg.n_ratings == 2
    assert cfg.folds == "configs/folds/folds_chaksu.json"
    assert cfg.processed_dir == tmp_path
    assert cfg.contested_d == pytest.approx(0.45)   # chaksu's own


def test_bind_puts_the_thresholds_on_the_bound_cohorts_scale(tmp_path,
                                                             monkeypatch):
    """The cohort's own measured absolute is what the run uses.

    Chákṣu is why this cannot be left to the config: its three-valued
    disagreement scale defeats every quantile, and LIDC's 1.0 selects the EMPTY
    SET on its 0..0.49 ruler — an empty contested subgroup that reports "no
    effect" rather than an error. On LIDC the two agree, so nothing moves."""
    import types

    from backend.domains import datasets as DS

    pd.DataFrame({"file": ["a.npz"], "disagreement": [0.5]}).to_csv(
        tmp_path / "index.csv", index=False)
    monkeypatch.setattr(DS, "paths",
                        lambda: types.SimpleNamespace(processed=tmp_path))

    # The cohort is pinned HERE rather than inherited from base.yaml: this test is
    # about LIDC's behaviour specifically, and `dataset:` is the switch a run flips
    # to train on another cohort. Reading it from the config made a legitimate
    # switch fail this test with a KeyError on the other cohort's `*_processed`.
    cfg = load("notebooks/base")
    cfg.dataset = "lidc"
    cfg.date_stamp = False                   # the stamp has its own test below
    # This test is about the THRESHOLDS. Decentring is declared per cohort and
    # would add `_dc20` to every folder below, so it is stubbed out here to keep
    # the run_dir assertions about the one thing they check; the suffix has its
    # own test in `test_decentred_runs_get_their_own_results_folder`.
    monkeypatch.setattr(DS, "cohort_decentring", lambda name: (0.0, 0))
    cfg.contested_d = 1.0
    DS.bind(cfg)
    assert cfg.contested_d == 1.0
    assert cfg.run_dir == "scratch"        # LIDC keeps the bare output folder

    # ...and a cohort on another ruler is corrected rather than inherited.
    cfg = load("notebooks/base")
    cfg.dataset = "chaksu"
    cfg.contested_d = 1.0
    monkeypatch.setattr(DS, "paths",
                        lambda: types.SimpleNamespace(chaksu_processed=tmp_path))
    DS.bind(cfg)
    assert cfg.contested_d == 0.45


def test_every_cohort_states_a_threshold_on_its_own_scale():
    """A cohort added without one silently inherits whichever ruler the yaml
    happens to carry, which is the failure this field exists to stop."""
    for name, spec in DATASETS.items():
        assert spec.contested_d is not None, f"{name} has no contested_d"
        # A rating s.d. is bounded well below 2.1, so a threshold at or above
        # it selects nothing at all. A report-level cohort's d is a mean of
        # per-attribute dispersions in cohort-s.d. units, so its declared 80th
        # percentile sits on that wider scale instead.
        bound = 5.0 if spec.report_level else 2.1
        assert 0 < spec.contested_d < bound, name


def test_base_config_selects_a_real_cohort():
    """`dataset:` is the one switch, so it must name a cohort that EXISTS — a typo
    (`glesson`, the way the raw folder is spelled) would otherwise surface as a
    KeyError deep inside a notebook's config cell."""
    cfg = load("notebooks/base")
    assert cfg.dataset in DATASETS


def test_bound_thresholds_are_on_the_selected_cohorts_own_scale():
    """The invariant that actually protects a cohort switch.

    `contested_d` has to be on the bound cohort's ruler —
    LIDC's s.d. of ratings runs to 2.0 and Chákṣu's vote s.d. to 0.49. A threshold left on the previous
    cohort's scale selects the EMPTY SET, and an empty contested subgroup reads
    downstream as "no effect", never as a misconfiguration.

    Asserted on the config a run actually uses, i.e. AFTER `bind` — the yaml
    cannot be right for more than one cohort at a time, so the cohort's own
    measured value is what has to hold up. The property, a real proper subgroup,
    rather than any particular number.

    Skipped when the selected cohort's samples are not on this machine; the
    check is about the config matching the DATA, so it cannot run without it."""
    from backend.domains import datasets as DS

    cfg = load("notebooks/base")
    spec = resolve(cfg.dataset)
    try:
        index = spec.processed_dir() / "index.csv"
    except (FileNotFoundError, KeyError) as exc:
        pytest.skip(f"{cfg.dataset} samples not present: {exc}")
    if not index.exists():
        pytest.skip(f"{cfg.dataset} has no index.csv at {index}")

    DS.bind(cfg)
    d = pd.read_csv(index)["disagreement"]
    frac = float((d > cfg.contested_d).mean())
    assert 0.0 < frac < 1.0, (
        f"contested_d={cfg.contested_d} selects {frac:.1%} of {cfg.dataset}, "
        f"whose disagreement runs {d.min():.3g}..{d.max():.3g}. Fix "
        f"DatasetSpec.contested_d for this cohort.")


def test_a_non_lidc_cohort_gets_its_own_runs_folder(tmp_path, monkeypatch):
    """The per-fold cache tag carries the model and the arm but NOT the dataset,
    so sharing `runs/<run_dir>` between cohorts would let cached_result() answer a
    Chákṣu fold with LIDC's numbers. bind() namespaces the folder; without it the
    switch is silently wrong rather than loudly."""
    import types

    from backend.domains import datasets as DS

    pd.DataFrame({"file": ["a.npz"], "disagreement": [0.5]}).to_csv(
        tmp_path / "index.csv", index=False)
    monkeypatch.setattr(DS, "paths",
                        lambda: types.SimpleNamespace(chaksu_processed=tmp_path))
    monkeypatch.setattr(DS, "cohort_decentring", lambda name: (0.0, 0))
    monkeypatch.setattr(DS, "cohort_data_version", lambda name: None)

    cfg = load("notebooks/base")
    cfg.dataset = "chaksu"
    cfg.date_stamp = False                   # the stamp has its own test below
    DS.bind(cfg)
    assert cfg.run_dir == "scratch_chaksu"

    DS.bind(cfg)                              # re-running the config cell is idempotent
    assert cfg.run_dir == "scratch_chaksu"


def test_date_stamped_run_dirs_adopt_an_unfinished_run_rather_than_orphan_it(
        tmp_path, monkeypatch):
    """`date_stamp` was being done by hand, and forgetting it is silent: two
    protocols land in one folder and cached_result() answers the second run with
    the first one's numbers.

    The reason it is not just a stamp: on Colab a fold can train overnight and a
    dead session is restarted the next morning. A bare stamp would come back to a
    fresh empty folder and throw away every checkpoint, on a project that is
    already compute-limited — so an existing NON-EMPTY dated run of the same
    study is continued instead."""
    import types
    from datetime import date, timedelta

    from backend.domains import datasets as DS

    runs = tmp_path / "runs"
    runs.mkdir()
    pd.DataFrame({"file": ["a.npz"], "disagreement": [0.5]}).to_csv(
        tmp_path / "index.csv", index=False)
    monkeypatch.setattr(DS, "paths",
                        lambda: types.SimpleNamespace(processed=tmp_path,
                                                      chaksu_processed=tmp_path,
                                                      runs=runs))
    # centred: this test is about the DATE stamp and the adoption rule, and the
    # cohort's `_dc20` suffix would only add noise to every assertion below.
    monkeypatch.setattr(DS, "cohort_decentring", lambda name: (0.0, 0))
    monkeypatch.setattr(DS, "cohort_data_version", lambda name: None)
    today = date.today().strftime("%d-%m-%y")

    def bound(dataset="lidc", **kw):
        cfg = load("notebooks/base")
        cfg.dataset, cfg.date_stamp, cfg.date_fmt = dataset, True, "%d-%m-%y"
        cfg.__dict__.update(kw)
        DS.bind(cfg)
        return cfg

    # nothing on disk: today's folder
    cfg = bound()
    assert cfg.run_dir == f"scratch-{today}"
    DS.bind(cfg)                              # re-running the config cell
    assert cfg.run_dir == f"scratch-{today}"

    # an unfinished run from two days ago is CONTINUED, not orphaned
    old = runs / f"scratch-{(date.today() - timedelta(days=2)):%d-%m-%y}"
    old.mkdir()
    (old / "resnet50_fold0.result.json").write_text("{}")
    assert bound().run_dir == old.name

    # an EMPTY one is not: nothing would be lost by starting fresh
    for f in old.iterdir():
        f.unlink()
    assert bound().run_dir == f"scratch-{today}"

    # the cohort suffix lands on the STEM, so a second bind cannot stack either
    cfg = bound("chaksu")
    assert cfg.run_dir == f"scratch_chaksu-{today}"
    DS.bind(cfg)
    assert cfg.run_dir == f"scratch_chaksu-{today}"


def test_bind_applies_the_cohorts_own_calibrated_decentring():
    """The cohort's declared decentring must reach the loader, and PER COHORT.

    ONE CONTROL, in the cohort's own file. `configs/cohorts/<name>.yaml` declares
    `decentring:` - the switch and the value, 0.0 being off - and
    `datasets.cohort_decentring` is the only reader.
    """
    import types

    from backend.domains import datasets as DS

    def bound(name, **kw):
        cfg = types.SimpleNamespace(dataset=name, run_dir="x", **kw)
        DS.bind(cfg, require_data=False)
        return cfg

    # the cohort's OWN declared value reaches the loader, per cohort
    for name in ("lidc", "chaksu", "lndb"):
        assert bound(name).shift_frac == 0.20, name

    # and it is read from the BOUND cohort's file, not inherited from whatever
    # the config was carrying
    cfg = types.SimpleNamespace(dataset="lidc", run_dir="x", shift_frac=0.5)
    DS.bind(cfg, require_data=False)
    assert cfg.shift_frac == 0.20, cfg.shift_frac

    # a cohort with no file, or a file that forgets the key, is an error rather
    # than a silent 0.0 - which would read as "centred" and not as "undeclared"
    with pytest.raises(KeyError, match="decentring"):
        DS.cohort_decentring("no_such_cohort")


def test_only_the_cohort_file_decides_the_decentring_condition():
    """One control, and it is the cohort's.

    No study config may state the condition: a study that does is a second
    opinion about a number the cohort already owns, and it will be the stale one.
    """
    from pathlib import Path

    import yaml as _yaml

    for path in sorted(Path("configs/notebooks").glob("*.yaml")):
        cfg = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key in ("decentring", "decentring_auto", "shift_frac"):
            assert key not in cfg, (
                f"{path.name} sets `{key}`. The decentring condition belongs to "
                f"the COHORT and is declared once in configs/cohorts/<name>.yaml "
                f"as `decentring:`; a second copy in a study's yaml is how the "
                f"studies came to disagree about the condition they measured in.")

    # ...and every cohort declares it, so the condition is never undeclared
    for path in sorted(Path("configs/cohorts").glob("*.yaml")):
        cfg = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert "decentring" in cfg, f"{path.name} does not declare `decentring:`"


def test_decentred_runs_get_their_own_results_folder(monkeypatch):
    """A decentred run must not land on top of a centred one.

    Decentring is a different EXPERIMENT, not a different setting - a fold
    trained at 0.20 is not comparable with one trained centred. `run_dir` is
    date-stamped, so without a condition suffix both land in the same folder on
    the same day; and since the freshness check re-runs a fold whose
    `shift_frac` differs, the new condition would OVERWRITE the banked one.
    """
    import types

    from backend.domains import datasets as DS

    # this test is about the decentring suffix; the data-build suffix is its own test
    monkeypatch.setattr(DS, "cohort_data_version", lambda n: None)

    def run_dir(name, decentring=None):
        if decentring is not None:
            monkeypatch.setattr(DS, "cohort_decentring",
                                lambda n, _v=decentring: (_v, 0))
        cfg = types.SimpleNamespace(dataset=name, run_dir="study",
                                    date_stamp=False)
        DS.bind(cfg, require_data=False)
        return cfg.run_dir

    centred = run_dir("chaksu", decentring=0.00)
    decentred = run_dir("chaksu", decentring=0.20)
    assert centred == "study_chaksu"
    assert decentred == "study_chaksu_dc20"
    assert centred != decentred, "the two conditions share a results folder"

    monkeypatch.undo()
    # the cohorts' real declared values
    assert run_dir("lidc") == "study_dc20"

    # idempotent: re-running the config cell in a live kernel must not stack
    cfg = types.SimpleNamespace(dataset="chaksu", run_dir="study",
                                date_stamp=False)
    DS.bind(cfg, require_data=False)
    DS.bind(cfg, require_data=False)
    assert cfg.run_dir == "study_chaksu_dc20_autocrop", cfg.run_dir


def test_a_rebuilt_cohort_gets_its_own_results_folder():
    """Chákṣu was re-cropped (annotation-free window); its cohort file declares
    `data_version: autocrop`, so no result trained on the old crops can be
    resumed into, or cached as, a new-crop run. LIDC keeps its folder name."""
    import types

    from backend.domains import datasets as DS
    names = {}
    for name in ("chaksu", "lidc"):
        cfg = types.SimpleNamespace(dataset=name, run_dir="study", date_stamp=False,
                                    report_level=True)
        DS.bind(cfg, require_data=False)
        names[name] = cfg.run_dir
    assert names["chaksu"].endswith("_autocrop")
    assert names["lidc"] == "study_lidc_report_dc20"
