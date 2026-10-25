"""Which dataset a run trains on — the one place that knows the difference.

Everything downstream of preprocessing is dataset-agnostic: the loader
(`backend/domains/adapter.py`), the split machinery (`field_disagreement.splits`) and the
models all work off the same .npz schema. But three things DO differ per cohort,
and getting any of them wrong is silent rather than loud:

  * where the processed samples live, and which frozen split belongs to them
  * K, the width of the opinion distribution — 5 for LIDC's malignancy scale,
    2 for Chákṣu's normal / glaucoma-suspect vote. A model built with the wrong K
    trains a head against a target of another length.
  * the SCALE of `disagreement`. The `contested_d` cut-off that selects ~14% of
    LIDC selects the EMPTY SET on Chákṣu's 0..0.49 scale, with no error raised.
    So each cohort declares its own cut-off (`DatasetSpec.contested_d`), on its
    own scale.

A notebook picks a cohort with one line:

    from backend.domains.datasets import resolve
    spec = resolve(cfg.dataset)          # "lidc" | "chaksu" | "lndb" | ...
    cfg.processed_dir = spec.processed_dir(require_exists=require_data)
    cfg.folds = spec.folds

The frozen splits live together in `configs/folds/`, one `folds_<cohort>.json`
per cohort.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from backend.config import load, paths, require
from backend.domains.report_condition import CHAKSU_ATTRS, LIDC_ATTRS


class UnsupportedExperiment(RuntimeError):
    """This cohort cannot answer the question this study asks.

    Raised BEFORE training rather than discovered in a null result afterwards."""


@dataclass(frozen=True)
class DatasetSpec:
    """Everything a training run needs to know about one cohort."""

    name: str
    processed_key: str      # the configs/paths.yaml entry holding its .npz files
    folds: str              # repo-relative path of its frozen split
    n_ratings: int          # K: the width of opinion_dist this domain supplies
    subject: str            # what one `patient_id` is, for the fold summary
    disagreement: str       # how `disagreement` is measured, and its range
    classification: bool    # whether `label` varies enough to train a classifier
    # Whether `opinion_dist` actually VARIES across items. K >= 2 is not enough:
    # a cohort whose panel never disagrees gives an opinion head a constant
    # target, and training on it produces a clean null indistinguishable from a
    # real one, which is why this is a flag and not a sentence in `note`.
    opinion_distribution: bool = True
    # Decentring (the translation of the inputs) is not here: it lives in
    # `configs/cohorts/<name>.yaml` as `decentring:`, and `bind()` reads it from
    # there (`cohort_decentring`).

    # Set when this spec is a second BUILD/CONDITION of another cohort rather than
    # a cohort of its own (e.g. the same cohort rebuilt at another resolution).
    # It reuses the base's
    # frozen split, so the registry invariant that every cohort owns a distinct
    # folds file must exempt it - and a reader must never count it as a fifth
    # dataset. Report it BESIDE its base, never instead of it.
    condition_of: str | None = None
    # `disagreement` is the panel's disagreement over the WHOLE report
    # (backend.domains.report_condition); the file also carries the design's
    # single-attribute `d_single` and the per-attribute `d_profile`.
    report_level: bool = False
    # How many report fields `d_profile` holds (0 = not a report-level cohort):
    # the width of the study's per-field dispersion head.
    n_fields: int = 0
    # ... and their names, in `d_profile` column order (empty = not report-level).
    field_names: tuple[str, ...] = ()
    # How results pages name the cohort and its positive class.
    title: str = ""
    positive: str = "positives"
    # The `disagreement` cut-off that puts this cohort's contested subgroup near
    # LIDC's ~14%, ON THIS COHORT'S OWN SCALE. It has to live here because the
    # scales are not comparable - LIDC's 1.0 is an s.d. of ratings 0..2, Chákṣu's
    # disagreement is bounded by 0.49 - so LIDC's value selects the EMPTY SET
    # there, and an empty subgroup reports "no effect" rather than an error. An
    # absolute, not a quantile: Chákṣu's disagreement takes three values, so no
    # quantile lands near 14%. None = no measurement; the config's own value
    # stands.
    contested_d: float | None = None
    # A SECOND CONDITION on an existing cohort rather than a new one: the name of
    # the cohort this is a variant of, or None for a cohort in its own right.
    #
    # It exists because such a variant must SHARE its base's frozen split - same
    # items, same subjects, same labels, so the two conditions are paired item for
    # item and a delta between them is a delta and not two different experiments.
    # That breaks the "one split file per spec" invariant every other cohort obeys,
    # and the difference between a deliberate share and an accidental one is
    # exactly the kind of thing that is invisible until a result is wrong. Saying
    # it here makes it checkable (`backend/tests/test_datasets.py`) instead of a
    # comment someone has to notice.
    note: str = ""

    def processed_dir(self, require_exists: bool = True) -> Path:
        """The folder of .npz samples, checked to exist unless `require_exists`
        is False (configuration-only callers that never open a sample)."""
        root = getattr(paths(), self.processed_key, None)
        if root is None:
            raise KeyError(
                f"configs/paths.yaml has no `{self.processed_key}` entry, which "
                f"the {self.name} dataset needs.")
        if not require_exists:
            return Path(root)
        return require(root, f"processed {self.name} samples")

    def require_capabilities(self, *capabilities: str) -> DatasetSpec:
        """Fail NOW if this cohort cannot support the study about to be run.

        The traps this catches are silent: a cohort with a constant label, or a
        constant `opinion_dist`, crashes nothing - the run completes and reports
        "no effect", which is indistinguishable from a genuine negative result
        and costs a GPU day to discover. A notebook that trains an opinion head says so up front:

            spec = bind(cfg)
            spec.require_capabilities("classification", "opinion_distribution")

        `ExperimentConfig.needed_capabilities()` supplies the list, so "which
        studies can run on which cohort" is answered by the config rather than
        restated in the notebook::

            spec.require_capabilities(*experiment.needed_capabilities())"""
        known = {"classification": self.classification,
                 "opinion_distribution": self.opinion_distribution}
        unknown = [c for c in capabilities if c not in known]
        if unknown:
            raise ValueError(
                f"unknown capability {unknown} (known: {sorted(known)})")

        missing = [c for c in capabilities if not known[c]]
        if missing:
            raise UnsupportedExperiment(
                f"dataset {self.name!r} does not support {missing}, which this "
                f"study needs. {self.note or ''}".strip() +
                f"\nPick a cohort that does, or run a study that does not need "
                f"it. Capabilities: {known}")
        return self


DATASETS: dict[str, DatasetSpec] = {
    "lidc": DatasetSpec(
        name="lidc",
        processed_key="processed",
        folds="configs/folds/folds_lidc.json",
        n_ratings=5,                       # malignancy 1..5
        subject="patient",
        disagreement="s.d. of the per-reader ratings, range 0..2",
        classification=True,
        contested_d=1.0,       # s.d. of ratings 0..2   -> 14.3% contested
    ),
    "chaksu": DatasetSpec(
        name="chaksu",
        processed_key="chaksu_processed",
        folds="configs/folds/folds_chaksu.json",
        n_ratings=2,                       # normal / glaucoma suspect
        subject="image, or the Bosch `P##` patient where the archive names one",
        disagreement="s.d. of the five binary votes, BOUNDED BY 0.5 and taking "
                     "only {0, 0.400, 0.490}",
        classification=True,
        contested_d=0.45,      # s.d. of 5 votes 0..0.49  -> 15.1% contested
        note="Five experts read every image and supply BOTH a mask and a "
             "decision, so 41.5% of panels are non-unanimous and 14.6% are "
             "genuine 2-3 splits. Two cautions. "
             "`contested_d: 1.0` selects the EMPTY SET on a 0..0.5 scale, and "
             "no quantile rescues it either because the variable takes "
             "three values (q=0.70 and q=0.80 both give 15.5%, q=0.90 gives "
             "0.2%) — set contested_d EXPLICITLY, 0.35 or 0.45. And prevalence "
             "is 14%, so expect wider CIs than on LIDC.",
    ),
    "lidc_report": DatasetSpec(
        name="lidc_report",
        processed_key="lidc_report_processed",
        folds="configs/folds/folds_lidc.json",
        n_ratings=5,                       # opinion_dist is still malignancy 1..5
        subject="patient",
        disagreement="mean over LIDC's 7 ordinal characteristics of the panel's "
                     "rating s.d., each in units of its cohort s.d.",
        classification=True,
        condition_of="lidc",
        report_level=True,
        n_fields=7,
        field_names=LIDC_ATTRS,
        title="LIDC-IDRI (lung CT)",
        positive="malignant-rated nodules",
        contested_d=1.8174,    # declared from labels before any run: 80th percentile
        note="LIDC with REPORT-LEVEL disagreement. Same images, labels, "
             "split and malignancy opinion as `lidc`; `disagreement` is over all "
             "7 ordinal characteristics, `d_single` keeps malignancy's d. "
             "rho(report d, malignancy d) +0.366. No mask is read.",
    ),
    "lndb": DatasetSpec(
        name="lndb",
        processed_key="lndb_report_processed",
        folds="configs/folds/folds_lndb.json",
        n_ratings=5,                       # malignancy 1..5, as LIDC
        subject="patient (one LNDb CT is one patient)",
        disagreement="mean over LIDC's 7 ordinal characteristics of the panel's "
                     "rating s.d., each in units of LIDC's cohort s.d.",
        classification=True,
        report_level=True,     # built at report level directly (backend.domains.lndb)
        n_fields=7,
        field_names=LIDC_ATTRS,
        title="LNDb (lung CT, a different hospital and reading panel)",
        positive="malignant-rated nodules",
        contested_d=1.2566,    # declared from labels before any run: 80th percentile
        note="LNDb (Pedrosa et al. 2019): 221 nodules >= 3 mm rated by 2-3 of five "
             "radiologists who read independently (no consensus review), on LIDC's "
             "fields. Prevalence 35/177 = 20% (neg:pos 4.1). Mostly two-reader panels, "
             "so each field's spread takes few values. Used as the EXTERNAL test of the "
             "LIDC-trained arms (shared/study_external.py); training on it is supported "
             "but small. CC BY-NC-ND: never redistribute the processed files.",
    ),
    "chaksu_report": DatasetSpec(
        name="chaksu_report",
        processed_key="chaksu_report_processed",
        folds="configs/folds/folds_chaksu.json",
        n_ratings=10,                      # opinion_dist is still VCDR on the 0.1 grid
        subject="image, or the Bosch `P##` patient where the archive names one",
        disagreement="mean over the experts' 10 recorded measurements (3 ratios: "
                     "s.d.; 7 sizes: coefficient of variation) of the panel's "
                     "disagreement, each in units of its cohort s.d.",
        classification=True,
        condition_of="chaksu",
        report_level=True,
        n_fields=10,
        field_names=CHAKSU_ATTRS,
        title="Chákṣu (fundus, glaucoma)",
        positive="glaucoma cases",
        contested_d=2.3349,    # declared from labels before any run: 80th percentile
        note="Chákṣu with REPORT-LEVEL disagreement. Same images, labels, "
             "split and VCDR opinion as the `chaksu_vcdr_processed` build; `disagreement` is over every "
             "measurement the experts recorded, `d_single` keeps VCDR's d. Read "
             "from the decision CSVs inside the release zips; no mask is read.",
    ),
}


def resolve(name: str) -> DatasetSpec:
    """Look up a cohort by name, with a listing instead of a KeyError."""
    try:
        return DATASETS[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r} (known: {sorted(DATASETS)})") from None


def run_dataset(cfg) -> str:
    """The cohort a run trains on: `dataset:` (base.yaml's cohort), or - when the
    study sets `report_level: true` - that cohort's report-level condition.
    Staging and `bind` both ask this, so they cannot disagree."""
    name = str(getattr(cfg, "dataset", "lidc")).lower()
    if not getattr(cfg, "report_level", False) or resolve(name).report_level:
        return name
    hits = [n for n, s in DATASETS.items() if s.report_level and s.condition_of == name]
    if not hits:
        raise ValueError(
            f"`report_level: true` needs a report-level condition of {name!r}; "
            f"there is none (known: {[n for n, s in DATASETS.items() if s.report_level]}). "
            f"Point base.yaml's `extends:` at cohorts/lidc or cohorts/chaksu.")
    return hits[0]


def cohort_data_version(name: str) -> str | None:
    """`data_version:` from the cohort's own file (a condition reads its base's),
    or None for the original build."""
    base = DATASETS[name].condition_of if name in DATASETS else None
    try:
        v = getattr(load(f"cohorts/{base or name}"), "data_version", None)
    except FileNotFoundError:
        return None
    return str(v) if v else None


def cohort_decentring(name: str) -> tuple[float, int]:
    """The decentring condition for a cohort: `(shift_frac, shift_seed)`.

    THE ONE PLACE. `configs/cohorts/<name>.yaml` holds `decentring:`, which is
    both the switch and the value - 0.0 is off - and this is the only thing that
    reads it. Read from the BOUND cohort's own file rather than from `cfg`, so
    switching `cfg.dataset` at run time picks up that cohort's condition.
    """
    base = DATASETS[name].condition_of if name in DATASETS else None
    name = base or name          # a condition of a cohort shares its cohort file
    try:
        cohort = load(f"cohorts/{name}")
    except FileNotFoundError:
        raise KeyError(
            f"cohort {name!r} has no configs/cohorts/{name}.yaml, so its "
            f"decentring condition is undeclared") from None
    if not hasattr(cohort, "decentring"):
        raise KeyError(
            f"configs/cohorts/{name}.yaml must declare `decentring:` - it is the "
            f"switch AND the value for this cohort's condition (0.0 = off)")
    shift = float(cohort.decentring)
    if not 0.0 <= shift < 1.0:
        raise ValueError(
            f"configs/cohorts/{name}.yaml: decentring is a fraction of width and "
            f"must be in [0, 1), got {shift}")
    return shift, int(getattr(cohort, "shift_seed", 0))


def bind(cfg, require_data: bool = True) -> DatasetSpec:
    """Point a notebook's `cfg` at the cohort named by `cfg.dataset`.

    Sets `processed_dir`, `folds`, `n_ratings`, `n_fields`, `run_dir` and the
    cohort's own `contested_d`. Returns the spec so the notebook can print what
    it bound to.

    `require_data=False` resolves and validates the configuration without
    checking that the processed samples exist - for tests of configuration
    invariants on a machine without the (non-redistributable) datasets. Every
    training and evaluation path keeps the fail-closed default."""
    spec = resolve(run_dataset(cfg))
    cfg.dataset = spec.name

    # A STUDY CAN PIN ITS COHORT. `configs/notebooks/base.yaml` selects the
    # cohort by extending one `configs/cohorts/<name>.yaml`, which is one line - and one
    # line is exactly what gets changed for a side experiment and forgotten. A
    # study whose result is FINAL says which cohort it is a result ON, and this
    # refuses to run it against any other. Nothing is coerced: an unpinned study
    # (the default) behaves exactly as before.
    # A condition of the pinned cohort (its report-level build: same images,
    # labels and split) is still that cohort.
    required = getattr(cfg, "require_dataset", None)
    if required is not None and str(required) not in (spec.name, spec.condition_of):
        raise ValueError(
            f"this study is pinned to `require_dataset: {required}`, but the "
            f"cohort selected in configs/notebooks/base.yaml is {spec.name!r}. "
            f"Point base.yaml's `extends:` at cohorts/{required}, or drop "
            f"the pin if you meant to run on {spec.name}.")

    cfg.processed_dir = spec.processed_dir(require_exists=require_data)
    cfg.folds = spec.folds
    cfg.n_ratings = spec.n_ratings
    cfg.n_fields = spec.n_fields

    # DECENTRING, from the ONE place that decides it.
    cfg.shift_frac, cfg.shift_seed = cohort_decentring(spec.name)

    # OUTPUT NAMESPACE. The notebook does `cfg.out = P.runs / cfg.run_dir`
    # right after this call, and the per-fold cache tag underneath it
    # (`<model>_<tag>.result.json`, tag = the fold / mode / arm) carries NO
    # dataset component. So switching `dataset` alone and leaving run_dir at
    # `baseline` would point a Chákṣu run at LIDC's folder, where
    # cached_result() returns the LIDC fold and resume() reloads LIDC weights —
    # silently, since a cached result is the normal fast path. Suffixing here
    # rather than in each yaml file is what lets `dataset:` be the ONLY switch.
    #
    # LIDC keeps the bare name so every path already measured stays valid, and
    # the suffix is applied at most once so re-running the config cell in a live
    # kernel cannot produce `baseline_chaksu_chaksu`.
    # The cohort suffix goes on the UNDATED stem. Applied to a dated name the
    # endswith() guard below would see no cohort at the end and append a second
    # one - a folder no cached fold lives in.
    stamped = bool(getattr(cfg, "date_stamp", False))
    fmt = getattr(cfg, "date_fmt", "%d-%m-%y")
    stem = _unstamped(str(cfg.run_dir), fmt) if stamped else str(cfg.run_dir)

    # STRIP ANY CONDITION SUFFIX FIRST, so the cohort guard below still sees the
    # cohort at the END of the stem. Without this, binding twice in a live kernel
    # gives `..._chaksu_dc20_chaksu_dc20` - the same stacking the date-stamp
    # comment above warns about, reintroduced by the condition suffix.
    version = cohort_data_version(spec.name)
    if version and stem.endswith(f"_{version}"):
        stem = stem[: -len(version) - 1]
    stem = re.sub(r"_dc\d+$", "", stem)

    suffix = f"_{spec.name}"
    if spec.name != "lidc" and not stem.endswith(suffix):
        stem = f"{stem}{suffix}"

    # THE CONDITION GOES IN THE PATH TOO. Decentring is a different EXPERIMENT,
    # not a different setting: a fold trained at 0.20 cannot be compared with,
    # or cached against, one trained centred. Without this the two land in the
    # same dated folder, and because the freshness check now re-runs a fold whose
    # `shift_frac` differs, the new condition would OVERWRITE the banked one.
    if float(getattr(cfg, "shift_frac", 0.0)) > 0.0:
        dc = f"_dc{int(round(100 * float(cfg.shift_frac)))}"
        if not stem.endswith(dc):
            stem = f"{stem}{dc}"
    # THE DATA BUILD GOES IN THE PATH TOO. A cohort whose samples were rebuilt
    # (new crops, same targets and folds) declares `data_version:` in its cohort
    # file; results from the old build then cannot be resumed or cached into the
    # new one. Absent = the original build, so existing folders keep their names.
    if version and not stem.endswith(f"_{version}"):
        stem = f"{stem}_{version}"
    cfg.run_dir = _date_stamped(stem, fmt) if stamped else stem

    # The contested cut-off, from the cohort's own measured value. The yaml
    # CANNOT be right for more than one cohort at a time - `contested_d: 1.0` is
    # LIDC's rating-s.d. and selects the empty set on Chákṣu's 0..0.49 scale -
    # and getting it wrong is silent: an empty contested subgroup reports "no
    # effect".
    if spec.contested_d is not None:
        was = getattr(cfg, "contested_d", None)
        cfg.contested_d = spec.contested_d
        if was is not None and abs(float(was) - spec.contested_d) > 1e-9:
            print(f"  [{spec.name}] contested_d {was:g} -> {spec.contested_d:g} "
                  f"(the config's value is on another cohort's scale)")

    print(f"dataset = {spec.name} | subject = {spec.subject} | K = {spec.n_ratings}")
    print(f"  samples <- {cfg.processed_dir}")
    print(f"  split   <- {cfg.folds}")
    print(f"  results -> runs/{cfg.run_dir}")
    print(f"  disagreement: {spec.disagreement}")
    print(f"  contested_d = {cfg.contested_d:.4g}")
    # The note is where a cohort's silent traps are written down, so print it
    # whenever there is one.
    if spec.note:
        print(f"  {'note' if spec.classification else 'WARNING'}: {spec.note}")
    return spec


def _date_stamped(run_dir: str, fmt: str) -> str:
    """`run_dir` with today's date appended — unless an EARLIER dated run of the
    same study is already on disk with results in it, in which case that one is
    reused.

    Without a stamp, two protocols run under the same `run_dir` would share a
    folder, and `cached_result()` would answer the second run with the first
    one's numbers.

    THE ADOPTION RULE IS THE POINT. A bare stamp would be actively dangerous on
    Colab: a fold that trains overnight, or a session that dies and is restarted
    the next morning, would come back to a fresh empty folder and throw away
    every checkpoint — on a project that is already compute-limited. So when a
    dated sibling exists and is non-empty, the run CONTINUES there and says so.
    Start a genuinely new protocol by changing `run_dir` itself, which is what
    distinguishes one protocol from another anyway.

    Already-stamped input is returned untouched, so re-running the config cell in
    a live kernel cannot produce a doubled date."""
    run_dir = str(run_dir)
    if _stamp_of(run_dir, fmt) is not None:
        return run_dir              # already stamped - today's, or one adopted
                                    # earlier and handed straight back in
    today = date.today().strftime(fmt)

    base = run_dir
    for old in _dated_siblings(base, fmt):
        if old.endswith(f"-{today}"):
            return old                      # today's, from an earlier cell run
        print(f"  [date_stamp] resuming {old} (non-empty) rather than "
              f"starting {base}-{today}; change run_dir for a new protocol")
        return old
    return f"{base}-{today}"


def _dated_siblings(base: str, fmt: str) -> list[str]:
    """Non-empty `runs/<base>-<date>` folders, newest date first.

    Parsed with the SAME `fmt` the stamp is written with, so a folder that merely
    happens to start with `base-` is not mistaken for a dated run of it."""
    try:
        root = Path(paths().runs)
    except Exception:                       # no paths.yaml yet: nothing to adopt
        return []
    found = []
    for d in root.glob(f"{base}-*"):
        if not d.is_dir() or not any(d.iterdir()):
            continue
        when = _stamp_of(d.name[len(base) + 1:], fmt)
        if when is not None:
            found.append((when, d.name))
    return [name for _, name in sorted(found, reverse=True)]


def _stamp_of(name: str, fmt: str):
    """The date `name` ends in, or None. `name` may BE the stamp, or end in one
    after a `-`. Every `-` is tried, because how many of them a stamp contains is
    the format's business, not this function's."""
    for candidate in (name, *(name[i + 1:] for i, ch in enumerate(name)
                              if ch == "-" and i)):
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return None


def _unstamped(name: str, fmt: str) -> str:
    """`name` with a trailing date stamp removed, if it carries one."""
    for i, ch in enumerate(name):
        if ch == "-" and i and _stamp_of(name[i + 1:], fmt) is not None:
            return name[:i]
    return name
