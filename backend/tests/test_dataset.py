"""Assertions about the REAL processed dataset.

`test_preprocessing.py` proves the preprocessing logic is correct on synthetic
input and runs anywhere. This proves the files actually on disk obey it. It
needs the dataset, so every test here skips cleanly when `configs/paths.yaml`
does not point at one — but where the dataset IS present, this is what makes a
violated invariant a failed build rather than a line in a report.

"""
from __future__ import annotations

import json

import numpy as np
import pytest

from field_disagreement import splits as S

from backend.config import ROOT, load, paths
from backend.domains.datasets import resolve
from backend.domains.lidc.patches import label_from_scores
from backend.domains.lidc.readers import MAX_READERS


def processed_dir():
    try:
        d = paths().processed
    except (FileNotFoundError, KeyError):
        return None
    return d if d.is_dir() and any(d.glob("*.npz")) else None


needs_data = pytest.mark.skipif(processed_dir() is None,
                                reason="no processed dataset on this machine")


@pytest.fixture(scope="module")
def files():
    return sorted(processed_dir().glob("*.npz"))


@pytest.fixture(scope="module")
def manifest():
    return S.build_manifest(processed_dir())


def frozen_path():
    """LIDC's frozen split, taken from the REGISTRY and not from base.yaml.

    Everything in this file walks `paths().processed`, which is LIDC's folder by
    definition — but `base.yaml.folds` follows `base.yaml.dataset`, which is the
    one-word switch a run flips to train on another cohort. Reading it here meant
    a legitimate `dataset: chaksu` made three LIDC tests fail with "693 subjects
    are not in the frozen split folds_chaksu.json". Same trap, same fix as
    `test_datasets.py`: pin the cohort where the test is about one cohort."""
    return ROOT / resolve("lidc").folds


# --------------------------------------------------------------------------- #
# every file obeys the preprocessing contract
# --------------------------------------------------------------------------- #
@needs_data
def test_every_nodule_obeys_the_preprocessing_contract(files):
    """Swept over the whole dataset, not a sample: one bad file is enough to
    make a downstream number wrong."""
    cfg = load("preprocessing/preprocess_lidc")
    P, channels = cfg.patch_size, 2 * cfg.neighbours + 1
    bad: dict[str, list[str]] = {}

    def note(kind, name):
        bad.setdefault(kind, []).append(name)

    for f in files:
        with np.load(f, allow_pickle=True) as d:
            name = f.name
            amap = d["agreement_map"].astype(np.int64)
            frac = d["agreement_frac"].astype(np.float64)
            mask = d["binary_mask"].astype(bool)
            scores = np.asarray(d["malignancy"]).astype(np.int64)
            nr = int(d["n_readers"])

            if d["image"].shape != (channels, P, P):
                note("image shape", name)
            if amap.shape != (P, P) or mask.shape != (P, P) or frac.shape != (P, P):
                note("map shape", name)
            if d["image"].min() < -1e-6 or d["image"].max() > 1 + 1e-6:
                note("image outside [0,1]", name)
            if nr < cfg.min_readers:
                note(f"fewer than {cfg.min_readers} readers", name)
            if nr > MAX_READERS:
                note(f"more than {MAX_READERS} readers (a LIDC panel has at "
                     f"most {MAX_READERS}; this is contours counted as people)",
                     name)
            if "reader_ids" in d.files:
                rids = np.asarray(d["reader_ids"]).ravel()
                if len(set(rids.tolist())) != nr:
                    note("n_readers disagrees with the distinct reader_ids", name)
                if len(rids) != len(scores):
                    note("one reader_id per rating expected", name)
            if amap.max() > nr:
                note("reader count exceeds panel size", name)
            if not np.allclose(frac, amap / nr, atol=1e-6):
                note("agreement_frac != agreement_map / n_readers", name)
            if frac.min() < 0 or frac.max() > 1 + 1e-6:
                note("agreement_frac outside [0,1]", name)
            if not np.array_equal(mask, amap > 0):
                note("binary_mask != agreement_map > 0", name)
            if not mask.any():
                note("empty mask", name)
            if abs(float(np.asarray(d["opinion_dist"]).sum()) - 1.0) > 1e-5:
                note("opinion_dist not normalised", name)
            if int(d["label"]) != label_from_scores(scores):
                note("label disagrees with the median rule", name)

    assert not bad, "\n".join(
        f"{kind}: {len(v)} file(s) e.g. {v[:3]}" for kind, v in bad.items())


# --------------------------------------------------------------------------- #
# the frozen split still describes this dataset
# --------------------------------------------------------------------------- #
@needs_data
def test_frozen_split_matches_the_dataset_on_this_machine(manifest):
    """Raises if the data changed since the folds were frozen — which would
    make results measured before and after quietly incomparable."""
    S.load_fold_assignment(frozen_path(), manifest)


@needs_data
def test_frozen_folds_are_patient_disjoint(manifest):
    cfg = load("notebooks/base")
    assign = S.load_fold_assignment(frozen_path(), manifest)
    for k in range(cfg.n_splits):
        S.assert_no_leakage(manifest, S.make_fold(manifest, assign, k, cfg.n_splits))


@needs_data
def test_every_patient_has_exactly_one_fold(manifest):
    cfg = load("notebooks/base")
    assign = S.load_fold_assignment(frozen_path(), manifest)
    patients = set(manifest.patient_id.astype(str))
    assert patients <= set(assign)
    folds = {assign[p] for p in patients}
    assert folds == set(range(cfg.n_splits))


# --------------------------------------------------------------------------- #
# the files were built by the config that is checked in
# --------------------------------------------------------------------------- #
@needs_data
def test_files_were_built_by_the_committed_preprocessing_config():
    """Catches a config edit that was never followed by a re-run."""
    pfile = processed_dir() / "preprocess_provenance.json"
    if not pfile.exists():
        pytest.skip("no preprocess_provenance.json — re-run prepare to write one")
    recorded = json.loads(pfile.read_text(encoding="utf-8"))["config"]
    current = vars(load("preprocessing/preprocess_lidc"))
    drift = {k: (recorded.get(k), v) for k, v in current.items() if recorded.get(k) != v}
    assert not drift, f"processed files predate the current config: {drift}"
