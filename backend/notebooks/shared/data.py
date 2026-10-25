"""The manifest and the frozen fold assignment - loaded the same way everywhere.

Three steps: build the manifest from the processed folder, optionally shrink it
to a handful of subjects for a local smoke run, and LOAD (never recompute) the
frozen split.
"""
from __future__ import annotations

import numpy as np

from field_disagreement import splits as S
from backend.config import ROOT


def subset_patients(df, max_patients, seed):
    """Shrink the manifest to `max_patients` whole subjects, for a fast local run.

    WHOLE subjects, so the subject-disjoint folds stay valid, and
    positive-bearing subjects first, so every fold still contains both classes at
    60 subjects instead of degenerating into an all-negative split."""
    rng = np.random.default_rng(seed)
    by_pat = df.groupby("patient_id")["label"]
    pos = [p for p, lab in by_pat if (lab == 1).any()]
    rest = [p for p in df["patient_id"].unique() if p not in set(pos)]
    rng.shuffle(pos)
    rng.shuffle(rest)
    keep = (pos + rest)[:max_patients]
    out = df[df["patient_id"].isin(keep)].reset_index(drop=True)
    print(f"[subset] limited to {len(keep)} subjects / {len(out)} samples")
    return out


def manifest_and_folds(cfg, *, verbose: bool = True):
    """(manifest, fold assignment) for this run's cohort. Returns `(df, patient_fold)`.

    The folds are FROZEN in `cfg.folds` and only LOADED here, never recomputed.
    `stratified_patient_kfold` reads whatever .npz files happen to be on disk, so
    re-running preprocessing could reshuffle subjects between folds and make
    results from before and after quietly incomparable. `load_fold_assignment`
    checks the manifest fingerprint recorded in the file and raises if the data
    drifted underneath it.

    A `max_patients` subset deliberately has a different fingerprint, so the
    check is relaxed for exactly that case and no other."""
    df = S.build_manifest(cfg.processed_dir)
    if cfg.max_patients is not None:
        df = subset_patients(df, cfg.max_patients, getattr(cfg, "split_seed", cfg.seed))
    patient_fold = S.load_fold_assignment(
        ROOT / cfg.folds, df, strict=cfg.max_patients is None)
    if verbose:
        print(S.fold_summary(df, patient_fold, cfg.n_splits).to_string(index=False))
    return df, patient_fold


def cluster_map(df):
    """{sample id: subject id}, the clustering the item-level bootstrap resamples.

    Several samples can come from one subject, so treating them as independent
    would report an interval far narrower than the evidence supports."""
    return dict(zip(df["file"].astype(str), df["patient_id"].astype(str)))
