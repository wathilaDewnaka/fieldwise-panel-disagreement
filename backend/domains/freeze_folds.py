"""Freeze a domain's subject-level 5-fold split into a committed file.

The folds used to be recomputed on every run by `stratified_patient_kfold`.
That is deterministic given the manifest, but the manifest is built by scanning
whatever .npz files happen to be on disk — so adding, dropping or re-running
preprocessing on one sample silently reshuffles subjects between folds, and
numbers from before and after are no longer comparable.

This writes the assignment once, together with a fingerprint of the manifest it
came from. Training then LOADS it. Re-running on unchanged data reproduces the
same file byte for byte.

    python -m backend.domains.freeze_folds                    # LIDC (the default)
    python -m backend.domains.freeze_folds --dataset chaksu --force

One file per dataset, `configs/folds/folds_<cohort>.json`, because the split is a
property of the cohort: the folds of one domain say nothing about another, and a
single split file shared between them would be silently wrong for whichever one
wrote it second.
The subject a fold groups by is `patient_id` in the .npz — the patient for LIDC
and LNDb; the image for Chákṣu, or the patient where the archive names one.

The paper's fold files are in `configs/folds/`; re-running this on the paper's
cohorts reproduces them.
"""
from __future__ import annotations

import argparse

from field_disagreement import splits as S

from backend.config import ROOT, load, paths, require
from backend.domains.datasets import DATASETS, resolve


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze a dataset's subject-level CV split.")
    ap.add_argument("--dataset", default="lidc", choices=sorted(DATASETS),
                    help="which cohort to split (default: lidc)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing frozen split (invalidates past results)")
    args = ap.parse_args()

    spec = resolve(args.dataset)
    cfg = load("notebooks/base")
    out = ROOT / spec.folds

    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} already exists. The split is meant to be frozen — pass --force "
            f"only if you intend to invalidate every result measured on it.")

    processed = require(getattr(paths(), spec.processed_key),
                        f"processed {args.dataset} samples")
    df = S.build_manifest(processed, rebuild=True)

    # A cohort whose label never varies cannot be split into comparable folds for
    # a classifier: say so here rather than let a fold table of all-positives
    # look healthy.
    if df.label.nunique() < 2:
        print(f"WARNING: every sample in {args.dataset} has label {df.label.iloc[0]}. "
              f"The split is still written, but this cohort cannot "
              f"train or evaluate a classifier.")

    assign = S.stratified_patient_kfold(df, n_splits=cfg.n_splits, seed=cfg.split_seed)

    # A frozen split is worthless if it leaks, so prove it here, on every fold,
    # before it is written — not only in the test suite.
    for k in range(cfg.n_splits):
        S.assert_no_leakage(df, S.make_fold(df, assign, k, cfg.n_splits))

    blob = S.save_fold_assignment(out, assign, df,
                                  seed=cfg.split_seed, n_splits=cfg.n_splits)

    print(f"wrote {out}")
    print(f"  subjects {blob['n_patients']}  samples {blob['n_items']}  "
          f"seed {blob['seed']}  fingerprint {blob['manifest_fingerprint']}")
    print()
    print(S.fold_summary(df, assign, cfg.n_splits).to_string(index=False))


if __name__ == "__main__":
    main()
