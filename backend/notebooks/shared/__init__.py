"""Boilerplate every experiment notebook needs, defined once.

The notebook spells out its own run - the arms, the seeds, the tables - so a
reader can follow it without opening a driver module. Everything around it is
here:

    colab      staging the repo and this run's cohort onto a Colab VM
    kaggle     the same for a Kaggle session, plus the local upload bundle
    session    which runtime this is, and stopping cleanly inside its time limit
    data       the manifest and the frozen fold assignment
    training   seeding, loaders, the differential-LR AdamW, cosine schedule, AMP
    runner     the DOMAIN half of a run - this fold's loaders and class
               balance, and the result cache around them. The loop itself is
               `field_disagreement.engine.ExperimentRunner`, in the shipped package.
    evaluate   the prediction passes over a split
    checkpoints the resumable store that keeps a free-GPU run alive across a
               disconnect
    report     pooling per-fold results and formatting the tables
    study      what a run covers
    study_arms the study's per-fold build / train / evaluate (train_evaluate)
    study_results  the study's tables, read back from a run folder
    panel      the panel-risk battery: separation, leakage, team, certificate
    study_external scoring the trained folds on the external cohort (LNDb)
    study_summary the per-item predictions and summary tables written by a run
    demo       the cohort and case figures the notebook draws

It lives here so that it is written once. Nothing in this package knows anything about a
contribution: it takes a `cfg` and returns an optimiser, or it takes a loader and
returns metrics.

Anything genuinely specific to one notebook stays in that notebook.
"""
from __future__ import annotations

from .checkpoints import CheckpointStore
from .data import cluster_map, manifest_and_folds, subset_patients
from .evaluate import panel_stats, predict_full
from .report import (auc_parity, flatten, pool_predictions, save_summary)
from .runner import CohortFolds, FoldRunner
from . import panel, study_arms, study_results
from . import study_external, study_summary, demo
from .panel import (panel_battery, panel_signals, separation_report,
                    leakage_report, team_report, certificate_report)
from .study import BASE_SEED, fold_count_note, run_scope, screen_arms
from .training import (amp_components, build_cls_loss, build_optimizer,
                       build_scheduler, loaders_for, pos_weight_for, predict,
                       set_seed)

__all__ = [
    # training plumbing
    "set_seed",
    "BASE_SEED",
    "demo",
    "pos_weight_for",
    "loaders_for",
    "build_optimizer",
    "build_scheduler",
    "amp_components",
    "build_cls_loss",
    "predict",
    # data
    "manifest_and_folds",
    "subset_patients",
    "cluster_map",
    # the per-fold unit of work (the loop is field_disagreement.engine.ExperimentRunner)
    "FoldRunner",
    # the same domain half as a FoldSource, for field_disagreement.TriageExperiment
    "CohortFolds",
    # evaluation passes
    "predict_full",
    "panel_stats",
    # checkpointing
    "CheckpointStore",
    # pooling per-fold results + formatting the tables
    "pool_predictions",
    "flatten",
    "save_summary",
    "auc_parity",
    # the panel-risk battery
    "panel",
    "study_results",
    "panel_battery",
    "panel_signals",
    "separation_report",
    "leakage_report",
    "team_report",
    "certificate_report",
    # what a run covers
    "run_scope",
    "fold_count_note",
    "screen_arms",
]
