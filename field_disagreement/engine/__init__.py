"""The composed entry points: the training loop, the whole study, and the triage decision.

`field_disagreement.engine` holds what COMPOSES the mechanisms in
`field_disagreement.triage` and `field_disagreement.metrics`.

TRAINING - the epoch loop, with the study's own step passed in::

    runner = field_disagreement.ExperimentRunner(cfg, store, device)
    best   = runner.fit(fold, train_step, selection="auc_band", min_epochs=10)

The loop is handed a model and two loaders and never asks where they came from;
building a fold (loaders, class balance, the per-fold cache) stays in
`backend/notebooks/shared/runner.py`. `TriageExperiment` composes the loop with the
model, objective, optimiser and schedule the typed config describes, over every fold.

DECISION - one case (`decide_or_defer`) or a whole split of stored predictions
(`decide_batch`)::

    dec = field_disagreement.decide_or_defer(prediction, calibrator=cal, policy=policy)
"""
from __future__ import annotations

from .calibration import Calibrator, DEFAULT_CLASS_NAMES
from .decide import (DECIDE, DEFER, VETO_RULE, Decision, Prediction, SignalContribution,
                     decide_batch, decide_or_defer)
from .experiment import TriageExperiment, FoldSource
from .history import selection_summary, signal_learned_by
from .optim import (amp_components, build_cls_loss, build_optimizer,
                    build_scheduler, set_seed)
from .runner import (SELECTIONS, ExperimentRunner, Fold, Selection, predict,
                     predict_full)

__all__ = [
    "Calibrator", "DEFAULT_CLASS_NAMES",
    "Prediction", "Decision", "SignalContribution", "decide_or_defer",
    "decide_batch", "DECIDE", "DEFER", "VETO_RULE",
    "ExperimentRunner", "Fold", "Selection", "SELECTIONS", "predict",
    "predict_full", "selection_summary", "signal_learned_by",
    "TriageExperiment", "FoldSource",
    "build_optimizer", "build_scheduler", "amp_components", "build_cls_loss",
    "set_seed",
]
