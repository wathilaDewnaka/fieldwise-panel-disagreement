"""Field-level prediction of expert-panel disagreement.

The arms, targets and triage rule of the study live in `field_disagreement.triage`;
the arms are defined in `field_disagreement/triage/ladder.py`.

Nothing in this package knows about any particular dataset. The cohorts are read
by `backend/domains/{lidc,chaksu,lndb}`, which share one reader,
`backend/domains/adapter.py`; it stays outside this package because it hard-codes
the .npz storage layout and the cohorts' field names.

The paper's network is `build_model("resnet50")`. Any encoder emitting an
(N, C, h, w) feature grid can be registered with one call::

    import field_disagreement as fd
    fd.register_backbone("mymodel", fd.adapter_spec(my_encoder_fn))
    model = fd.build_model("mymodel")

The training loop takes a model and two loaders and is handed the study's own
step::

    runner = fd.ExperimentRunner(cfg, store, device)
    best   = runner.fit(fold, train_step, selection="auc_band")

`TriageExperiment` composes that loop with the model, objective, optimiser and
schedule the typed config describes, over every fold. The triage decision for one
case is `fd.decide_or_defer(prediction, calibrator=..., policy=...)`.
"""
from __future__ import annotations

from . import metrics, triage, splits, objectives
from . import engine
from .registry import BackboneSpec, register_backbone, build_model, model_input_spec, BACKBONES
from .backbones import AdapterBackbone, adapter_spec
from .spectrum import (agreement_surface, agreement_from_count,
                       mean_pairwise_dice, opinion_histogram, rating_entropy,
                       rating_spread, median_label)
from .config import (ConformalConfig, DataLoaderConfig,
                     DeferralConfig, ExperimentConfig, InvalidConfig,
                     ModelConfig, OpinionConfig, OptimizerConfig,
                     TrainingConfig, UncertaintyConfig)
from .engine import (TriageExperiment, Calibrator, Decision, ExperimentRunner,
                     Fold, FoldSource, Prediction, Selection, decide_batch,
                     decide_or_defer)
from .objectives import Objective

__version__ = "1.0.0"

__all__ = [
    # the panel reductions a domain uses
    "agreement_surface", "agreement_from_count",
    "mean_pairwise_dice", "opinion_histogram", "rating_entropy",
    "rating_spread", "median_label",
    "BackboneSpec", "register_backbone", "build_model", "model_input_spec", "BACKBONES",
    "AdapterBackbone", "adapter_spec",
    # inference façade
    "Prediction", "Decision", "Calibrator", "decide_or_defer", "decide_batch",
    # training façade
    "ExperimentRunner", "Fold", "Selection", "Objective",
    # the whole study, composed from the config
    "TriageExperiment", "FoldSource",
    # typed configuration
    "ExperimentConfig", "ModelConfig", "OpinionConfig",
    "UncertaintyConfig", "ConformalConfig",
    "DeferralConfig", "TrainingConfig", "OptimizerConfig", "DataLoaderConfig",
    "InvalidConfig",
    # sub-packages
    "metrics", "triage", "splits", "objectives",
    "engine",
]
