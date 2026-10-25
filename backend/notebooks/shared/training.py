"""Seeding, loaders, optimiser, schedule, AMP, and a plain prediction pass.

Every function here takes what it needs as an argument - there is no module
state and nothing reads a notebook global - so the same call works from any
notebook and from a test.

Most of this module is now a re-export. `predict`, the optimiser, the schedule,
AMP and the classification loss all live in `field_disagreement.engine` rather than here,
for the reason `predict` always did: they are read by the framework's own
training loop, so a package that shipped the loop without them shipped a loop
nobody could supply an optimiser to. The names stay available from this module
because that is where the notebook imports them from.

What genuinely stays is the DOMAIN half - `loaders_for`, which knows the
processed-.npz layout and the frozen split, and `pos_weight_for`, which reads
the manifest for a fold's class balance.
"""
from __future__ import annotations

from field_disagreement import model_input_spec
from field_disagreement.config import ExperimentConfig
from field_disagreement.engine import (amp_components, build_cls_loss, build_optimizer,
                             build_scheduler, predict, set_seed)
from backend.domains import adapter as D

__all__ = [
    "amp_components",
    "build_cls_loss",
    "build_optimizer",
    "build_scheduler",
    "loaders_for",
    "balanced_sampling",
    "pos_weight_for",
    "predict",
    "set_seed",
]


CLASS_BALANCE = ("weighted_loss", "balanced_sampler")


def balanced_sampling(cfg) -> bool:
    """`class_balance:` in base.yaml. `weighted_loss` (the default, and the
    protocol of every reported run) up-weights positives in the loss;
    `balanced_sampler` instead draws training batches with equal class odds and
    leaves the loss unweighted, so the imbalance is corrected once, not twice."""
    mode = str(getattr(cfg, "class_balance", "weighted_loss"))
    if mode not in CLASS_BALANCE:
        raise ValueError(f"class_balance must be one of {CLASS_BALANCE}, got {mode!r}")
    return mode == "balanced_sampler"


def pos_weight_for(df, files, cfg=None):
    """BCE pos_weight = #neg / #pos over the training files (class imbalance);
    1.0 when the sampler already balances the classes."""
    if cfg is not None and balanced_sampling(cfg):
        return 1.0
    lab = df.set_index("file").loc[files, "label"]
    n_pos = int((lab == 1).sum()); n_neg = int((lab == 0).sum())
    return (n_neg / n_pos) if n_pos else 1.0


def loaders_for(cfg, df, patient_fold, fold):
    """Per-fold train/val/test/score3 loaders, sized + normalised for the backbone.

    The batch protocol - size, workers, and `shift_frac` (decentring,
    contribution B) - comes from the typed `DataLoaderConfig`, whose 0.0
    `shift_frac` default is `make_loaders`' own. A notebook that never sets it
    gets the historical behaviour bit for bit, and the one that does needs no
    separate copy of this function.

    What stays here is the part no framework can supply: which folder the
    processed cohort is in, and which items this fold's frozen split names."""
    size, norm = model_input_spec(cfg.model)
    data = ExperimentConfig.of(cfg).data
    return D.make_loaders(cfg.processed_dir, df, patient_fold, fold,
                          batch_size=data.batch_size,
                          num_workers=data.num_workers,
                          img_size=size, normalize=norm,
                          shift_frac=data.shift_frac,
                          shift_seed=data.shift_seed,
                          balanced=balanced_sampling(cfg))

