"""Optimiser, schedule, AMP and the classification loss, built from typed config.

These are part of the package rather than the notebook. That was defensible while they read a flat namespace
nobody had typed, but it meant `field_disagreement` shipped a loop it could not actually
supply an optimiser to: anyone installing the package still had to write the
differential-LR grouping and the warm-up-then-cosine schedule themselves, and
those are exactly the two that had drifted between notebook copies.

Nothing here knows what the data is. `build_cls_loss` takes `pos_weight` as an
argument for that reason - class balance is a property of a split, so computing
it needs the manifest and stays with the domain, while CHOOSING between weighted
BCE and focal is protocol and belongs to the config.

    opt    = build_optimizer(model, cfg)
    sched  = build_scheduler(opt, cfg)
    amp, scaler = amp_components(device, cfg)

`cfg` is anything `ExperimentConfig.of` resolves: the typed config, or the
flat namespace the notebook loads.
"""
from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
from torch import nn

from ..config import ExperimentConfig
from ..objectives import focal_loss

__all__ = ["amp_components", "build_cls_loss", "build_optimizer",
           "build_scheduler", "set_seed"]


def set_seed(seed: int) -> None:
    """Seed every generator a run draws from.

    `torch.cuda` is seeded unconditionally: it is a no-op without CUDA, and
    guarding it is how a run ends up seeded on one machine and not the other.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN picks algorithms that use atomics, so the same seed on the same GPU
    # can still diverge run to run. `deterministic` costs some speed; it does not
    # remove the need for seed REPEATS, it just stops one seed from being two
    # different experiments.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # cuDNN determinism is not enough on its own: some CUDA ops (e.g. the backward
    # of bilinear upsampling) accumulate with atomicAdd and stay nondeterministic
    # whatever cuDNN is told.
    #
    # `warn_only=True` rather than True: the point is not to forbid a
    # nondeterministic op, it is to make any REMAINING one announce itself by
    # name instead of quietly costing more accuracy than the effect being
    # measured. A run that prints nothing here is reproducible; one that prints
    # has named its own bug.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_optimizer(model, cfg):
    """AdamW with the differential learning rate the pretrained trunks want.

    A backbone that exposes `param_groups` splits itself into the pretrained
    trunk (which wants the smaller step) and everything learned from scratch.
    One that does not has no such split and trains entirely at `lr_head` - which
    is the right answer for a from-scratch net rather than a missing feature.
    """
    opt = ExperimentConfig.of(cfg).optimizer
    if hasattr(model, "param_groups"):
        groups = model.param_groups(opt.lr_trunk, opt.lr_head)
    else:
        groups = [{"params": list(model.parameters()), "lr": opt.lr_head}]
    return torch.optim.AdamW(groups, weight_decay=opt.weight_decay)


def build_scheduler(opt, cfg):
    """Linear warm-up for `warmup` epochs, then cosine decay to 0.

    The same multiplier scales every group, so whatever ratio
    `build_optimizer` set between the trunk and the heads holds for the whole
    run rather than closing as the schedule decays.
    """
    train = ExperimentConfig.of(cfg).training
    warm = max(1, train.warmup)
    span = max(1, train.epochs - train.warmup)

    def lr_lambda(epoch):
        if epoch < train.warmup:
            return float(epoch + 1) / warm
        prog = min(1.0, (epoch - train.warmup) / span)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def amp_components(device, cfg):
    """(use_amp, scaler). Mixed precision is active only on CUDA."""
    use = (device.type == "cuda") and ExperimentConfig.of(cfg).training.amp
    return use, torch.amp.GradScaler(enabled=use)


def build_cls_loss(cfg, pos_weight=None):
    """Class-weighted BCE (default) or focal loss.

    `pos_weight` is passed in rather than derived: it is #neg/#pos over the
    TRAINING split, so computing it needs the manifest and the fold assignment -
    the domain concepts this package stays clear of. None = unweighted.
    """
    train = ExperimentConfig.of(cfg).training
    if train.loss == "focal":
        alpha, gamma = train.focal_alpha, train.focal_gamma
        return lambda logit, y: focal_loss(logit, y, alpha=alpha, gamma=gamma)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return lambda logit, y: bce(logit, y)
