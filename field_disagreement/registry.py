"""The open backbone registry: build_model / model_input_spec / register_backbone.

Every backbone - the hand-written classes in `backbones/builtin.py` AND any
plugged-in encoder - is a `BackboneSpec` in `BACKBONES`; `build_model(name)` and
`model_input_spec(name)` are single look-ups. The registry is OPEN: a user adds a
backbone at run time with one call,

    register_backbone("mymodel", adapter_spec(my_encoder_fn))   # my_encoder_fn(pt)->(enc,C)

with no library edit; `adapter_spec` supplies the attention pool and all heads
for a plain feature extractor.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch.nn as nn


@dataclass
class BackboneSpec:
    """One registry entry per backbone.
      * build(pretrained, **head_kw) -> a full nn.Module (backbone + heads). Each build
        keeps only the kwargs its backbone accepts (feat_stage, opinion_head, ...).
      * input_size / normalize : how the loaders size + ImageNet-normalise the input.
    """
    build: Callable[..., nn.Module]
    input_size: int = 224
    normalize: bool = True


BACKBONES: dict = {}

def register_backbone(name: str, spec: BackboneSpec, aliases=()) -> None:
    """Register a backbone (+ optional aliases) so build_model / model_input_spec find it."""
    for n in (name, *aliases):
        BACKBONES[n.lower()] = spec


def model_input_spec(name: str) -> tuple[int, bool]:
    """(input_size, imagenet_normalize) for a registered model name."""
    try:
        spec = BACKBONES[name.lower()]
    except KeyError:
        raise ValueError(f"unknown model '{name}' (registered: {sorted(BACKBONES)})")
    return spec.input_size, spec.normalize


def build_model(name: str, **kw) -> nn.Module:
    """Construct a backbone by registered name - a single registry look-up. `kw`
    (pretrained, feat_stage, opinion_head, defer_head, ...) is forwarded to the
    spec's build, which keeps only the kwargs that backbone accepts."""
    try:
        spec = BACKBONES[name.lower()]
    except KeyError:
        raise ValueError(f"unknown model '{name}' (registered: {sorted(BACKBONES)})")
    pretrained = kw.pop("pretrained", True)
    return spec.build(pretrained, **kw)
