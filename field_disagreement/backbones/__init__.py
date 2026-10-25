"""Backbones and heads.

The registered backbones map an input image to a logit, all sharing the attention
pool and the head assembly of `_HeadMixin` (backbones/heads.py):

  * ResNetOnly           - `resnet50`: an ImageNet ResNet-50 truncated after its
                           third stage (1024 x 14 x 14 on 3x224x224 input), attention
                           pooling, and a 512-d projection. The network of the paper.
  * CNNBaseline          - `cnn`: a compact all-conv network (3x64x64, from scratch).
                           It trains in seconds on a CPU and is what the unit tests use.

Plug-in path: `register_backbone(name, adapter_spec(fn))` registers ANY encoder that
emits a spatial feature grid (N, C, h, w); `build_model(name)` then returns a generic
`AdapterBackbone` with the same attention pool and heads.

Heads, all on the same pooled vector z:
  * classifier          - 1 logit -> sigmoid (probability of the positive class).
  * opinion head        - K logits -> softmax = predicted rating histogram.
  * dispersion head     - the panel-disagreement outputs (one per report field, plus
                          the per-field histograms for the field-level arms).
  * deferral head       - 3 logits -> {negative, positive, defer} (not used by the
                          study's arms).

`forward(x)` returns the classification logit; `forward(x, return_all=True)` returns a
dict with the logit plus whichever heads are enabled.
"""
from __future__ import annotations

from .heads import SpatialAttention
from .adapter import AdapterBackbone, adapter_spec
from . import builtin

__all__ = [
    "SpatialAttention",
    "AdapterBackbone",
    "adapter_spec",
    "builtin",
]
