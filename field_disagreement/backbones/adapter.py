"""Plug ANY encoder emitting (N, C, h, w) into the framework."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..registry import BackboneSpec
from .heads import _HeadMixin


class AdapterBackbone(nn.Module, _HeadMixin):
    """Generic model: ANY encoder emitting a spatial grid (N,C,h,w) gets the SAME
    attention pool + classifier/opinion/panel heads + `out` dict as the hand-written
    backbones, so every arm runs on it unchanged.
    `encoder_fn(pretrained) -> (encoder module, feature_channels)`."""
    def __init__(self, encoder_fn, pretrained: bool = True,
                 p_drop: float = 0.3, opinion_head: bool = False, defer_head: bool = False,
                 n_ratings: int = 5, dispersion_head: bool = False):
        nn.Module.__init__(self)
        encoder, feat_ch = encoder_fn(pretrained)
        self.encoder = encoder
        self._feat_ch = feat_ch
        self._build_aux_heads(feat_ch, p_drop, opinion_head, defer_head, n_ratings,
                              dispersion_head=dispersion_head)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        feat = self.encoder(x)
        z = self._attn_pool(feat, self.attn(feat))   # attention pool
        if not return_all:
            return self.classifier(self.dropout(z)).squeeze(1)
        return self._assemble(feat, z)

    def param_groups(self, lr_vit: float, lr_head: float):
        """Pretrained encoder trunk at lr_vit; attention pool + all heads at lr_head."""
        enc_ids = {id(p) for p in self.encoder.parameters()}
        enc_ps = [p for p in self.parameters() if id(p) in enc_ids]
        rest = [p for p in self.parameters() if id(p) not in enc_ids]
        return [{"params": enc_ps, "lr": lr_vit},
                {"params": rest, "lr": lr_head}]


def adapter_spec(encoder_fn, input_size: int = 224, normalize: bool = True) -> BackboneSpec:
    """Make a BackboneSpec from a feature-extractor `encoder_fn(pretrained) -> (encoder, C)`.
    The registered model is an AdapterBackbone (attention pool + all heads supplied for
    free) - this is the ~1-line way to plug in your own model."""
    def build(pretrained=True, **kw):
        kw.pop("feat_stage", None)                   # hand-written-backbone-only knob, ignored here
        return AdapterBackbone(encoder_fn, pretrained=pretrained, **kw)
    return BackboneSpec(build, input_size, normalize)
