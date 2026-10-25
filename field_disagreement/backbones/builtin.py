"""The backbone DEFINITIONS + registrations.

The registry machinery (BackboneSpec, register_backbone, build_model,
model_input_spec) lives in `field_disagreement/registry.py`; this file holds the
concrete backbone CLASSES and REGISTERS them. To add your own backbone without
editing the library:

    def my_encoder(pretrained):                 # must return (encoder, C):
        m = my_backbone(pretrained)             #   encoder(x) -> (N, C, h, w) feature grid
        return m.features, 512                  #   C = channel count of that grid
    register_backbone("mymodel", adapter_spec(my_encoder))

Then `build_model("mymodel")` / `cfg.model = "mymodel"` works everywhere.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from ..registry import BackboneSpec, register_backbone
from .heads import _HeadMixin

try:                                                         # ResNet-50 needs torchvision
    from torchvision.models import resnet50, ResNet50_Weights
    _HAS_TV = True
except Exception:                                            # torchvision optional for the test network
    _HAS_TV = False


def conv_block(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )


class CNNBaseline(nn.Module, _HeadMixin):
    INPUT_SIZE = 64                                          # native 2.5D patch
    NORMALIZE = False                                        # trained from scratch on [0,1]

    def __init__(self, in_ch: int = 3, width: int = 32, p_drop: float = 0.3,
                 opinion_head: bool = False, defer_head: bool = False,
                 n_ratings: int = 5, dispersion_head: bool = False):
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            conv_block(in_ch, width),         nn.MaxPool2d(2),   # 64 -> 32
            conv_block(width, width * 2),     nn.MaxPool2d(2),   # 32 -> 16
            conv_block(width * 2, width * 4), nn.MaxPool2d(2),   # 16 -> 8
            conv_block(width * 4, width * 4),                    # 8
        )
        self._feat_ch = width * 4
        self._build_aux_heads(width * 4, p_drop, opinion_head, defer_head, n_ratings,
                              dispersion_head=dispersion_head)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        feat = self.features(x)
        z = self._attn_pool(feat, self.attn(feat))             # attention pool
        if not return_all:
            return self.classifier(self.dropout(z)).squeeze(1)
        return self._assemble(feat, z)


class ResNetOnly(nn.Module, _HeadMixin):
    """ResNet-50 truncated after a chosen stage, attention pooling, a projection.

    The network of the paper: ImageNet weights (torchvision IMAGENET1K_V2),
    truncated after `feat_stage` ('layer3' gives a 1024 x 14 x 14 map on a
    3x224x224 input), attention pooling over the grid, and a Linear -> GELU ->
    Dropout projection to `fusion_dim` (512) that every head reads. One parameter
    group: the trunk, the projection and the heads all train at `lr_head`.
    """
    INPUT_SIZE = 224
    NORMALIZE = True                                         # ImageNet mean/std (see data.py)

    def __init__(self, in_ch: int = 3, pretrained: bool = True,
                 fusion_dim: int = 512, p_drop: float = 0.3,
                 opinion_head: bool = False, defer_head: bool = False,
                 n_ratings: int = 5, dispersion_head: bool = False,
                 feat_stage: str = "layer3"):
        nn.Module.__init__(self)
        if not _HAS_TV:
            raise RuntimeError("resnet50 needs torchvision (import failed)")
        if in_ch != 3:
            raise ValueError("resnet50 expects 3-channel input (ImageNet stem)")

        rn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        children = list(rn.children())                       # ..., layer3, layer4, avgpool, fc
        stage = str(feat_stage).lower()
        if stage in ("layer4", "stage4", "7x7"):
            self.cnn = nn.Sequential(*children[:-2])         # drop avgpool + fc
            self._feat_ch = 2048
        elif stage in ("layer3", "stage3", "14x14"):
            self.cnn = nn.Sequential(*children[:-3])         # drop avgpool + fc + layer4
            self._feat_ch = 1024
        else:
            raise ValueError(f"feat_stage must be 'layer3' or 'layer4', got {feat_stage!r}")
        self.feat_stage = stage

        # pooled feature -> z
        self.proj = nn.Sequential(
            nn.Linear(self._feat_ch, fusion_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
        )
        self._build_aux_heads(fusion_dim, p_drop, opinion_head, defer_head, n_ratings,
                              dispersion_head=dispersion_head)

    def _encode(self, x: torch.Tensor):
        feat = self.cnn(x)                                   # (N, C, h, w)
        z_c = self._attn_pool(feat, self.attn(feat))         # attention pool
        h = self.proj(z_c)                                   # (N, fusion_dim)
        return feat, h

    def forward(self, x: torch.Tensor, return_all: bool = False):
        feat, h = self._encode(x)
        if not return_all:
            return self.classifier(self.dropout(h)).squeeze(1)
        return self._assemble(feat, h)


# --------------------------------------------------------------------------- #
# The built-in backbones - a thin build() that constructs the class and drops the
# knobs it does not take.
# --------------------------------------------------------------------------- #
def _build_resnet50(pretrained=True, **kw):
    return ResNetOnly(pretrained=pretrained, **kw)        # accepts feat_stage, opinion/defer_head

def _build_cnn(pretrained=True, **kw):
    kw.pop("feat_stage", None)
    return CNNBaseline(**kw)                             # from scratch, ignores pretrained

register_backbone("resnet50", BackboneSpec(_build_resnet50, 224, True),
                  aliases=["resnet", "rn50", "resnet_only"])
register_backbone("cnn", BackboneSpec(_build_cnn, 64, False), aliases=["cnnbaseline"])
