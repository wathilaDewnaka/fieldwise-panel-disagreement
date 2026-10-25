"""Framework design: any encoder emitting an (N, C, h, w) grid inherits the
mechanisms. This is that claim as a test — a hand-rolled encoder the framework
has never seen, registered at run time with no library edit.
"""

import torch
import torch.nn as nn

import field_disagreement as fd


def _tiny_encoder(pretrained: bool = False):
    """encoder_fn(pretrained) -> (module, feature_channels)."""
    enc = nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(8, 12, 3, stride=2, padding=1), nn.ReLU(),
    )
    return enc, 12


def test_arbitrary_encoder_plugs_in():
    fd.register_backbone("tiny_test", fd.adapter_spec(_tiny_encoder, input_size=64,
                                                      normalize=False))
    assert "tiny_test" in fd.BACKBONES
    assert fd.model_input_spec("tiny_test") == (64, False)

    model = fd.build_model("tiny_test", pretrained=False,
                           opinion_head=True, defer_head=True).eval()
    x = torch.rand(2, 3, 64, 64)

    with torch.no_grad():
        out = model(x, return_all=True)

    # the attention pool and every head come for free via AdapterBackbone.
    assert out["logit"].shape == (2,)
    assert out["feat"].shape == (2, 12, 16, 16)
    assert out["opinion"].shape == (2, 5)
    assert out["defer"].shape == (2, 3)


# --------------------------------------------------------------------------- #
