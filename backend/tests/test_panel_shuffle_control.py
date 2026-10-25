"""The circularity control (`H_fldp`): every item trained on a random other
item's WHOLE panel."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from field_disagreement.config import ExperimentConfig, InvalidConfig, OpinionConfig
from field_disagreement.objectives import Objective


def test_default_is_off():
    assert OpinionConfig().panel_shuffle == "none"


@pytest.mark.parametrize("mode", ["sometimes", "within_class"])
def test_unknown_mode_is_refused(mode):
    with pytest.raises(InvalidConfig):
        OpinionConfig(panel_shuffle=mode)


@pytest.mark.parametrize("target", ["disagreement", "d_single"])
def test_histogram_and_dispersion_move_together(target, monkeypatch):
    """The control must hand each image ONE other panel, not two halves of two."""
    cfg = ExperimentConfig()
    cfg = replace(cfg, opinion=replace(cfg.opinion, panel_shuffle="all", loss="emd"),
                  deferral=replace(cfg.deferral, enabled=True, regress_dispersion=True,
                                   regress_target=target))
    seen = {}
    import field_disagreement.objectives as O
    real_opinion, real_mse = O.triage.opinion_loss, torch.nn.functional.mse_loss

    def spy_opinion(kind, pred, q):
        seen["q"] = q.clone()
        return real_opinion(kind, pred, q)

    def spy_mse(pred, d, *a, **k):
        seen["d"] = d.clone()
        return real_mse(pred, d, *a, **k)
    monkeypatch.setattr(O.triage, "opinion_loss", spy_opinion)
    monkeypatch.setattr(torch.nn.functional, "mse_loss", spy_mse)

    n, k = 12, 5
    q = torch.softmax(torch.randn(n, k), 1)
    d = torch.arange(n, dtype=torch.float32)          # d identifies the item
    q[:, 0] = d / 100.0                                # ...and so does q
    q = q / q.sum(1, keepdim=True)
    batch = {"label": torch.tensor([0, 1] * 6), "opinion_dist": q,
             "disagreement": d, "d_single": d.clone()}
    out = {"logit": torch.zeros(n, requires_grad=True),
           "opinion": torch.zeros(n, k, requires_grad=True),
           "dispersion": torch.zeros(n, requires_grad=True)}
    Objective(cfg, cls_loss=torch.nn.BCEWithLogitsLoss(), tau_d=1.0)(
        None, batch, out, epoch=100, device="cpu", rng=np.random.default_rng(0))
    src_of_q = [int(torch.argmin((q - row).abs().sum(1))) for row in seen["q"]]
    src_of_d = seen["d"].long().tolist()
    assert src_of_q == src_of_d                       # the SAME permutation
    assert src_of_d != list(range(n))                 # and it moved something


def test_the_control_needs_the_epoch_generator():
    cfg = ExperimentConfig()
    cfg = replace(cfg, opinion=replace(cfg.opinion, panel_shuffle="all"))
    n = 4
    batch = {"label": torch.tensor([0, 1, 0, 1]),
             "opinion_dist": torch.full((n, 5), 0.2),
             "disagreement": torch.zeros(n)}
    out = {"logit": torch.zeros(n, requires_grad=True),
           "opinion": torch.zeros(n, 5, requires_grad=True)}
    with pytest.raises(ValueError, match="rng"):
        Objective(cfg, cls_loss=torch.nn.BCEWithLogitsLoss())(
            None, batch, out, epoch=100, device="cpu", rng=None)
