"""Two further prior-art baselines: the random-reader and evidential arms. The
other arms never touch this module, so their training and scoring are unchanged.

    random reader   train the classifier on ONE reader's call, drawn per item per step
                    from the panel (Jensen et al. 2019; the noisy-label view of the
                    panel). Its confidence is then a direct estimate of the panel's
                    split. A neutral call (LIDC's 3) is a 0.5 target, as `cons_half`.
    evidential      a Beta over the binary call (Sensoy et al. 2018, evidential deep
                    learning): two non-negative evidences, the expected-MSE loss and
                    the annealed KL to the uniform Beta. Routes on its vacuity 2 / S.

Used by the ladder (`study_arms`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from field_disagreement import Objective

__all__ = ["sample_reader_targets", "soft_target_bce", "RandomReaderObjective",
           "EvidentialNet", "edl_loss", "EvidentialObjective", "EDL_ANNEAL_EPOCHS"]

EDL_ANNEAL_EPOCHS = 10       # Sensoy et al. 2018: lambda_t = min(1, t / 10)


# --------------------------------------------------------------------------- #
# random-reader training
# --------------------------------------------------------------------------- #
def sample_reader_targets(reader_targets: torch.Tensor,
                          rng: np.random.Generator) -> torch.Tensor:
    """One valid reader's binary target per row of `reader_targets` (B, R), padded
    with -1. Uniform over that item's readers, drawn from `rng` (the objective's
    epoch-seeded generator), so a re-run draws the same readers."""
    t = reader_targets.detach().cpu().numpy()
    out = np.empty(len(t), dtype=np.float32)
    for i, row in enumerate(t):
        valid = np.flatnonzero(row >= 0)
        if not len(valid):
            raise ValueError("an item with no reader call: random-reader training "
                             "needs `reader_targets` from the processed data")
        out[i] = row[valid[rng.integers(len(valid))]]
    return torch.from_numpy(out)


def soft_target_bce(pos_weight):
    """Class-weighted BCE for targets in {0, 0.5, 1}: a positive call carries the
    split's `pos_weight` (as the consensus arm's loss), a negative or neutral call
    weight 1 (as `cons_half` treats the 3s). Mean over the batch's weights."""
    def loss(logit, t):
        t = t.float()
        per = F.binary_cross_entropy_with_logits(logit.float(), t, reduction="none")
        w = torch.where(t >= 1.0, torch.as_tensor(float(pos_weight), device=per.device),
                        torch.ones_like(per))
        return (per * w).sum() / w.sum()
    return loss


@dataclass
class RandomReaderObjective(Objective):
    """The standard objective with the label replaced, per item per step, by one
    reader's call. Every other term is whatever the arm's config enables."""

    def __call__(self, model, batch, out, epoch: int, device, rng=None):
        if rng is None:
            rng = np.random.default_rng(int(self.cfg.seed) + 1000 * int(epoch))
        t = sample_reader_targets(batch["reader_targets"], rng)
        return super().__call__(model, {**batch, "label": t}, out, epoch, device, rng=rng)


# --------------------------------------------------------------------------- #
# evidential deep learning
# --------------------------------------------------------------------------- #
class EvidentialNet(nn.Module):
    """Wraps a network whose two-wide dispersion head emits the evidences.

    `logit` becomes log(alpha_1 / alpha_0), so the probability every downstream
    reader takes from it is the Beta mean alpha_1 / S; `dispersion` becomes the
    vacuity 2 / S, which the evaluation moves to the `epistemic` column. The
    wrapped net's own logit head is unused. Unknown attributes (param_groups,
    ...) are the wrapped net's."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("net"), name)

    def forward(self, x, return_all: bool = False, **kw):
        out = self.net(x, return_all=True, **kw)
        ev = F.softplus(out["dispersion"].float())                # (N, 2) >= 0
        alpha = ev + 1.0
        S = alpha.sum(dim=1)
        out = dict(out)
        out["logit"] = torch.log(alpha[:, 1]) - torch.log(alpha[:, 0])
        out["evidence"] = ev
        out["dispersion"] = 2.0 / S                               # vacuity in (0, 1]
        return out if return_all else out["logit"]


def edl_loss(evidence: torch.Tensor, y: torch.Tensor, epoch: int, pos_weight) -> torch.Tensor:
    """Sensoy et al. 2018, eq. 5 + the annealed KL regulariser, for K = 2, class
    weighted as the consensus loss (positives carry `pos_weight`)."""
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    Y = F.one_hot(y.long().clamp(min=0), 2).float()
    mse = ((Y - p) ** 2 + p * (1 - p) / (S + 1)).sum(dim=1)
    at = Y + (1 - Y) * alpha                                      # misleading evidence only
    St = at.sum(dim=1)
    kl = (torch.lgamma(St) - torch.lgamma(torch.tensor(2.0, device=at.device))
          - torch.lgamma(at).sum(dim=1)
          + ((at - 1) * (torch.digamma(at) - torch.digamma(St)[:, None])).sum(dim=1))
    lam = min(1.0, float(epoch) / EDL_ANNEAL_EPOCHS)
    w = torch.where(y >= 1, torch.as_tensor(float(pos_weight), device=mse.device),
                    torch.ones_like(mse))
    return ((mse + lam * kl) * w).sum() / w.sum()


@dataclass
class EvidentialObjective(Objective):
    """The evidential loss in place of the classification loss. The evidential arm
    enables no other term, which `ladder_config` guarantees."""
    pos_weight: float = 1.0

    def __call__(self, model, batch, out, epoch: int, device, rng=None):
        y = batch["label"].to(device)
        loss = edl_loss(out["evidence"], y, epoch, self.pos_weight)
        self.last_terms = {"cls": loss.detach()}
        return loss, ""
