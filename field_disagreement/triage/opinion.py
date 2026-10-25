"""Opinion-distribution objectives over the annotator rating histogram.

The panel does not hand us one label, it hands us a histogram q_i over the K
rating categories. Two supervision choices are defensible and they disagree with
each other, which is exactly why the study has to run both:

  * `kl`  treats the K categories as unordered. It is the maximum-likelihood fit
    to the panel's vote and it is what a soft-label classifier does.
  * `emd` treats them as the ordinal scale they actually are: a model that puts
    its mass on rating 4 when the panel said 5 is less wrong than one that put it
    on rating 1. KL cannot see that difference; squared EMD over the CDFs can.

`opinion_loss(kind, ...)` is the switch the ablation flips, so the notebook never
hard-codes one of them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..metrics._util import _EPS


# --------------------------------------------------------------------------- #
# opinion-distribution learning 
# --------------------------------------------------------------------------- #
def opinion_kl_loss(opinion_logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy / KL(q || p_hat) over the 5 rating categories."""
    logp = F.log_softmax(opinion_logits, dim=1)
    logq = torch.log(q.clamp_min(_EPS))
    return (q * (logq - logp)).sum(dim=1).mean()


def opinion_emd_loss(opinion_logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Ordinal-aware alternative: squared Earth-Mover distance over the rating
    CDFs (respects the 1..5 ordering)."""
    p = F.softmax(opinion_logits, dim=1)
    return ((p.cumsum(1) - q.cumsum(1)) ** 2).sum(dim=1).mean()


OPINION_LOSSES = {
    "kl": opinion_kl_loss,
    "emd": opinion_emd_loss,
}


def opinion_loss(kind: str, opinion_logits: torch.Tensor,
                 q: torch.Tensor) -> torch.Tensor:
    """Dispatch to the named opinion objective.

    `kind="none"` returns a zero that is still attached to the graph, so an arm
    with the opinion head switched off runs through the identical training step
    instead of a branched one - the ablation then differs in the objective and
    nothing else."""
    if kind == "none":
        return opinion_logits.sum() * 0.0
    try:
        return OPINION_LOSSES[kind](opinion_logits, q)
    except KeyError:
        raise ValueError(
            f"unknown opinion loss {kind!r}; expected one of "
            f"{sorted(OPINION_LOSSES) + ['none']}") from None


# --------------------------------------------------------------------------- #
# read-outs of a predicted opinion distribution
# --------------------------------------------------------------------------- #
def opinion_entropy(p, normalize: bool = True):
    """Shannon entropy of each predicted rating histogram - the model's estimate
    of how divided the panel would be, i.e. the ALEATORIC term.

    `p` is (N, K) or (K,), numpy or torch; the return matches. With
    `normalize=True` it is divided by log K so it reads as a fraction of maximum
    disagreement and is comparable across panels of different rating scales."""
    if isinstance(p, torch.Tensor):
        p = p.clamp_min(_EPS)
        h = -(p * p.log()).sum(dim=-1)
        return h / torch.log(torch.tensor(float(p.shape[-1]))) if normalize else h
    p = np.clip(np.asarray(p, dtype=float), _EPS, None)
    h = -(p * np.log(p)).sum(axis=-1)
    return h / np.log(p.shape[-1]) if normalize else h


def opinion_spread(p, ratings=None):
    """s.d. of the PREDICTED rating histogram - the panel-risk read-out.

    This is `spectrum.rating_spread` (the s.d. of the panel's actual ratings,
    which is what a cohort's `disagreement` column holds and what the deferral
    head's `tau_d` is a threshold on) evaluated under the predicted histogram
    instead of the observed one. So it is a prediction OF THE SUPERVISED
    QUANTITY, on its own scale and in its own units, and predicted-vs-true can
    be reported as an error in rating points rather than only as a correlation.

    `opinion_entropy` is NOT that quantity and cannot be made into it. Entropy
    is invariant to which categories the mass sits in, so a panel that split
    1/1/5/5 and one that split 2/2/4/4 score identically, while their spreads
    are 2.0 and 1.0. Telling those apart is the whole agreed-vs-divided
    question, which is why the spread is the signal to rank contestedness by
    and the entropy is a shape statistic reported beside it."""
    if isinstance(p, torch.Tensor):
        k = p.shape[-1]
        r = (torch.arange(1, k + 1, dtype=p.dtype, device=p.device)
             if ratings is None
             else torch.as_tensor(ratings, dtype=p.dtype, device=p.device))
        p = p.clamp_min(_EPS)
        p = p / p.sum(dim=-1, keepdim=True)
        mu = (p * r).sum(dim=-1, keepdim=True)
        var = (p * (r - mu) ** 2).sum(dim=-1)
        return var.clamp_min(0.0).sqrt()
    p = np.clip(np.asarray(p, dtype=float), _EPS, None)
    p = p / p.sum(axis=-1, keepdims=True)
    r = (np.arange(1, p.shape[-1] + 1, dtype=float) if ratings is None
         else np.asarray(ratings, dtype=float))
    mu = (p * r).sum(axis=-1, keepdims=True)
    return np.sqrt(np.clip((p * (r - mu) ** 2).sum(axis=-1), 0.0, None))
