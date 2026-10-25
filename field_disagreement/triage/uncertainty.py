"""Predictive uncertainty.

Panel disagreement and predictive uncertainty are two different things and the
distinction is the whole point of this module. `opinion_entropy` says *how
divided the model thinks the readers were*; that is a property of the case and it
is irreducible - the ALEATORIC term. It says nothing about whether the model
itself is out of its depth. For that you have to ask the same input of several
plausible parameterisations of the model and see whether they agree - the
EPISTEMIC term.

Two estimators are provided, and they are the same arithmetic over different
sources of spread:

  * MC dropout       - T stochastic forward passes with the dropout layers left
                       on at eval time. Cheap, needs no extra training, and the
                       head already carries a `Dropout` module.
  * fold ensemble    - the K cross-validation checkpoints. Free (they exist
                       anyway) but each fold saw different training data, so its
                       spread confounds epistemic uncertainty with train-set
                       variation. Report it as an ensemble estimate, not as a
                       deep ensemble, and say so.

Both feed `entropy_decomposition`, the standard mutual-information split:

    total       H(mean_t p_t)      all the uncertainty there is
    aleatoric   mean_t H(p_t)      spread each member predicts INSIDE the panel
    epistemic   total - aleatoric  disagreement BETWEEN the members  (= MI)

With one member the epistemic term is exactly 0. That is honest rather than
missing: a single model has no spread to measure.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# the decomposition (source-agnostic: MC samples or ensemble members)
# --------------------------------------------------------------------------- #
def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def entropy_decomposition(dists, normalize: bool = True) -> dict:
    """Mutual-information split of a stack of categorical predictions.

    `dists` is (T, ..., K): T members (MC samples or ensemble folds) each giving
    a distribution over K categories. The leading axis is the one reduced over.
    Returns total / aleatoric / epistemic arrays shaped like `dists[0, ..., 0]`.

    With `normalize=True` each term is divided by log K, so all three read as
    fractions of the maximum and are comparable across rating scales."""
    d = np.asarray(dists, dtype=float)
    if d.ndim < 2:
        raise ValueError(f"expected (T, ..., K), got shape {d.shape}")
    k = d.shape[-1]
    total = _entropy(d.mean(axis=0))
    aleatoric = _entropy(d).mean(axis=0)
    # Both terms are clamped at 0: the eps inside the log leaves a ~1e-13
    # negative residue on a one-hot distribution, and a "-0.0000 aleatoric" in a
    # clinical read-out is a bug report waiting to happen.
    total = np.maximum(total, 0.0)
    aleatoric = np.maximum(aleatoric, 0.0)
    epistemic = np.maximum(total - aleatoric, 0.0)
    scale = np.log(k) if normalize else 1.0
    return {"total": total / scale,
            "aleatoric": aleatoric / scale,
            "epistemic": epistemic / scale}


def binary_decomposition(probs, normalize: bool = True) -> dict:
    """Same split for the binary malignancy head.

    `probs` is (T, ...) of P(malignant) per member; it is widened to the two-class
    distribution and handed to `entropy_decomposition`, so the binary and the
    rating-histogram uncertainties are computed by identical arithmetic and can
    be printed in the same table."""
    p = np.asarray(probs, dtype=float)
    return entropy_decomposition(np.stack([1.0 - p, p], axis=-1), normalize)


def predictive_variance(probs) -> np.ndarray:
    """Variance of P(malignant) across the T members.

    The blunter epistemic read-out. Kept alongside the MI term because it is what
    most MC-dropout papers report."""
    return np.asarray(probs, dtype=float).var(axis=0)


# --------------------------------------------------------------------------- #
# MC dropout
# --------------------------------------------------------------------------- #
def has_dropout(model: nn.Module) -> bool:
    """Is there any dropout to sample from? A model built with `p_drop=0` gives T
    identical passes and an epistemic term of exactly 0 - a silent no-op the
    caller needs to be able to detect and report."""
    return any(isinstance(m, nn.Dropout) and m.p > 0 for m in model.modules())


@contextmanager
def mc_dropout(model: nn.Module):
    """Eval mode everywhere EXCEPT the dropout layers, which stay stochastic.

    BatchNorm must keep using its running statistics, so a blanket `model.train()`
    is wrong here: it would let the batch composition change the prediction and
    the "uncertainty" would partly be an artefact of how the loader batched the
    data. Restores every module's original mode on exit."""
    was_training = {m: m.training for m in model.modules()}
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    try:
        yield model
    finally:
        for m, mode in was_training.items():
            m.train(mode)


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_passes: int = 20,
                       seed: int | None = None) -> dict:
    """T stochastic passes over one batch.

    Returns the per-pass stacks, NOT their means - the spread is the answer, so
    averaging here would throw away what was asked for::

        probs    (T, N)      P(malignant)
        opinion  (T, N, K)   rating histogram, when the opinion head is present
        defer    (T, N, 3)   deferral distribution, when the defer head is present
        dispersion (T, N) or (T, N, P)   predicted panel dispersion (P = one per
                             report field), when that head is present

    Every head the model exposes must be stacked here, or it never reaches
    `summarise` and is silently unreadable downstream.
    """
    if seed is not None:
        torch.manual_seed(seed)
    probs, opinions, defers, disps = [], [], [], []
    with mc_dropout(model):
        for _ in range(max(1, n_passes)):
            out = model(x, return_all=True)
            probs.append(torch.sigmoid(out["logit"]).float().cpu().numpy())
            if "opinion" in out:
                opinions.append(out["opinion"].softmax(1).float().cpu().numpy())
            if "defer" in out:
                defers.append(out["defer"].softmax(1).float().cpu().numpy())
            if "dispersion" in out:
                disps.append(out["dispersion"].float().cpu().numpy())
    stacks = {"probs": np.stack(probs)}
    if opinions:
        stacks["opinion"] = np.stack(opinions)
    if defers:
        stacks["defer"] = np.stack(defers)
    if disps:
        stacks["dispersion"] = np.stack(disps)
    return stacks


def summarise(stacks: dict, normalize: bool = True) -> dict:
    """Per-case uncertainty summary from the stacks `mc_dropout_predict` returns.

    Everything here is a length-N vector, ready to be stacked into the deferral
    policy's signal matrix::

        prob                 mean P(malignant) over the passes (the prediction)
        binary_total/_epi    MI split of the binary head
        prob_var             variance of P(malignant)
        opinion              mean rating histogram
        opinion_total        H of the mean histogram  (all panel uncertainty)
        opinion_aleatoric    mean H                   (irreducible: panel spread)
        opinion_epistemic    the MI residue           (reducible: model spread)
    """
    out = {"prob": np.asarray(stacks["probs"], dtype=float).mean(axis=0),
           "prob_var": predictive_variance(stacks["probs"])}
    b = binary_decomposition(stacks["probs"], normalize)
    out["binary_total"] = b["total"]
    out["binary_epistemic"] = b["epistemic"]
    if "opinion" in stacks:
        op = np.asarray(stacks["opinion"], dtype=float)
        o = entropy_decomposition(op, normalize)
        out["opinion"] = op.mean(axis=0)
        out["opinion_total"] = o["total"]
        out["opinion_aleatoric"] = o["aleatoric"]
        out["opinion_epistemic"] = o["epistemic"]
    if "defer" in stacks:
        out["defer"] = np.asarray(stacks["defer"], dtype=float).mean(axis=0)
    if "dispersion" in stacks:
        out["dispersion"] = np.asarray(stacks["dispersion"],
                                       dtype=float).mean(axis=0)
    return out
