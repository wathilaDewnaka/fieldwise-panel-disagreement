"""Evaluation passes over a loader.

Each runs the model over a split, or reads the split, and collects what the
study reports. Nothing here reads a notebook global: the device and the batch
keys are arguments, so the same call works from a notebook and from a test.
"""
from __future__ import annotations

import numpy as np

from field_disagreement.engine import predict_full as engine_predict_full


# --------------------------------------------------------------------------- #
# prediction with predictive uncertainty
# --------------------------------------------------------------------------- #
def predict_full(model, loader, device, mc_passes: int = 0, seed: int = 0) -> dict:
    """Forward a loader collecting everything the triage evaluation needs.

    The forward pass itself is `field_disagreement.engine.predict_full` — the framework
    now needs it too, because `TriageExperiment.calibrate` fits `q_hat` and the
    deferral policy from exactly these signals. Two copies of "how a model's
    predicted signals are collected" is the drift the packaged contract exists
    to prevent, so this is the framework's pass plus the two ANNOTATION targets
    an evaluation scores against, which the framework never reads.

    `q_true` is `opinion_dist` under the name the metrics use: the panel's own
    histogram, as opposed to `opinion`, the model's prediction of it."""
    out = engine_predict_full(model, loader, device, mc_passes=mc_passes,
                             seed=seed,
                             extra_keys=("opinion_dist", "disagreement"))
    out["q_true"] = out.pop("opinion_dist")
    return out


def panel_stats(loader):
    """Per-item panel histogram + binary label over a loader - no forward pass.

    The opinion BASELINES are fitted on these (training) panels. Without them the
    ablation's opinion columns print "-" for the classification-only arm, and the
    remaining rungs are only ever compared against each other."""
    q, y = [], []
    for batch in loader:
        q.append(batch["opinion_dist"].numpy())
        y.append(batch["label"].numpy())
    if not q:
        return np.zeros((0, 5)), np.zeros(0)
    return np.concatenate(q), np.concatenate(y)
