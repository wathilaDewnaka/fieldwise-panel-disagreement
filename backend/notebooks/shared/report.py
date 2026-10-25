"""Pooling per-fold results and formatting them into the notebook tables.

Results share one shape - a list of per-fold dicts per arm, some of whose
values are per-item maps - and these are the operations over it.

What is NOT here is which arms, which metrics and which contrasts the study
reports. That stays in the notebook and the study's report module.

Nothing reads a notebook global; every function takes what it needs.
"""
from __future__ import annotations

import json

from field_disagreement import metrics as M

# --------------------------------------------------------------------------- #
# pooling per-fold results
# --------------------------------------------------------------------------- #
def pool_predictions(per_fold: list[dict], key: str = "test_preds") -> dict:
    """{item id: (prob, label)} pooled over folds."""
    return {str(e["file"]): (float(e["prob"]), int(e["label"]))
            for r in per_fold for e in (r.get(key) or [])}


def flatten(prefix: str, d: dict) -> dict:
    """Nested metric dict -> flat scalar keys. `aggregate_folds` walks the top
    level only, so a metric one level down is invisible to it otherwise."""
    return {f"{prefix}.{k}": float(v) for k, v in d.items()
            if isinstance(v, (int, float))}


# --------------------------------------------------------------------------- #
# formatting one aggregate cell
# --------------------------------------------------------------------------- #
def save_summary(path, model: str, by_arm: dict, agg: dict, arms=None) -> None:
    """Write the per-fold scalars and the aggregate to JSON (per-item maps and
    predictions are left in each fold's own result file).

    `arms` is what the summary records under "arms" - the arm names by default,
    or a richer description such as each arm's active components."""
    slim = {m: [{k: v for k, v in r.items()
                 if not k.endswith("_pernode") and k != "test_preds"}
                for r in res] for m, res in by_arm.items()}
    with open(path, "w") as fh:
        json.dump({"model": model, "arms": list(by_arm) if arms is None else arms,
                   "by_arm": slim, "aggregate": agg}, fh, indent=2)


# --------------------------------------------------------------------------- #
# paired contrasts
# --------------------------------------------------------------------------- #
def auc_parity(by_arm: dict, arm_a: str, arm_b: str, clusters: dict,
               seed: int = 0) -> dict | None:
    """Subject-clustered paired AUC bootstrap on the pooled test predictions."""
    pa, pb = pool_predictions(by_arm[arm_a]), pool_predictions(by_arm[arm_b])
    ids = sorted(set(pa) & set(pb))
    if not ids:
        return None
    return M.paired_auc_bootstrap([pa[f][1] for f in ids], [pa[f][0] for f in ids],
                                  [pb[f][0] for f in ids],
                                  [clusters.get(f, f) for f in ids], seed=seed)
