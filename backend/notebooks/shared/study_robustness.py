"""Robustness analyses of a study run folder: the LIDC median-3 nodules, and the
primary re-read on untranslated test inputs.

    from backend.notebooks.shared import study_robustness as REV
    REV.median3(res, d0)        # res = study_results.load_run(run_dir, seed)
    REV.centred(res)

Nothing here trains. Every number is read from the stored per-item `test_*` /
`s3_*` columns of the result files.

  median3   the LIDC median-3 nodules (held out of training): report-level rho,
            contested AUROC, and per-field rho where the run stored it
  centred   the primary re-read on untranslated test inputs
"""
from __future__ import annotations

import numpy as np

from field_disagreement import metrics as M

from . import study_results as R

CLAIM = R.CLAIM


# --------------------------------------------------------------------------- #
# the median-3 nodules
# --------------------------------------------------------------------------- #
def median3(res: dict, d0: float, arms=None, n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Report-level rho, contested AUROC (d > d0) and, where stored, the per-field
    primary on the score-3 items, which no arm trained or was selected on. The
    contrast is CLAIM minus each arm, patient-clustered bootstrap per fold."""
    arms = arms or R.ARMS
    folds = [k for k in sorted(res) if CLAIM in res[k] and res[k][CLAIM].get("s3_disagreement")]
    if not folds:
        return None
    rng = np.random.default_rng(seed)
    out = {a: {"rho": [], "auroc": [], "field": []} for a in arms}
    diff = {a: {"rho": [], "field": []} for a in arms if a != CLAIM}
    for k in folds:
        d = np.asarray(res[k][CLAIM]["s3_disagreement"], float)
        cl = np.asarray(res[k][CLAIM]["s3_clusters"])
        D = (np.asarray(res[k][CLAIM]["s3_d_profile"], float)
             if res[k][CLAIM].get("s3_d_profile") else None)
        idx = [np.flatnonzero(cl == c) for c in np.unique(cl)]
        boots = [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
                 for _ in range(n_boot)]
        sig, fld = {}, {}
        for a in arms:
            r = res[k].get(a)
            col = R.ships(a)
            if r is None or not r.get(f"s3_{col}"):
                continue
            sig[a] = np.asarray(r[f"s3_{col}"], float)
            out[a]["rho"].append(R.spearman(sig[a], d))
            out[a]["auroc"].append(M.roc_auc((d > d0).astype(int), sig[a])
                                   if 0 < (d > d0).sum() < len(d) else np.nan)
            if D is not None:
                F = (np.asarray(r["s3_panel_disp_fields"], float) if r.get("s3_panel_disp_fields")
                     else np.repeat(sig[a][:, None], D.shape[1], 1))
                fld[a] = F
                out[a]["field"].append(float(np.mean(R.field_rho(F, D))))
        if CLAIM not in sig:
            continue
        cb = np.array([R.spearman(sig[CLAIM][b], d[b]) for b in boots])
        fb = (np.array([R._score(fld[CLAIM], D, b) for b in boots])
              if CLAIM in fld else None)
        for a in diff:
            if a in sig:
                diff[a]["rho"].append(cb - np.array([R.spearman(sig[a][b], d[b]) for b in boots]))
            if fb is not None and a in fld:
                diff[a]["field"].append(fb - np.array([R._score(fld[a], D, b) for b in boots]))
    table = {a: {m: (float(np.nanmean(v)) if v else None) for m, v in o.items()}
             for a, o in out.items() if o["rho"]}
    for a, dd in diff.items():
        for m in ("rho", "field"):
            if len(dd[m]) == len(folds):
                boot = np.mean(dd[m], axis=0)
                table.setdefault(a, {})[f"claim_minus_{m}"] = (
                    float(np.mean([x.mean() for x in dd[m]])),
                    *map(float, np.percentile(boot, [2.5, 97.5])))
    return {"n_items": int(sum(len(res[k][CLAIM]["s3_disagreement"]) for k in folds)),
            "table": table, "per_field_stored": any(o["field"] for o in out.values())}


# --------------------------------------------------------------------------- #
# evaluation geometry: untranslated test inputs
# --------------------------------------------------------------------------- #
def centred(res: dict, arms=None) -> dict | None:
    arms = arms or R.ARMS
    rows = {}
    for a in arms:
        v = []
        for k in sorted(res):
            r = res[k].get(a)
            if r is None:
                continue
            col = R.ships(a)
            if not r.get(f"test_centred_{col}"):
                continue
            D = R._profile(res[k])
            F = (np.asarray(r["test_centred_panel_disp_fields"], float)
                 if r.get("test_centred_panel_disp_fields")
                 else np.repeat(np.asarray(r[f"test_centred_{col}"], float)[:, None],
                                D.shape[1], 1))
            v.append((float(np.mean(R.field_rho(F, D))),
                      float(np.mean(R.field_rho(R._fields(r, a, D.shape[1]), D))),
                      M.roc_auc(np.asarray(r["test_labels"]), np.asarray(r["test_centred_probs"])),
                      r["auc"]))
        if v:
            v = np.asarray(v)
            rows[a] = {"rho_centred": float(v[:, 0].mean()), "rho_primary": float(v[:, 1].mean()),
                       "auc_centred": float(v[:, 2].mean()), "auc_primary": float(v[:, 3].mean())}
    return rows or None
