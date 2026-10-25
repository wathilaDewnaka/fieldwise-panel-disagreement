"""Field prioritisation on the external LNDb cohort (no retraining): each
LIDC-IDRI-trained model (3 seeds x 5 folds) scores every LNDb nodule; same
metrics as part A of reanalysis.py, averaged over the 15 models, with a
patient-clustered bootstrap whose draws are shared by every method."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, pickle
from pathlib import Path
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from scipy.stats import rankdata
from backend.notebooks.shared import study_external as EXT
from backend.notebooks.shared import study_results as R

RUNS = ROOT + "/data/runs"
dirs = [f"{RUNS}/lidc_report_weight03"] + [f"{RUNS}/lidc_report_seed{s}" for s in (42, 43, 44)]
M = {}
for d in dirs:
    for a, v in EXT.load(Path(d) / "external_lndb").items():
        for k, r in v.items():
            M.setdefault(a, {}).setdefault(k, r)
keys = sorted(M["H_fld"])
print("models", len(keys), "arms", sorted(M))
METHODS = ("H_fld", "H_disp", "P_emd", "H_fld_mean", "P_emd_mean", "D_reg", "A_cls")
BUD = (0.05, 0.10, 0.20)


def pct(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        return (rankdata(x) - 0.5) / len(x)
    return np.column_stack([(rankdata(x[:, j]) - 0.5) / len(x) for j in range(x.shape[1])])


r0 = M["H_fld"][keys[0]]
D = np.asarray(r0["test_d_profile"], float); N, P = D.shape
ok = np.isfinite(D).all(1)
D = D[ok]; N = len(D)
Dp = pct(D)
hot = (D > 0) & (D >= np.quantile(D, 0.9, axis=0))
cl = np.asarray(r0["test_clusters"])[ok]
groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
rng = np.random.default_rng(11)
BOOT = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(1000)]
jit = np.random.default_rng(3).random((N, P)) * 1e-9


def sig(arm, k):
    r = M[arm][k]
    if r.get("test_panel_disp_fields") is not None:
        return np.asarray(r["test_panel_disp_fields"], float)[ok]
    return np.repeat(np.asarray(r[f"test_{R.ships(arm)}"], float)[ok][:, None], P, 1)


def metrics(Sp, idx):
    d = D[idx]; dp = Dp[idx]; h = hot[idx]
    keep = d.max(1) > 0
    T = dp >= dp.max(1, keepdims=True) - 1e-12
    out = {}
    for m, sp in Sp.items():
        sp = sp[idx]
        Pm = sp >= sp.max(1, keepdims=True) - 1e-12
        hit = (Pm & T).sum(1) / Pm.sum(1)
        v = [hit[keep].mean()]
        sc = (sp + jit[idx]).ravel(); order = np.argsort(-sc)
        for b in BUD:
            flag = np.zeros(sc.size, bool); flag[order[:int(round(b * sc.size))]] = True
            v.append((flag & h.ravel()).sum() / max(h.sum(), 1))
        out[m] = np.array(v)
    out["chance"] = np.array([(T.sum(1) / P)[keep].mean()] + list(BUD))
    return out


pts, drs = [], []
for k in keys:
    S = {a: sig(a, k) for a in ("H_fld", "H_disp", "P_emd", "D_reg", "A_cls")}
    Sp = {m: pct(v) for m, v in S.items()}
    Sp["H_fld_mean"] = np.repeat(pct(Sp["H_fld"].mean(1))[:, None], P, 1)
    Sp["P_emd_mean"] = np.repeat(pct(Sp["P_emd"].mean(1))[:, None], P, 1)
    pts.append(metrics(Sp, np.arange(N)))
    drs.append([metrics(Sp, b) for b in BOOT])
names = METHODS + ("chance",)
pt = {m: np.mean([p[m] for p in pts], 0) for m in names}
dr = {m: np.mean([[x[m] for x in d] for d in drs], 0) for m in names}
print(f"LNDb n={N} nodules, {len(groups)} patients, fields={P}")
for m in names:
    print(f"  {m:11s} " + "  ".join(f"{x:.3f}" for x in pt[m]))
for m in names[1:]:
    diff = dr["H_fld"] - dr[m]; p = pt["H_fld"] - pt[m]
    lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
    print(f"  H_fld-{m:11s} " + "  ".join(f"{a:+.3f} [{b:+.3f},{c:+.3f}]" for a, b, c in zip(p, lo, hi)))
pickle.dump({"pt": pt, "dr": dr, "n": N}, open(Path(__file__).with_name("lndb_prio.pkl"), "wb"))
