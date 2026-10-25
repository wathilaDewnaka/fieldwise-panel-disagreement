"""Outcome anchor on LIDC-IDRI (no retraining): nodules of patients with a TCIA
diagnosis (biopsy, resection, 2-year stability or progression) and exactly one
nodule in the cohort. Does predicted disagreement flag nodules on which the
panel's consensus label disagrees with the reference diagnosis?"""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import io, contextlib, pickle, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import rankdata
with contextlib.redirect_stdout(io.StringIO()):
    import ms_subsets_w as MS
R = MS.R
SEEDS = (42, 43, 44)
x = pd.read_excel(ROOT + "/data/raw/lidc-idri/others/tcia-diagnosis-data-2012-04-20.xls")
x.columns = ["pid", "pdx", "pmeth", "site"] + [f"n{i}{t}" for i in range(1, 6) for t in ("dx", "meth")]
dx = dict(zip(x.pid, x.pdx))
runs = {s: R.load_run(MS.COH["LIDC"][s], s) for s in SEEDS}
folds = sorted(runs[42])
MAL = 6  # LIDC_ATTRS index of malignancy


def pct(v):
    return (rankdata(v) - 0.5) / len(v)


def auc(s, e):
    r = rankdata(s); n1 = e.sum(); n0 = len(e) - n1
    return (r[e].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


cnt = Counter(np.concatenate([np.asarray(runs[42][k]["H_fld"]["test_clusters"]) for k in folds]))
SIG = ("H_fld_mal", "H_fld_mean", "P_emd_mal", "P_emd_mean", "D_reg", "A_cls_unc", "panel_d_mal", "panel_d")
per_seed = {}
for s in SEEDS:
    rows = []
    for k in folds:
        fr = runs[s][k]; h = fr["H_fld"]
        cl = np.asarray(h["test_clusters"])
        pick = np.array([cnt[c] == 1 and dx.get(c) in (1, 2, 3) for c in cl])
        F = np.asarray(h["test_panel_disp_fields"], float); Pf = np.asarray(fr["P_emd"]["test_panel_disp_fields"], float)
        Dp = np.asarray(h["test_d_profile"], float)
        sig = {"H_fld_mal": pct(F[:, MAL]), "H_fld_mean": pct(np.column_stack([pct(c) for c in F.T]).mean(1)),
               "P_emd_mal": pct(Pf[:, MAL]), "P_emd_mean": pct(np.column_stack([pct(c) for c in Pf.T]).mean(1)),
               "D_reg": pct(np.asarray(fr["D_reg"][f"test_{R.ships('D_reg')}"], float)),
               "A_cls_unc": pct(np.asarray(fr["A_cls"]["test_confidence"], float)),
               "panel_d_mal": pct(Dp[:, MAL]), "panel_d": pct(np.asarray(h["test_disagreement"], float))}
        y = np.array([dx[c] in (2, 3) if p else False for c, p in zip(cl, pick)])
        for i in np.flatnonzero(pick):
            rows.append((cl[i], int(y[i]), int(h["test_labels"][i]), float(fr["A_cls"]["test_probs"][i] >= 0.5),
                         *[sig[n][i] for n in SIG]))
    rows.sort(key=lambda r: r[0])
    per_seed[s] = rows
ids = [r[0] for r in per_seed[42]]
A = {s: np.array([r[1:] for r in per_seed[s]], float) for s in SEEDS}
y, lab = A[42][:, 0].astype(bool), A[42][:, 1].astype(bool)
perr = lab != y
print(f"n={len(ids)} malignant={y.sum()} benign={(~y).sum()} panel errors={perr.sum()} "
      f"(false neg {(y & ~lab).sum()}, false pos {(~y & lab).sum()})")
rng = np.random.default_rng(0)
B = [rng.integers(0, len(ids), len(ids)) for _ in range(5000)]


def stat(idx, j, target):
    e = target[idx]
    if e.all() or not e.any():
        return np.nan
    return np.mean([auc(A[s][idx, 3 + j], e) for s in SEEDS])


out = {}
for name, target in (("panel error", perr), ("classifier error", None)):
    print(f"\nAUROC for {name} (mean over seeds; 95% bootstrap over patients)")
    for j, n in enumerate(SIG):
        if target is None:
            vals = []
            for idx in [np.arange(len(ids))] + B:
                v = []
                for s in SEEDS:
                    e = (A[s][idx, 2].astype(bool) != y[idx])
                    v.append(auc(A[s][idx, 3 + j], e) if 0 < e.sum() < len(e) else np.nan)
                vals.append(np.nanmean(v))
            pt, dr = vals[0], np.array(vals[1:])
        else:
            pt = stat(np.arange(len(ids)), j, target); dr = np.array([stat(b, j, target) for b in B])
        lo, hi = np.nanpercentile(dr, [2.5, 97.5])
        out[(name, n)] = (pt, lo, hi)
        print(f"  {n:11s} {pt:.3f} [{lo:.3f}, {hi:.3f}]")
cerr = np.mean([(A[s][:, 2].astype(bool) != y) for s in SEEDS], 0)
print("classifier errors (mean over seeds):", cerr.sum().round(1))
# panel accuracy in the upper vs lower half of H_fld's malignancy-field signal
hi_half = np.mean([A[s][:, 3] for s in SEEDS], 0) >= 0.5
print(f"panel error rate: upper half of predicted malignancy disagreement {perr[hi_half].mean():.3f} "
      f"({perr[hi_half].sum()}/{hi_half.sum()}), lower half {perr[~hi_half].mean():.3f} ({perr[~hi_half].sum()}/{(~hi_half).sum()})")
pickle.dump({"out": out, "n": len(ids), "perr": int(perr.sum())}, open(os.path.join(os.path.dirname(__file__), "anchor.pkl"), "wb"))
