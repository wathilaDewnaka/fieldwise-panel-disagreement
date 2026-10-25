"""Field-aware routing, event counts and a three-seed ensemble (no retraining).

  1  matched reading budgets (as part B of reanalysis.py) with the per-field
     outputs combined in three ways: their mean (the signal the study ships),
     their maximum over fields, and the key field alone (LIDC-IDRI malignancy,
     Chaksu VCDR); plus Field-soft's per-field outputs combined the same way, and
     a three-seed ensemble of the Confidence arm (spread of its three seeds'
     probabilities); outcomes also as event COUNTS
  2  per-field Spearman rho of the ensemble spread, against Field-disp

    python field_routing.py LIDC      (writes field_routing_<coh>.pkl)
"""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import ms_subsets_w as MS
R = MS.R

HERE = Path(__file__).parent
coh = sys.argv[1]
SEEDS = (42, 43, 44)
NB = 1000
KEY = {"LIDC": 6, "Chaksu": 1}[coh]          # malignancy / VCDR
BUDGETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
runs = {s: R.load_run(MS.COH[coh][s], s) for s in SEEDS}
folds = sorted(runs[42])
t0 = time.time()


def col(r, name):
    return np.asarray(r[f"test_{name}"], float)


def pct(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        return (rankdata(x) - 0.5) / len(x)
    return np.column_stack([(rankdata(x[:, j]) - 0.5) / len(x) for j in range(x.shape[1])])


def boots_for(k):
    cl = np.asarray(runs[42][k]["H_fld"]["test_clusters"])
    groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    rng = np.random.default_rng(7000 + k)
    return [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            for _ in range(NB)]


def field_rho(sig, D):
    """mean over fields of the Spearman correlation (sig: (n,) or (n, P))."""
    v = []
    for j in range(D.shape[1]):
        ok = np.isfinite(D[:, j])
        s = sig[ok] if sig.ndim == 1 else sig[ok, j]
        a, b = rankdata(s), rankdata(D[ok, j])
        if a.std() > 0 and b.std() > 0:
            v.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(v))


res, rho, counts = {}, {}, {}
for k in folds:
    a42 = runs[42][k]["H_fld"]
    files = list(a42["test_files"])
    for s in SEEDS:                                  # same test items in every seed
        assert list(runs[s][k]["A_cls"]["test_files"]) == files
    y = col(a42, "labels").astype(int); d = col(a42, "disagreement")
    D = np.asarray(a42["test_d_profile"], float)
    P3 = np.stack([col(runs[s][k]["A_cls"], "probs") for s in SEEDS])
    ens = P3.std(0)                                  # three-seed ensemble spread
    B = boots_for(k)
    hi_all = d >= np.quantile(d, 0.9)
    counts[k] = dict(n=len(y), pos=int(y.sum()), hi=int(hi_all.sum()), patients=len(set(a42["test_clusters"])))
    for s in SEEDS:
        fr = runs[s][k]
        p = col(fr["A_cls"], "probs"); neg = p < 0.5
        Fh = pct(col(fr["H_fld"], "panel_disp_fields")); Fp = pct(col(fr["P_emd"], "panel_disp_fields"))
        sig = {"H_fld": col(fr["H_fld"], R.ships("H_fld")), "H_fld_max": Fh.max(1), "H_fld_key": Fh[:, KEY],
               "P_emd": col(fr["P_emd"], R.ships("P_emd")), "P_emd_max": Fp.max(1), "P_emd_key": Fp[:, KEY],
               "A_cls": col(fr["A_cls"], R.ships("A_cls")), "ensemble": ens}
        refer = {}
        for nm, x in sig.items():
            for b in BUDGETS:
                cut = np.quantile(x[neg], 1 - b)     # label-free: this fold's negative calls
                refer[(nm, b)] = neg & (x >= cut)

        def outc(idx):
            hi = d[idx] >= np.quantile(d[idx], 0.9)
            pos = y[idx] == 1
            o = {}
            for (nm, b), rf in refer.items():
                cleared = neg[idx] & ~rf[idx]
                o[(nm, b)] = np.array([(cleared & hi).sum() / hi.sum(), (cleared & pos).sum() / len(idx) * 1000,
                                       (cleared & hi).sum(), (cleared & pos).sum()])
            for b in BUDGETS:                         # random referral of the same share: expectation
                cl = neg[idx] * (1 - b)
                o[("random", b)] = np.array([(cl * hi).sum() / hi.sum(), (cl * pos).sum() / len(idx) * 1000,
                                             (cl * hi).sum(), (cl * pos).sum()])
            return o
        full = outc(np.arange(len(p)))
        draws = [outc(b) for b in B]
        res[(s, k)] = {"point": full, "draws": {key: np.stack([x[key] for x in draws]) for key in full},
                       "neg": int(neg.sum())}
        hf = col(fr["H_fld"], "panel_disp_fields")
        rho[(s, k)] = {"point": np.array([field_rho(hf, D), field_rho(ens, D), field_rho(col(fr["A_cls"], R.ships("A_cls")), D)]),
                       "draws": np.array([[field_rho(hf[b], D[b]), field_rho(ens[b], D[b]),
                                           field_rho(col(fr["A_cls"], R.ships("A_cls"))[b], D[b])] for b in B])}
    print(f"{coh} fold {k} {time.time() - t0:.0f}s", flush=True)

pickle.dump({"res": res, "rho": rho, "counts": counts, "budgets": BUDGETS},
            open(HERE / f"field_routing_{coh}.pkl", "wb"))


def pooled(store, key=None, minus=None):
    pts, drs = [], []
    for s in SEEDS:
        g = lambda k, kind, m: store[(s, k)][kind][m] if m is not None else store[(s, k)][kind]
        a = np.mean([g(k, "point", key) for k in folds], 0); b = np.mean([g(k, "draws", key) for k in folds], 0)
        if minus is not None:
            a = np.mean([g(k, "point", minus) for k in folds], 0) - a
            b = np.mean([g(k, "draws", minus) for k in folds], 0) - b
        pts.append(a); drs.append(b)
    d = np.mean(drs, 0)
    return np.mean(pts, 0), np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)


c = counts
print(f"\n{coh}: per fold n={np.mean([v['n'] for v in c.values()]):.0f} positives={np.mean([v['pos'] for v in c.values()]):.1f} "
      f"highly disputed={np.mean([v['hi'] for v in c.values()]):.1f}; totals n={sum(v['n'] for v in c.values())} "
      f"pos={sum(v['pos'] for v in c.values())} hi={sum(v['hi'] for v in c.values())}; "
      f"negative calls per fold {np.mean([r['neg'] for r in res.values()]):.1f}")
sigs = ("H_fld", "H_fld_max", "H_fld_key", "P_emd", "P_emd_max", "P_emd_key", "A_cls", "ensemble", "random")
print("budget curves: disputed cleared % / missed per 1000 / disputed cleared n per fold / missed n per fold")
for nm in sigs:
    print(f"  {nm:10s} " + " ".join("{:5.1f}/{:4.1f}/{:4.1f}/{:4.2f}".format(100 * p[0], p[1], p[2], p[3])
                                    for p in (pooled(res, (nm, b))[0] for b in BUDGETS)))
for b in (0.3, 0.5):
    print(f" budget {b}: H_fld minus each (disputed pts | missed /1000)")
    for nm in sigs[1:]:
        p, lo, hi = pooled(res, (nm, b), minus=("H_fld", b))
        print(f"   {nm:10s} {100*p[0]:+5.1f} [{100*lo[0]:+5.1f},{100*hi[0]:+5.1f}] | {p[1]:+5.1f} [{lo[1]:+5.1f},{hi[1]:+5.1f}]")
    for nm in ("P_emd_max", "P_emd_key"):
        base = nm.replace("P_emd", "H_fld")
        p, lo, hi = pooled(res, (nm, b), minus=(base, b))
        print(f"   {base}-{nm} {100*p[0]:+5.1f} [{100*lo[0]:+5.1f},{100*hi[0]:+5.1f}] | {p[1]:+5.1f} [{lo[1]:+5.1f},{hi[1]:+5.1f}]")
p, lo, hi = pooled(rho)
print(f"per-field rho: H_fld {p[0]:.3f}  ensemble {p[1]:.3f}  confidence {p[2]:.3f}")
dr = np.mean([np.mean([rho[(s, k)]["draws"][:, 0] - rho[(s, k)]["draws"][:, 1] for k in folds], 0) for s in SEEDS], 0)
print(f"  H_fld - ensemble {p[0] - p[1]:+.3f} [{np.percentile(dr, 2.5):+.3f},{np.percentile(dr, 97.5):+.3f}]")
print("DONE", time.time() - t0)
