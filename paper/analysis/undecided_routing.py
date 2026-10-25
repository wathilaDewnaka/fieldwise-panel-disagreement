"""Where would the undecided LIDC-IDRI nodules (median malignancy exactly 3) go?
(no retraining)

Each test fold's decided and undecided nodules are pooled, each signal refers the
top b of the pooled cases to a reader (no label is read), and we report the
share of undecided nodules referred, against random referral (= b). The
undecided nodules have no consensus label, so missed positives are undefined;
the question is only whether a signal would send them to a reader.

    python undecided_routing.py      (writes undecided_routing.pkl)
"""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pathlib import Path
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import ms_subsets_w as MS
R = MS.R

HERE = Path(__file__).parent
SEEDS, NB, BUD = (42, 43, 44), 1000, (0.2, 0.3, 0.5)
SIG = {"H_fld": "panel_disp", "P_emd": "panel_spread", "D_reg": "panel_disp", "A_cls": "confidence",
       "A_cls_mc": "epistemic"}
runs = {s: R.load_run(MS.COH["LIDC"][s], s) for s in SEEDS}
folds = sorted(runs[42])
res = {}
for k in folds:
    h = runs[42][k]["H_fld"]
    cl = np.r_[np.asarray(h["test_clusters"]), np.asarray(h["s3_clusters"])]
    und = np.r_[np.zeros(len(h["test_clusters"]), bool), np.ones(len(h["s3_clusters"]), bool)]
    groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    rng = np.random.default_rng(9000 + k)
    B = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(NB)]
    jit = np.random.default_rng(k).random(len(cl)) * 1e-9
    for s in SEEDS:
        sc = {}
        for nm, c in SIG.items():
            arm = "A_cls" if nm.startswith("A_cls") else nm
            r = runs[s][k][arm]
            sc[nm] = np.r_[np.asarray(r[f"test_{c}"], float), np.asarray(r[f"s3_{c}"], float)] + jit

        def outc(idx):
            o = {}
            for nm, x in sc.items():
                xi, ui = x[idx], und[idx]
                o[nm] = np.array([(ui & (xi >= np.quantile(xi, 1 - b))).sum() / ui.sum() for b in BUD])
            return o
        res[(s, k)] = {"point": outc(np.arange(len(cl))), "draws": [outc(b) for b in B],
                       "share": float(und.mean())}
    print("fold", k, flush=True)
pickle.dump(res, open(HERE / "undecided_routing.pkl", "wb"))

print(f"undecided share of the pooled test folds {np.mean([r['share'] for r in res.values()]):.3f}")
print("share of undecided nodules referred at budgets", BUD, "(random = budget)")
for nm in SIG:
    p = np.mean([res[key]["point"][nm] for key in res], 0)
    d = np.mean([np.mean([np.stack([x[nm] for x in res[(s, k)]["draws"]]) for k in folds], 0) for s in SEEDS], 0)
    lo, hi = np.percentile(d, [2.5, 97.5], 0)
    print(f"  {nm:9s} " + "  ".join(f"{a:.3f} [{b:.3f},{c:.3f}]" for a, b, c in zip(p, lo, hi)))
for nm in SIG:
    if nm == "H_fld":
        continue
    dd = np.mean([np.mean([np.stack([x["H_fld"] - x[nm] for x in res[(s, k)]["draws"]]) for k in folds], 0) for s in SEEDS], 0)
    p = np.mean([res[key]["point"]["H_fld"] - res[key]["point"][nm] for key in res], 0)
    lo, hi = np.percentile(dd, [2.5, 97.5], 0)
    print(f"  H_fld-{nm:9s} " + "  ".join(f"{a:+.3f} [{b:+.3f},{c:+.3f}]" for a, b, c in zip(p, lo, hi)))
