"""Field prioritisation as review workload: highly disputed fields found per 100 field
reviews (reads rean_<cohort>_A.pkl from reanalysis.py; no retraining).

Recall at budget b (Table of field prioritisation) is the share of the highly
disputed case-field pairs that are flagged when b of all pairs are flagged. Per
100 field reviews this is 100 x recall x (highly disputed pairs) / (flagged pairs),
computed per test fold and pooled as the other field-prioritisation numbers
(mean over folds, then seeds; the same bootstrap draws, scaled by each fold's
factor). A random choice of fields finds 100 x (highly disputed share).

    python review_workload.py
"""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import ms_subsets_w as MS
R = MS.R

H = Path(__file__).parent
SEEDS = (42, 43, 44)
METHODS = ("H_fld", "P_emd", "H_fld_mean")
NAMES = {"H_fld": "Field-disp", "P_emd": "Field-soft", "H_fld_mean": "Field-disp, averaged"}
out = {}
for coh in ("LIDC", "Chaksu"):
    A = pickle.load(open(H / f"rean_{coh}_A.pkl", "rb"))
    runs = R.load_run(MS.COH[coh][42], 42)
    for bi, b in enumerate(A["budgets"]):
        col = bi + 1                                  # column 0 is the top-field hit rate
        fac, rnd = {}, {}
        for k in sorted(runs):
            D = np.asarray(runs[k]["H_fld"]["test_d_profile"], float)
            hot = (D > 0) & (D >= np.quantile(D, 0.9, axis=0))
            fac[k] = 100 * hot.sum() / int(round(b * D.size))
            rnd[k] = 100 * hot.mean()

        def pooled(m, minus=None):
            pts, drs = [], []
            for s in SEEDS:
                ks = [k for (ss, k) in A["res"] if ss == s]
                a = np.mean([A["res"][(s, k)]["point"][m][col] * fac[k] for k in ks])
                d = np.mean([A["res"][(s, k)]["draws"][m][:, col] * fac[k] for k in ks], axis=0)
                if minus is not None:
                    a = np.mean([A["res"][(s, k)]["point"][minus][col] * fac[k] for k in ks]) - a
                    d = np.mean([A["res"][(s, k)]["draws"][minus][:, col] * fac[k] for k in ks], axis=0) - d
                pts.append(a); drs.append(d)
            d = np.mean(drs, axis=0)
            return float(np.mean(pts)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

        row = {m: pooled(m) for m in METHODS}
        row["random"] = float(np.mean(list(rnd.values())))
        row["H_fld-P_emd"] = pooled("P_emd", minus="H_fld")
        row["H_fld-H_fld_mean"] = pooled("H_fld_mean", minus="H_fld")
        out[(coh, b)] = row
        print(f"{coh:7s} budget {b:.0%}: " + ", ".join(f"{NAMES[m]} {row[m][0]:.1f}" for m in METHODS)
              + f", random {row['random']:.1f} per 100 field reviews")
        for c in ("H_fld-P_emd", "H_fld-H_fld_mean"):
            p, lo, hi = row[c]
            print(f"          {c:16s} {p:+.1f} [{lo:+.1f}, {hi:+.1f}]")
pickle.dump(out, open(H / "review_workload.pkl", "wb"))
