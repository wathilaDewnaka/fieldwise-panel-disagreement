"""Prints the field-prioritisation, matched-budget and deployment-style numbers
(reanalysis.py parts A, B and C) as quoted in the text."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import pickle, numpy as np
from pathlib import Path
H = Path(__file__).parent
SEEDS = (42, 43, 44)


def pooled(res, key, idx=None, minus=None):
    """point: mean over folds then seeds; draws: same, draw by draw; minus: claim minus key."""
    pts, drs = [], []
    for s in SEEDS:
        ks = [k for (ss, k) in res if ss == s]
        def g(r, m, kind):
            v = r[kind][m]
            return v[..., idx] if idx is not None else v
        a = np.mean([g(res[(s, k)], key, "point") for k in ks], axis=0)
        b = np.mean([g(res[(s, k)], key, "draws") for k in ks], axis=0)
        if minus is not None:
            a = np.mean([g(res[(s, k)], minus, "point") for k in ks], axis=0) - a
            b = np.mean([g(res[(s, k)], minus, "draws") for k in ks], axis=0) - b
        pts.append(a); drs.append(b)
    p = np.mean(pts, axis=0); d = np.mean(drs, axis=0)
    return p, np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0)


for coh in ("LIDC", "Chaksu"):
    A = pickle.load(open(H / f"rean_{coh}_A.pkl", "rb"))
    print(f"\n######## {coh}  A: top-field hit | recall of highly disputed fields at budgets {A['budgets']}")
    for m in A["methods"] + ("chance",):
        p, lo, hi = pooled(A["res"], m)
        print(f"  {m:11s} " + "  ".join(f"{x:.3f}" for x in p))
    for m in A["methods"][1:]:
        p, lo, hi = pooled(A["res"], m, minus="H_fld")
        print(f"  H_fld-{m:11s} " + "  ".join(f"{x:+.3f} [{a:+.3f},{b:+.3f}]" for x, a, b in zip(p, lo, hi)))
    p, lo, hi = pooled(A["res"], "chance", minus="H_fld")
    print(f"  H_fld-chance      " + "  ".join(f"{x:+.3f} [{a:+.3f},{b:+.3f}]" for x, a, b in zip(p, lo, hi)))

    Bq = pickle.load(open(H / f"rean_{coh}_B.pkl", "rb"))
    print(f"  B: budget curves (disputed cleared %, missed /1000) by budget {Bq['budgets']}")
    for sname in Bq["signals"]:
        row = []
        for b in Bq["budgets"]:
            p, _, _ = pooled(Bq["res"], (sname, b))
            row.append(f"{100 * p[0]:5.1f}/{p[1]:5.1f}")
        print(f"    {sname:9s} " + " ".join(row))
    for b in (0.3, 0.5):
        print(f"   budget {b}: H_fld minus each, disputed pts | missed")
        for sname in Bq["signals"]:
            if sname == "H_fld":
                continue
            p, lo, hi = pooled(Bq["res"], (sname, b), minus=("H_fld", b))
            print(f"     {sname:9s} {100*p[0]:+5.1f} [{100*lo[0]:+5.1f},{100*hi[0]:+5.1f}] | {p[1]:+5.1f} [{lo[1]:+5.1f},{hi[1]:+5.1f}]")
    print("   neg share", np.mean([r["neg_share"] for r in Bq["res"].values()]))

    Cq = pickle.load(open(H / f"rean_{coh}_C.pkl", "rb"))
    print(f"  C: deployment-style (auto%, error%, missed/1000, disputed cleared%)")
    for m in Cq["res"][(42, 0)]["point"]:
        p, _, _ = pooled(Cq["res"], m)
        print(f"    {m:9s} {100*p[0]:5.1f} {100*p[1]:5.1f} {p[2]:5.1f} {100*p[3]:5.1f}")
    print("   H_fld minus each: disputed pts | missed | auto pts")
    for m in Cq["res"][(42, 0)]["point"]:
        if m == "H_fld":
            continue
        p, lo, hi = pooled(Cq["res"], m, minus="H_fld")
        print(f"     {m:9s} {100*p[3]:+5.1f} [{100*lo[3]:+5.1f},{100*hi[3]:+5.1f}] | {p[2]:+5.1f} [{lo[2]:+5.1f},{hi[2]:+5.1f}] | {100*p[0]:+5.1f} [{100*lo[0]:+5.1f},{100*hi[0]:+5.1f}]")
