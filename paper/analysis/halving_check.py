"""Stability of the triage intervals to the number of calibration/evaluation
halvings per bootstrap draw: the study's 2000 draws x 10 halvings
(boot_<coh>_*.pkl) against 500 draws x 50 halvings (halv50_<coh>_*.pkl, from
`python boot_triage.py <coh> 500 50 0,1,2,3,4 halv50`).

    python halving_check.py
"""
import glob, pickle, sys
from pathlib import Path
sys.path.insert(0, Path(__file__).resolve().parents[2].as_posix())
import numpy as np

HERE = Path(__file__).parent
SEEDS = (42, 43, 44)
ROWS = ["A_cls", "A_cls_mc", "B_kl", "C_emd", "D_reg", "R_rr", "U_edl", "P_emd", "H_fldp", "H_disp",
        "matched", "randveto", "sota"]


def load(tag, coh):
    st = {"tri": {}}
    for f in glob.glob(str(HERE / f"{tag}_{coh}_*.pkl")):
        p = pickle.load(open(f, "rb"))
        st["tri"].update(p["tri"]); st["rows"] = p["rows"]; st["outs"] = p["outs"]
    return st


def diff(st, rowB, out, scale):
    ri, oi = st["rows"], st["outs"].index(out)
    d = np.mean([np.nanmean([A[:, ri.index("H_fld"), oi] - A[:, ri.index(rowB), oi]
                             for (s, k), A in st["tri"].items() if s == ss], axis=0) for ss in SEEDS], axis=0) * scale
    return np.array([np.nanmean(d), *np.nanpercentile(d, [2.5, 97.5])])


worst = {"high_neg": 0.0, "missed_per_1000": 0.0}
flips = []
for coh in ("LIDC", "Chaksu"):
    a, b = load("boot", coh), load("halv50", coh)
    print(f"\n{coh}: H_fld minus each | 2000x10 | 500x50  (disputed pts; missed /1000)")
    for r in ROWS:
        for out, sc in (("high_neg", 100), ("missed_per_1000", 1)):
            x, y = diff(a, r, out, sc), diff(b, r, out, sc)
            worst[out] = max(worst[out], np.abs(x[1:] - y[1:]).max())
            if out == "high_neg" and ((x[2] < 0) != (y[2] < 0)):
                flips.append((coh, r, out, x, y))
            if out == "missed_per_1000" and ((x[2] < 5) != (y[2] < 5)):
                flips.append((coh, r, out, x, y))
            print(f"  {r:9s} {out:16s} {x[0]:+6.2f} [{x[1]:+6.2f},{x[2]:+6.2f}] | {y[0]:+6.2f} [{y[1]:+6.2f},{y[2]:+6.2f}]")
print("\nlargest change of an interval bound:", {k: round(v, 2) for k, v in worst.items()})
print("conclusions that change (disputed: upper bound < 0; missed: upper bound < 5):", len(flips))
for f in flips:
    print("  ", f[0], f[1], f[2], np.round(f[3], 2), np.round(f[4], 2))
