"""Prints the classification-AUC contrasts with their intervals, the median-of-three
robustness rows and the triage point contrasts, from the main pickles."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import io, contextlib
import numpy as np
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import ms_print_w03 as P
S, SEEDS, m = P.S, P.SEEDS, P.m
for coh in ("LIDC", "Chaksu"):
    print("==", coh)
    X = [S[(coh, s)] for s in SEEDS]
    # AUC point diffs + bootstrap CI
    st = P.BT[coh]
    for a in P.RIV:
        pt = np.mean(sum((x["auc_fold"]["H_fld"] for x in X), [])) - np.mean(sum((x["auc_fold"][a] for x in X), []))
        sd = [np.mean([D["H_fld"] - D[a] for (ss, k), D in st["auc"].items() if ss == s], axis=0) for s in SEEDS]
        d = np.mean(sd, axis=0)
        print(f"  AUC {a:7s} {pt:+.4f} [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]")
    if X[0]["median3"]:
        t = [x["median3"]["table"] for x in X]
        for a in ("H_fld", "P_emd", "H_disp"):
            print("  median3", a, "S1 rho", round(np.mean([tt[a]["rho"] for tt in t if a in tt]), 3))
    # triage point contrasts (H_fld - each) from pooled tables
    pool = P.R.pool_served_priority([x["sp"] for x in X])
    ref = {k: {o: sum((x["sp"]["claim_v4"][k][o] for x in X), []) for o in X[0]["sp"]["claim_v4"][k]} for k in ("sota", "matched", "randveto")}
    c = pool["claim_v4"]["served"]
    rows = {**{a: v["served"] for a, v in pool["signal"].items()}, **ref}
    for a, r in rows.items():
        print(f"  TRI {a:9s} disp {100*(m(c['high_neg'])-m(r['high_neg'])):+.1f}  missed {m(c['missed_per_1000'])-m(r['missed_per_1000']):+.1f}  auto {100*(m(c['auto'])-m(r['auto'])):+.1f}  err {100*(m(c['error'])-m(r['error'])):+.1f}  fewer {sum(np.array(c['high_neg'])<np.array(r['high_neg']))}/15")
# pooled both cohorts point
pool2 = P.R.pool_served_priority([S[(c, s)]["sp"] for c in ("LIDC", "Chaksu") for s in SEEDS])
c = pool2["claim_v4"]["served"]
for a, v in pool2["signal"].items():
    r = v["served"]
    print(f"  POOLED {a:9s} disp {100*(m(c['high_neg'])-m(r['high_neg'])):+.1f} missed {m(c['missed_per_1000'])-m(r['missed_per_1000']):+.1f}")
