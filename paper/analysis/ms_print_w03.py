"""Print every manuscript number of the one-weight study (ms_w03_*.pkl, boot_*.pkl)."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, pickle, glob
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from pathlib import Path
from backend.notebooks.shared import study_results as R

HERE = Path(__file__).parent
S = {}
for f in glob.glob(str(HERE / "ms_w03_*_*.pkl")):
    S.update(pickle.load(open(f, "rb")))
if (HERE / "ms_w03.pkl").exists():
    S.update(pickle.load(open(HERE / "ms_w03.pkl", "rb")))
ARMS = ["A_cls", "B_kl", "C_emd", "D_reg", "R_rr", "U_edl", "P_emd", "H_fld", "H_fldp", "H_disp"]
RIV = [a for a in ARMS if a != "H_fld"]
SEEDS = (42, 43, 44)
m = lambda v: float(np.nanmean(v))


def seed_pool(key, coh, alpha=0.05, drop=()):
    out = {}
    for r in RIV:
        deltas, boots, won, nf = [], [], 0, 0
        for s in SEEDS:
            per = S[(coh, s)][key]["contrast"].get(r)
            if not per:
                continue
            per = [t for k, t in zip(S[(coh, s)][key]["folds"], per) if (r, s, k) not in drop]
            deltas += [t["delta"] for t in per]
            boots.append(np.mean([t["boot"] for t in per], axis=0))
            won += sum(t["delta"] > 0 for t in per); nf += len(per)
        if not boots:
            continue
        b = np.mean(boots, axis=0)
        lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[r] = (float(np.mean(deltas)), float(lo), float(hi), won, nf)
    return out


def show(P, title):
    print(title)
    for r, (d, lo, hi, w, n) in P.items():
        print(f"  {r:7s} {d:+.3f} [{lo:+.3f},{hi:+.3f}] {w}/{n}")


for coh in ("LIDC", "Chaksu"):
    if not all((coh, s) in S for s in SEEDS):
        print("missing", coh); continue
    X = [S[(coh, s)] for s in SEEDS]
    print(f"\n################ {coh}")
    print(f"{'arm':8s} {'pf_rho':>7s} {'S1':>7s} {'S2':>7s} {'S3':>7s} {'S4':>7s} {'AUC':>7s}")
    for a in ARMS:
        pf = m([m(x["prim"]["rho"][a]) for x in X])
        s1 = m([m(x["rep"]["rho"][a]) for x in X])
        row = [x["tab"]["rows"][a] for x in X]
        print(f"{a:8s} {pf:7.3f} {s1:7.3f} {m([m(r['semi_partial']) for r in row]):7.3f} "
              f"{m([m(r['contested_auroc']) for r in row]):7.3f} {m([m(r['leakage']) for r in row]):7.3f} "
              f"{m([m(r['auc']) for r in row]):7.3f}")
    print("leak base", m([m(x["tab"]["rows"]["A_cls"]["leak_base"]) for x in X]))
    show(seed_pool("prim", coh), "PRIMARY")
    show(seed_pool("prim", coh, alpha=0.05 / 16), "PRIMARY Bonferroni 16")
    show(seed_pool("rep", coh), "S1 report-level")
    print("SENS H_fld", m([m(x["sens_prim"]["rho"]["H_fld"]) for x in X]))
    show(seed_pool("sens_prim", coh), "SENSITIVITY primary")
    fl = [x["floor"] for x in X]
    for s, f in zip(SEEDS, fl):
        print(f"FLOOR s{s} underfit {f['underfit']} drifted {f['drifted']}")
        for a in ("P_emd", "D_reg", "H_disp", "B_kl"):
            print("   ", a, [u for u in f["units"].get(a, []) if u[0] == 0])
    if coh == "Chaksu":
        drop = {(a, s, k) for s, f in zip(SEEDS, fl) for a, k in f["underfit"] + f["drifted"] if k == 0}
        drop |= {(r, s, k) for (a, s, k) in list(drop) for r in RIV}  # remove the fold-seed unit from every contrast? no:
        print("drop (unit-level):", sorted({(a, s, k) for s, f in zip(SEEDS, fl) for a, k in f["underfit"] + f["drifted"] if k == 0}))
    names = R.FIELD_NAMES[len(X[0]["per_field"]["H_fld"])]
    print("PER FIELD")
    print("  " + f"{'field':12s}" + "".join(f"{a:>8s}" for a in ARMS))
    for j, nm in enumerate(names):
        print(f"  {nm[:12]:12s}" + "".join(f"{np.mean([x['per_field'][a][j] for x in X]):+8.3f}" for a in ARMS))
    best = [all(np.mean([x['per_field']['H_fld'][j] for x in X]) >= np.mean([x['per_field'][a][j] for x in X]) - 1e-9
                for a in RIV if a != "H_disp") for j in range(len(names))]
    print("  H_fld best (vs 9-arm set) on", sum(best), "/", len(names), [n for n, b in zip(names, best) if not b])
    print("CENTRED rho_c rho_p auc_c auc_p")
    for a in ARMS:
        rows = [x["centred"][a] for x in X if x["centred"] and a in x["centred"]]
        if rows:
            print(f"  {a:7s} " + " ".join(f"{np.mean([r[k] for r in rows]):.3f}" for k in ("rho_centred", "rho_primary", "auc_centred", "auc_primary")))
    print("GAP", {a: round(m(sum((x['oc'][a]['gap'] for x in X), [])), 3) for a in ARMS})
    COLS = [("auto", 100), ("error", 100), ("missed_per_1000", 1), ("false_fast", 100),
            ("high_neg", 100), ("obvious_neg_auto", 100)]
    pool = R.pool_served_priority([x["sp"] for x in X])
    ref = {k: {o: sum((x["sp"]["claim_v4"][k][o] for x in X), []) for o in X[0]["sp"]["claim_v4"][k]}
           for k in ("sota", "matched", "randveto", "samecut")}
    print("TRIAGE (pooled 15): auto error missed FA disp obv")
    for nm, r in [("standard 5%", ref["sota"]), ("cert 4% no veto", ref["samecut"]),
                  ("cert same auto", ref["matched"]), ("random brake", ref["randveto"]),
                  ("H_fld", pool["claim_v4"]["served"])] + [(a, v["served"]) for a, v in pool["signal"].items()]:
        print(f"  {nm:18s}" + "".join(f"{sc * m(r[o]):7.1f}" for o, sc in COLS))
    print("OWN network")
    for a in ("A_cls", "C_emd", "U_edl", "P_emd", "H_fld"):
        rr = [x["own"][a]["served"] for x in X if a in x["own"]]
        print(f"  {a:7s}" + "".join(f"{sc * m(sum((r[o] for r in rr), [])):7.1f}" for o, sc in COLS))

E = S.get("lndb_pooled")
if E:
    print(f"\nLNDb n {E['n_items']} patients {E['n_patients']}")
    for a in E["rho"]:
        c = E["contrast"].get(a)
        print(f"  {a:7s} {E['rho'][a]:.3f} " + (f"{c['delta']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}] {c['won']}/{c['n_nets']}" if c else "")
              + f" S1 {E['report_level'][a]:.3f} AUC {E['auc'][a]:.3f}")
    names = R.FIELD_NAMES[7]
    for a in ("H_fld", "P_emd", "H_disp"):
        print(f"  {a} per field:", dict(zip(names, np.round(E["per_field"][a], 3))))

# ---- bootstrap triage + AUC
BT = {}
for coh in ("LIDC", "Chaksu"):
    parts = [pickle.load(open(f, "rb")) for f in glob.glob(str(HERE / f"boot_{coh}_*.pkl"))]
    if not parts:
        continue
    st = {"tri": {}, "auc": {}}
    for p in parts:
        st["tri"].update(p["tri"]); st["auc"].update(p["auc"]); st["rows"] = p["rows"]; st["outs"] = p["outs"]
    BT[coh] = st


def boot_diff(cohs, rowA, rowB, out, scale):
    """claim-row minus other-row; draws: mean over folds within seed, then over seeds."""
    seeds_d = []
    for s in SEEDS:
        per = []
        for c in cohs:
            st = BT[c]; ri = st["rows"]; oi = st["outs"].index(out)
            for (ss, k), A in st["tri"].items():
                if ss == s:
                    per.append(A[:, ri.index(rowA), oi] - A[:, ri.index(rowB), oi])
        seeds_d.append(np.nanmean(per, axis=0))
    d = np.mean(seeds_d, axis=0) * scale
    return float(np.nanmean(d)), *np.nanpercentile(d, [2.5, 97.5])


for cohs in (("LIDC",), ("Chaksu",), ("LIDC", "Chaksu")):
    if not all(c in BT for c in cohs) or any(len(BT[c]["tri"]) < 15 for c in cohs):
        print("boot incomplete", cohs, [len(BT[c]["tri"]) for c in cohs if c in BT]); continue
    print(f"\nBOOT TRIAGE {cohs}: H_fld minus each; disputed pts | missed /1000 | auto pts")
    for r in ["A_cls", "A_cls_mc", "B_kl", "C_emd", "D_reg", "R_rr", "U_edl", "P_emd", "H_fldp", "H_disp",
              "matched", "randveto", "sota"]:
        dd = boot_diff(cohs, "H_fld", r, "high_neg", 100)
        mm = boot_diff(cohs, "H_fld", r, "missed_per_1000", 1)
        au = boot_diff(cohs, "H_fld", r, "auto", 100)
        er = boot_diff(cohs, "H_fld", r, "error", 100)
        print(f"  {r:9s} {dd[0]:+5.1f} [{dd[1]:+5.1f},{dd[2]:+5.1f}] | {mm[0]:+5.1f} [{mm[1]:+5.1f},{mm[2]:+5.1f}] "
              f"| {au[0]:+5.1f} [{au[1]:+5.1f},{au[2]:+5.1f}] | err {er[0]:+5.1f} [{er[1]:+5.1f},{er[2]:+5.1f}]")
    if len(cohs) == 1:
        st = BT[cohs[0]]
        print("  boot mean outcomes (sanity):")
        for i, r in enumerate(st["rows"]):
            v = np.nanmean([np.nanmean(A[:, i, :], axis=0) for A in st["tri"].values()], axis=0)
            print(f"    {r:9s}" + "".join(f"{x * (1 if o == 'missed_per_1000' else 100):7.1f}" for x, o in zip(v, st["outs"])))
        print("  AUC: H_fld minus each (bootstrap)")
        for a in ARMS:
            if a == "H_fld":
                continue
            seeds_d = []
            for s in SEEDS:
                per = [D["H_fld"] - D[a] for (ss, k), D in st["auc"].items() if ss == s and a in D]
                seeds_d.append(np.mean(per, axis=0))
            d = np.mean(seeds_d, axis=0)
            pt = m([m(S[(cohs[0], s)]["auc_fold"]["H_fld"]) - m(S[(cohs[0], s)]["auc_fold"][a]) for s in SEEDS]) \
                if all((cohs[0], s) in S for s in SEEDS) else float("nan")
            print(f"    {a:7s} {pt:+.4f} [{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}] bootmean {d.mean():+.4f}")
