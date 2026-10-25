"""Every manuscript number with ONE panel weight (D_reg, P_emd retrained at 0.3,
plus the H_disp ablation), read with the study's other arms. Triage rows of the
unchanged signals are reused from ms_compute.pkl (same results, same seeds ->
identical); D_reg / P_emd / H_disp triage rows are recomputed."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, pickle
from dataclasses import replace
from pathlib import Path
os.chdir(ROOT); sys.path.insert(0, ROOT)
from backend.notebooks.shared import study_results as R
from backend.notebooks.shared import study_external as EXT
from backend.notebooks.shared import study_robustness as REV

RUNS = ROOT + "/data/runs"
W = {"LIDC": f"{RUNS}/lidc_report_weight03",
     "Chaksu": f"{RUNS}/chaksu_report_weight03"}
OLD = {"LIDC": [f"{RUNS}/lidc_report_seed{s}" for s in (42, 43, 44)],
       "Chaksu": [f"{RUNS}/chaksu_report_seed42",
                  f"{RUNS}/chaksu_report_seeds43_44"]}
DIRS = {c: [W[c]] + OLD[c] for c in W}
ARMS9 = tuple(R.ARMS)
ARMSX = ARMS9 + ("H_disp",)
RIVX = tuple(R.RIVALS) + ("H_disp",)
NEW = ("D_reg", "P_emd", "H_disp")
HERE = Path(__file__).parent
only = sys.argv[1:]          # e.g. LIDC 42
OUT = HERE / (f"ms_w03_{only[0]}_{only[1]}.pkl" if only else "ms_w03.pkl")
OLDP = pickle.load(open(HERE / "ms_compute.pkl", "rb"))
rule = replace(R.study_rule(), splits=1000)
store = pickle.load(open(OUT, "rb")) if OUT.exists() else {}

for coh in DIRS:
    d0 = R.cohort_d0(OLD[coh][0])
    for s in (42, 43, 44):
        if (coh, s) in store or (only and (coh != only[0] or s != int(only[1]))):
            continue
        res = R.load_run(DIRS[coh], s)
        old = OLDP[(coh, s)]
        x = {"d0": d0}
        x["prim"] = R.primary(res, rivals=RIVX)
        x["rep"] = R.report_level(res, rivals=RIVX)
        x["per_field"] = R.per_field(res, arms=ARMSX)
        x["tab"] = R.arm_table(res, d0, arms=ARMSX)
        x["auc_fold"] = {a: [res[k][a]["auc"] for k in sorted(res) if a in res[k]] for a in ARMSX}
        x["n_test"] = [len(res[k]["H_fld"]["test_disagreement"]) for k in sorted(res)]
        x["oc"] = R.overfit_check(res, arms=ARMSX)
        x["kept"] = {a: [res[k][a]["selection"]["epoch"] for k in sorted(res) if a in res[k]] for a in ARMSX}
        x["floor"] = R.floor_check(res, 10, 0.9, 30, 5, arms=ARMSX)
        sres = R.load_run(DIRS[coh], s, "sensitivity")
        x["sens_prim"] = R.primary(sres, rivals=RIVX)
        x["centred"] = REV.centred(res, arms=ARMSX)
        x["median3"] = REV.median3(res, d0, arms=ARMSX) if coh == "LIDC" else None
        # triage: unchanged signals reused, changed ones recomputed
        sp = dict(old["sp"]); sp["signal"] = dict(old["sp"]["signal"])
        for a in NEW:
            v = R.served_folds(res, "A_cls", a, d0, rule=rule)
            if v:
                sp["signal"][a] = v
        x["sp"] = sp
        own = dict(old["own"])
        own["P_emd"] = R.served(res, d0, arms=("P_emd",), rule=rule)["own"]["P_emd"]
        x["own"] = own
        store[(coh, s)] = x
        pickle.dump(store, open(OUT, "wb"))
        print("done", coh, s, flush=True)

if not only and "lndb_pooled" not in store:
    merged = {}
    for d in [W["LIDC"]] + OLD["LIDC"]:          # priority: the rerun first
        for a, v in EXT.load(Path(d) / "external_lndb").items():
            for k, r in v.items():
                merged.setdefault(a, {}).setdefault(k, r)
    if merged:
        orig = EXT.load
        EXT.load = lambda ext: merged
        store["lndb_pooled"] = EXT.report(OLD["LIDC"][0], rivals=RIVX, verbose=False)
        EXT.load = orig
        pickle.dump(store, open(OUT, "wb"))
    else:
        print("no LNDb results under data/runs: the external-cohort numbers are skipped")
print("ALL DONE")
