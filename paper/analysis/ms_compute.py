"""Every manuscript number, from the six final runs (3 seeds x 2 cohorts) + LNDb."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, pickle
from dataclasses import replace
from pathlib import Path
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from backend.notebooks.shared import study_results as R
from backend.notebooks.shared import study_external as EXT
from backend.notebooks.shared import study_robustness as REV

RUNS = ROOT + "/data/runs"
COH = {
    "LIDC": {s: f"{RUNS}/lidc_report_seed{s}" for s in (42, 43, 44)},
    "Chaksu": {42: f"{RUNS}/chaksu_report_seed42",
               43: f"{RUNS}/chaksu_report_seeds43_44",
               44: f"{RUNS}/chaksu_report_seeds43_44"},
}
OUT = Path(__file__).with_name("ms_compute.pkl")
rule = replace(R.study_rule(), splits=1000)
store = pickle.load(open(OUT, "rb")) if OUT.exists() else {}

only = sys.argv[1:]          # e.g. LIDC 42 (one run of about 20 min); none = all, then LNDb
for coh, seeds in COH.items():
    d0 = R.cohort_d0(seeds[42])
    for s, d in seeds.items():
        if (coh, s) in store or (only and (coh, str(s)) != (only[0], only[1])):
            continue
        res = R.load_run(d, s)
        x = {"d0": d0}
        x["prim"] = R.primary(res)
        x["rep"] = R.report_level(res)
        x["per_field"] = R.per_field(res)
        x["tab"] = R.arm_table(res, d0)
        x["par"] = R.auc_parity(res)
        x["auc_fold"] = {a: [res[k][a]["auc"] for k in sorted(res) if a in res[k]] for a in R.ARMS}
        x["n_test"] = [len(res[k]["H_fld"]["test_disagreement"]) for k in sorted(res)]
        x["n_pos"] = [int(np.sum(res[k]["H_fld"]["test_labels"])) for k in sorted(res)]
        x["oc"] = R.overfit_check(res)
        x["kept"] = {a: [res[k][a]["selection"]["epoch"] for k in sorted(res) if a in res[k]] for a in R.ARMS}
        sres = R.load_run(d, s, "sensitivity")
        x["sens_prim"] = R.primary(sres)
        x["sens_par"] = R.auc_parity(sres)
        x["centred"] = REV.centred(res)
        x["median3"] = REV.median3(res, d0) if coh == "LIDC" else None
        x["sp"] = R.served_priority(res, d0, rule=rule)
        x["tri"] = R.two_sided_triage(res, d0)
        x["own"] = R.served(res, d0, arms=("A_cls", "P_emd", "H_fld", "C_emd", "U_edl"), rule=rule)["own"]
        if coh == "LIDC" and EXT.load(Path(d) / "external_lndb"):   # LNDb is optional
            x["ext"] = EXT.report(d, verbose=False)
        store[(coh, s)] = x
        pickle.dump(store, open(OUT, "wb"))
        print("done", coh, s, flush=True)

# LNDb pooled over the 15 LIDC networks (3 seeds x 5 folds) of each arm
if "lndb_pooled" not in store and not only:
    merged = {}
    for s, d in COH["LIDC"].items():
        for a, v in EXT.load(Path(d) / "external_lndb").items():
            merged.setdefault(a, {}).update(v)
    if merged:
        orig = EXT.load
        EXT.load = lambda ext: merged
        store["lndb_pooled"] = EXT.report(COH["LIDC"][42], verbose=False)
        EXT.load = orig
        pickle.dump(store, open(OUT, "wb"))
    else:
        print("no LNDb results under data/runs: the external-cohort numbers are skipped")
print("ALL DONE")
