"""External test: pools the LNDb scores of the LIDC-IDRI networks (all seeds, the
weight-0.3 retrains first) and builds the LNDb table. Needs the LNDb outputs under
data/runs/*/external_lndb, which are not redistributed."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, pickle
from pathlib import Path
os.chdir(ROOT); sys.path.insert(0, ROOT)
from backend.notebooks.shared import study_external as EXT
from backend.notebooks.shared import study_results as R
RUNS = ROOT + "/data/runs"
dirs = [f"{RUNS}/lidc_report_weight03"] + [f"{RUNS}/lidc_report_seed{s}" for s in (42, 43, 44)]
merged = {}
for d in dirs:
    for a, v in EXT.load(Path(d) / "external_lndb").items():
        for k, r in v.items():
            merged.setdefault(a, {}).setdefault(k, r)
print({a: len(v) for a, v in merged.items()})
EXT.load = lambda ext: merged
out = EXT.report(dirs[1], rivals=tuple(R.RIVALS) + ("H_disp",), verbose=True)
pickle.dump({"lndb_pooled": out}, open(Path(__file__).with_name("ms_w03_lndb_x.pkl"), "wb"))
