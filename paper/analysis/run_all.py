"""Recompute every number in the manuscript from the stored per-item test
predictions under data/runs/ (CPU only; no training). See README.md.

    python run_all.py            # everything, in dependency order
    python run_all.py reanalysis # one step by name
"""
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
COHORTS, SEEDS = ("LIDC", "Chaksu"), ("42", "43", "44")
STEPS = [
    ("compute",    [["ms_compute.py", c, s] for c in COHORTS for s in SEEDS] + [["ms_compute.py"]]),
    ("main",       [["ms_w03.py", c, s] for c in COHORTS for s in SEEDS]),
    ("bootstrap",  [["boot_triage.py", c, "2000", "10"] for c in COHORTS]),
    ("lndb",       [["lndb_w03.py"]]),
    ("subsets",    [["ms_subsets.py"], ["ms_subsets_w.py"]]),
    ("baselines",  [["rev_handcrafted.py"], ["rev_fields_w.py"], ["rev_sigma_w.py"], ["rev_embed.py"]]),
    ("reanalysis", [["reanalysis.py", c, p] for c in COHORTS for p in "ABC"]),
    ("workload",   [["review_workload.py"]]),
    ("glaucoma",   [["glaucoma_votes.py"]]),
    ("routing",    [["field_routing.py", c] for c in COHORTS] + [["undecided_routing.py"]]),
    ("external",   [["lndb_prio.py"]]),
    ("reference",  [["anchor.py"]]),
    ("audits",     [["bin_audit.py"], ["clamp_audit.py"]]),
    ("stability",  [["boot_triage.py", c, "500", "50", "0,1,2,3,4", "halv50"] for c in COHORTS]
                   + [["halving_check.py"]]),
    ("print",      [["ms_print_w03.py"], ["extra_w03.py"], ["drop_bkl.py"], ["rean_print.py"]]),
]

# The LNDb outputs are not redistributed (LNDb's licence); without them the two
# external-cohort steps are skipped and everything else runs.
LNDB = any((HERE.parents[1] / "data" / "runs").glob("**/external_lndb/checkpoints/*.result.json"))
only = set(sys.argv[1:])
for name, cmds in STEPS:
    if only and name not in only:
        continue
    if name in ("lndb", "external") and not LNDB:
        print(f"{name:10s} skipped: no LNDb results under data/runs", flush=True)
        continue
    for cmd in cmds:
        t0 = time.time()
        log = HERE / ("log_" + "_".join(Path(cmd[0]).stem.split() + cmd[1:]) + ".txt")
        with open(log, "w", encoding="utf8") as fh:
            rc = subprocess.call([sys.executable, str(HERE / cmd[0])] + cmd[1:], stdout=fh,
                                 stderr=subprocess.STDOUT, cwd=HERE)
        print(f"{name:10s} {' '.join(cmd):32s} rc={rc} {time.time() - t0:6.0f}s -> {log.name}", flush=True)
        if rc:
            sys.exit(rc)
