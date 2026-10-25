"""Field scales sigma_j and the contested cut d0 from TRAINING items only.
Labels only (no network). For each test fold: sigma_j' = s.d. of field j over the
fold's training items (d_profile is already d/sigma_j, so a constant per field),
d' = mean_j d_ij / sigma_j', d0' = 80th percentile of d' over training items.
Reports rank agreement of d and d' on test items, the share of test items whose
contested label flips, and S1/S3 for every arm under both definitions."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, json
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from field_disagreement.metrics.classification import roc_auc as roc_auc_score
from field_disagreement import splits as S
from backend.notebooks.shared import study_results as R

RUNS = ROOT + "/data/runs"
COH = {
    "LIDC": ("lidc_report", "configs/folds/folds_lidc.json", 1.8174,
             {s: f"{RUNS}/lidc_report_seed{s}" for s in (42, 43, 44)}),
    "Chaksu": ("chaksu_report", "configs/folds/folds_chaksu.json", 2.3349,
               {42: f"{RUNS}/chaksu_report_seed42",
                43: f"{RUNS}/chaksu_report_seeds43_44",
                44: f"{RUNS}/chaksu_report_seeds43_44"}),
}

W = {"LIDC": ROOT + "/data/runs/lidc_report_weight03", "Chaksu": ROOT + "/data/runs/chaksu_report_weight03"}
COH = {c: (a, b, d, {s: [W[c], x] for s, x in r.items()}) for c, (a, b, d, r) in COH.items()}
for coh, (name, folds_file, d0, runs_dirs) in COH.items():
    proc = Path(f"data/processed/{name}")
    df = S.build_manifest(proc)
    assign = json.loads(Path(folds_file).read_text())["assignment"]
    P = {f: np.asarray(np.load(proc / f, allow_pickle=True)["d_profile"], float) for f in df["file"]}
    dstore = {f: float(np.load(proc / f, allow_pickle=True)["disagreement"]) for f in df["file"]}
    chk = max(abs(np.nanmean(P[f]) - dstore[f]) for f in df["file"])
    runs = {s: R.load_run(d, s) for s, d in runs_dirs.items()}
    rho, flip, n_cont = [], [], []
    s1 = {}; s3 = {}
    for k in range(5):
        sp = S.make_fold(df, assign, k, 5)
        lab = df.set_index("file")["label"]
        tr = [f for f in sp["train"] if lab[f] >= 0]
        sd = np.nanstd(np.stack([P[f] for f in tr]), axis=0)
        dtr = np.array([np.nanmean(P[f] / sd) for f in tr])
        d0p = np.percentile(dtr, 80)
        te = list(runs[42][k]["H_fld"]["test_files"])
        d = np.array([dstore[f] for f in te])
        dp = np.array([np.nanmean(P[f] / sd) for f in te])
        rho.append(spearmanr(d, dp)[0])
        c, cp = d > d0, dp > d0p
        flip.append(np.mean(c != cp)); n_cont.append((c.sum(), cp.sum()))
        for s in runs:
            for a, r in runs[s][k].items():
                if a not in R.ARMS:
                    continue
                assert list(r["test_files"]) == te
                sig = np.asarray(r[f"test_{R.ships(a)}"], float)
                s1.setdefault(a, []).append((spearmanr(sig, d)[0], spearmanr(sig, dp)[0]))
                s3.setdefault(a, []).append((roc_auc_score(c, sig), roc_auc_score(cp, sig)))
    print(f"\n== {coh}: |mean(d_profile)-disagreement| max {chk:.2e}")
    print(f"  rank agreement d vs train-only d' on test: min {min(rho):.4f} mean {np.mean(rho):.4f}")
    print(f"  contested label flips: mean {100*np.mean(flip):.2f}%  max {100*max(flip):.2f}%  counts {n_cont}")
    for a in R.ARMS:
        x, y = np.array(s1[a]), np.array(s3[a])
        print(f"  {a:7s} S1 {x[:,0].mean():+.3f} -> {x[:,1].mean():+.3f}   S3 {y[:,0].mean():.3f} -> {y[:,1].mean():.3f}")
