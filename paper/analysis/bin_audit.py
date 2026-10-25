"""Chakshu size-field histogram levels: cohort-wide decile cuts (as used) against
cuts from each fold's training items only. How many test-item expert values
change level, and how much the per-field histogram spread (P_emd's target) moves."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import os, sys, json
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from backend.config import paths
from backend.domains.report_condition import CHAKSU_ATTRS, CHAKSU_RATIOS, chaksu_fields
from field_disagreement import splits as S

P = paths()
folder = Path(P.chaksu_report_processed)
files = sorted(folder.glob("*.npz"))
table = chaksu_fields(Path(P.chaksu_raw))
nr = len(CHAKSU_RATIOS)
vals, uid_of = {}, {}
for f in files:
    uid = str(np.load(f, allow_pickle=True)["series_uid"])
    m = np.full((len(table[uid]), len(CHAKSU_ATTRS)), np.nan)
    for r, row in enumerate(table[uid].values()):
        for j, a in enumerate(CHAKSU_ATTRS):
            try:
                x = float(row.get(a, ""))
            except ValueError:
                continue
            if np.isfinite(x) and x > 0:
                m[r, j] = x
    vals[f.name] = m


def rel(names):
    return np.concatenate([vals[n][:, nr:] / np.nanmean(vals[n][:, nr:], 0) for n in names])


def cuts(names):
    r = rel(names)
    return [np.nanquantile(r[:, j], np.linspace(0.1, 0.9, 9)) for j in range(r.shape[1])]


def hist_sd(name, cs):
    m = vals[name]
    out = []
    for j in range(m.shape[1] - nr):
        col = m[np.isfinite(m[:, nr + j]), nr + j]
        lev = np.searchsorted(cs[j], col / np.mean(col))
        h = np.bincount(lev, minlength=10)[:10] / len(col)
        k = np.arange(10)
        mu = (h * k).sum()
        out.append(np.sqrt((h * (k - mu) ** 2).sum()))
    return np.array(out)


def levels(name, cs):
    m = vals[name]
    out = []
    for j in range(m.shape[1] - nr):
        col = m[np.isfinite(m[:, nr + j]), nr + j]
        out.append(np.searchsorted(cs[j], col / np.mean(col)))
    return out


df = S.build_manifest(folder)
assign = json.loads(Path("configs/folds/folds_chaksu.json").read_text())["assignment"]
labelled = set(df["file"])
full = cuts([n for n in vals])
changed, total, rhos, maxshift = 0, 0, [], 0.0
for k in range(5):
    sp = S.make_fold(df, assign, k, 5)
    tr = [f for f in sp["train"] if f in vals]
    te = [f for f in sp["test"] if f in vals]
    ct = cuts(tr)
    maxshift = max(maxshift, max(np.abs(a - b).max() for a, b in zip(ct, full)))
    for n in te:
        for a, b in zip(levels(n, full), levels(n, ct)):
            changed += int((a != b).sum()); total += len(a)
    A = np.stack([hist_sd(n, full) for n in te]); B = np.stack([hist_sd(n, ct) for n in te])
    rhos.append([spearmanr(A[:, j], B[:, j])[0] for j in range(A.shape[1])])
rhos = np.array(rhos)
print(f"size-field expert values on test items whose level changes: {changed}/{total} = {changed / total:.2%}")
print(f"largest shift of any decile cut (relative size units): {maxshift:.4f}")
print(f"Spearman of test-item histogram SD, cohort cuts vs train-only cuts: min {rhos.min():.4f} mean {rhos.mean():.4f}")
