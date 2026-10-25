"""How many distinct disagreement dimensions the profile holds, and whether
the per-field outputs rank the FIELD-SPECIFIC part of each field's disagreement.

1. Labels only: Spearman correlation between fields' disagreement, eigenvalues of
   that matrix, participation ratio, components for 80% / 90% of the variance.
2. Banked test outputs: partial per-field rho = Spearman(signal_j, d_ij) with the
   mean of the OTHER fields' d (ranks) regressed out of both. A signal that only
   tracks the shared level of disagreement scores ~0. Seeds pooled draw by draw,
   patient-clustered bootstrap shared by every arm (as ms_subsets)."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import rankdata
from ms_subsets_w import R, COH, PROC, ARMS, CLAIM, NB, fields
from pathlib import Path
from field_disagreement import splits as S


def zrank(x):
    r = rankdata(x); return (r - r.mean()) / (r.std() + 1e-12)


def resid(y, x):
    X = np.c_[x, np.ones_like(x)]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def partial_field_rho(Sg, D):
    out = []
    for j in range(D.shape[1]):
        others = np.delete(D, j, axis=1).mean(1)
        c = zrank(others)
        a, b = resid(zrank(Sg[:, j]), c), resid(zrank(D[:, j]), c)
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            out.append((a * b).sum() / den)
    return float(np.mean(out))


def per_field(Sg, D):
    return np.array([np.corrcoef(rankdata(Sg[:, j]), rankdata(D[:, j]))[0, 1]
                     if np.ptp(D[:, j]) > 0 else np.nan for j in range(D.shape[1])])


NAMES = {"LIDC": "lidc_report", "Chaksu": "chaksu_report"}
for coh in ("LIDC", "Chaksu"):
    proc = Path(PROC[coh])
    df = S.build_manifest(proc)
    lab = df[df["label"] >= 0]["file"]
    z0 = np.load(proc / lab.iloc[0], allow_pickle=True)
    names = [str(x) for x in z0["d_profile_attributes"]]
    D = np.stack([np.asarray(np.load(proc / f, allow_pickle=True)["d_profile"], float) for f in lab])
    Rk = np.column_stack([rankdata(D[:, j]) for j in range(D.shape[1])])
    C = np.corrcoef(Rk, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    off = C[np.triu_indices_from(C, 1)]
    print(f"\n== {coh}: {len(lab)} labelled items, fields {names}")
    print("  Spearman between fields' disagreement: median %.2f, range %.2f..%.2f"
          % (np.median(off), off.min(), off.max()))
    print("  matrix:\n" + "\n".join("   " + " ".join(f"{v:+.2f}" for v in row) for row in C))
    print("  eigen share:", " ".join(f"{e/ev.sum():.3f}" for e in ev))
    print(f"  first component {100*ev[0]/ev.sum():.1f}%; participation ratio {ev.sum()**2/(ev**2).sum():.2f};"
          f" components for 80%: {int(np.searchsorted(cum, .8))+1}, 90%: {int(np.searchsorted(cum, .9))+1}")

    # 2. partial per-field rho, and per-field rho by field group
    runs = {s: R.load_run(d, s) for s, d in COH[coh].items()}
    rng = np.random.default_rng(0)
    point = {a: [] for a in ARMS}
    draws = {a: np.zeros(NB) for a in ARMS if a != CLAIM}
    pf = {a: [] for a in ARMS}
    folds = sorted(runs[42])
    for k in folds:
        c = runs[42][k][CLAIM]
        Dt = np.asarray(c["test_d_profile"], float)
        cl = np.asarray(c["test_clusters"])
        groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
        boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                 for _ in range(NB)]
        for s in runs:
            res = runs[s][k]
            Fc = fields(res[CLAIM], "test", Dt.shape[1])
            cb = np.array([partial_field_rho(Fc[b], Dt[b]) for b in boots])
            point[CLAIM].append(partial_field_rho(Fc, Dt))
            for a in ARMS:
                Fa = fields(res[a], "test", Dt.shape[1])
                pf[a].append(per_field(Fa, Dt))
                if a == CLAIM:
                    continue
                point[a].append(partial_field_rho(Fa, Dt))
                ab = np.array([partial_field_rho(Fa[b], Dt[b]) for b in boots])
                draws[a] += (cb - ab) / (len(folds) * len(runs))
    print("  partial per-field rho (other fields' mean removed):")
    print("   " + "  ".join(f"{a} {np.mean(point[a]):+.3f}" for a in ARMS))
    for a in draws:
        diff = np.array(point[CLAIM]) - np.array(point[a])
        lo, hi = np.percentile(draws[a], [2.5, 97.5])
        print(f"   H_fld - {a:7s} {diff.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {int((diff > 0).sum())}/{len(diff)}")
    if coh == "Chaksu":
        ratio = [i for i, n in enumerate(names) if "cdr" in n.lower()]
        size = [i for i in range(len(names)) if i not in ratio]
        print(f"  ratio fields {[names[i] for i in ratio]}; size fields {[names[i] for i in size]}")
        for a in ARMS:
            m = np.nanmean(np.stack(pf[a]), axis=0)
            print(f"   {a:7s} ratio {np.nanmean(m[ratio]):+.3f}  size {np.nanmean(m[size]):+.3f}")
