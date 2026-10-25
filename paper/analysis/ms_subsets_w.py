"""Reviewer analyses on banked outputs (no retraining):
  * LIDC 4-reader and 3-reader subsets
  * Chaksu per camera
  * LIDC median-3 nodules, seed-pooled CIs
  * Chaksu: continuous vs 0.1-rounded VCDR dispersion (DUP target)
Per-field rho per fold, mean over folds and seeds; H_fld minus each arm with a
fold-stratified patient-clustered bootstrap whose draws are shared by every arm
and every seed (seeds pooled draw by draw)."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, glob, pickle
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from scipy.stats import rankdata
from backend.notebooks.shared import study_results as R

RUNS = ROOT + "/data/runs"
COH = {
    "LIDC": {s: f"{RUNS}/lidc_report_seed{s}" for s in (42, 43, 44)},
    "Chaksu": {42: f"{RUNS}/chaksu_report_seed42",
               43: f"{RUNS}/chaksu_report_seeds43_44",
               44: f"{RUNS}/chaksu_report_seeds43_44"},
}
PROC = {"LIDC": "data/processed/lidc_report", "Chaksu": "data/processed/chaksu_report"}
W = {"LIDC": ROOT + "/data/runs/lidc_report_weight03", "Chaksu": ROOT + "/data/runs/chaksu_report_weight03"}
COH = {c: {s: [W[c], d] for s, d in v.items()} for c, v in COH.items()}
ARMS, CLAIM, NB = R.ARMS, "H_fld", 1000


def frho(S, D):
    """mean over fields of Spearman(S[:, j], D[:, j]); constant columns skipped."""
    rs = rankdata(S, axis=0); rd = rankdata(D, axis=0)
    rs = rs - rs.mean(0); rd = rd - rd.mean(0)
    den = np.sqrt((rs ** 2).sum(0) * (rd ** 2).sum(0))
    ok = den > 0
    return float(np.mean((rs * rd).sum(0)[ok] / den[ok])) if ok.any() else np.nan


def fields(r, view, P):
    k = f"{view}_panel_disp_fields"
    if r.get(k) is not None:
        return np.asarray(r[k], float)
    return np.repeat(np.asarray(r[f"{view}_{R.ships(r['arm'])}"], float)[:, None], P, 1)


def meta(coh, key):
    out = {}
    for f in glob.glob(f"{PROC[coh]}/*.npz"):
        out[os.path.basename(f)] = np.load(f, allow_pickle=True)[key].item() \
            if np.load(f, allow_pickle=True)[key].ndim == 0 else np.load(f, allow_pickle=True)[key]
    return out


def analyse(coh, view, select, label):
    """select(fold_result_of_claim) -> boolean mask over that view's items (or None = all)."""
    runs = {s: R.load_run(d, s) for s, d in COH[coh].items()}
    folds = sorted(runs[42])
    rng = np.random.default_rng(0)
    point = {a: [] for a in ARMS}
    draws = {a: np.zeros(NB) for a in ARMS if a != CLAIM}
    n_items, nf = 0, 0
    for k in folds:
        c = runs[42][k][CLAIM]
        m = select(c)
        D = np.asarray(c[f"{view}_d_profile"], float)
        cl = np.asarray(c[f"{view}_clusters"])
        if m is None:
            m = np.ones(len(D), bool)
        idx_all = np.flatnonzero(m)
        n_items += len(idx_all)
        groups = [idx_all[cl[idx_all] == u] for u in np.unique(cl[idx_all])]
        boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                 for _ in range(NB)]
        nf += 1
        for s in runs:
            res = runs[s][k]
            Fc = fields(res[CLAIM], view, D.shape[1])
            cb = np.array([frho(Fc[b], D[b]) for b in boots])
            point[CLAIM].append(frho(Fc[idx_all], D[idx_all]))
            for a in ARMS:
                if a == CLAIM or a not in res:
                    continue
                Fa = fields(res[a], view, D.shape[1])
                point[a].append(frho(Fa[idx_all], D[idx_all]))
                ab = np.array([frho(Fa[b], D[b]) for b in boots])
                draws[a] += (cb - ab) / (len(folds) * len(runs))
    out = {"label": label, "n_items": n_items,
           "rho": {a: float(np.nanmean(v)) for a, v in point.items()}, "contrast": {}}
    for a in draws:
        diff = np.array(point[CLAIM]) - np.array(point[a])
        won = int((diff > 0).sum())
        out["contrast"][a] = (float(np.nanmean(diff)), *np.nanpercentile(draws[a], [2.5, 97.5]), won, len(diff))
    return out


def show(o):
    print(f"\n== {o['label']}  (n items over folds = {o['n_items']})")
    print("  " + "  ".join(f"{a} {o['rho'][a]:.3f}" for a in ARMS))
    for a, (d, lo, hi, w, n) in o["contrast"].items():
        print(f"   H_fld - {a:7s} {d:+.3f} [{lo:+.3f},{hi:+.3f}] {w}/{n}")


if __name__ == "__main__":
    store = {}
    nread = {k: int(v) for k, v in meta("LIDC", "n_readers").items()}
    dev = {k: str(v) for k, v in meta("Chaksu", "device").items()}

    def by_files(c, fn):
        return np.array([fn(os.path.basename(str(f))) for f in c["test_files"]])

    for n in (4, 3):
        o = analyse("LIDC", "test", lambda c, n=n: by_files(c, lambda f: nread[f] == n), f"LIDC {n} readers")
        show(o); store[f"lidc_r{n}"] = o
    for cam in ("Remidio", "Bosch", "Forus"):
        o = analyse("Chaksu", "test", lambda c, cam=cam: by_files(c, lambda f: dev[f] == cam), f"Chaksu {cam}")
        show(o); store[f"chaksu_{cam}"] = o
    o = analyse("LIDC", "s3", lambda c: None, "LIDC median-3 nodules")
    show(o); store["median3"] = o

    # DUP target: continuous vs 0.1-rounded VCDR dispersion
    raw, rnd = [], []
    for f in glob.glob(f"{PROC['Chaksu']}/*.npz"):
        z = np.load(f, allow_pickle=True)
        v = np.asarray(z["vcdr"], float); v = v[np.isfinite(v)]
        raw.append(v.std()); rnd.append(float(z["d_single"]))
    from scipy.stats import spearmanr, pearsonr
    print(f"\nDUP target Chaksu: Spearman(sd raw VCDR, d_single) = {spearmanr(raw, rnd)[0]:.3f}, "
          f"Pearson {pearsonr(raw, rnd)[0]:.3f}, n {len(raw)}, distinct d_single {len(set(np.round(rnd, 6)))}")
    store["dup_vcdr"] = (spearmanr(raw, rnd)[0], pearsonr(raw, rnd)[0])
    pickle.dump(store, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ms_subsets_w.pkl"), "wb"))
