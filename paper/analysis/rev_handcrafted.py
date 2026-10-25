"""A simple, non-deep predictor of the per-field disagreement.

Hand-crafted statistics of the input image only (no mask, no reader data, no
network): intensity moments and percentiles of the whole patch and of its
central region, the size of the bright central region, gradient-magnitude and
Laplacian (sharpness) statistics, and a radial intensity profile. One ridge
regression per field, fitted per fold on the TRAIN + VALIDATION items against the
rank-normal target (Eq. 2, fitted on those items), penalty chosen by 5-fold inner
CV on the same items, applied to the fold's test items (centred input).
Scored like every arm: per-field Spearman, mean over fields; H_fld minus the
baseline with the seed-pooled, patient-clustered bootstrap of ms_subsets."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pathlib import Path
from scipy.stats import rankdata, norm
from scipy import ndimage as ndi
from ms_subsets import R, COH, PROC, ARMS, CLAIM, NB, fields, frho
from field_disagreement import splits as S

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = {"LIDC": "configs/folds/folds_lidc.json", "Chaksu": "configs/folds/folds_chaksu.json"}


def feats(img):
    """img: (3, H, W) in [0, 1]."""
    g = img.mean(0)
    H, W = g.shape
    yy, xx = np.mgrid[:H, :W]
    rr = np.hypot(yy - (H - 1) / 2, xx - (W - 1) / 2) / (H / 2)
    cen = g[rr < 0.5]
    out = []
    for x in (g.ravel(), cen):
        out += [x.mean(), x.std(), *np.percentile(x, [5, 25, 50, 75, 95]),
                ((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-9)]
    for c in range(3):                                    # per-channel moments
        out += [img[c].mean(), img[c].std(), img[c][rr < 0.5].mean()]
    gm = np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1))
    lap = ndi.laplace(ndi.gaussian_filter(g, 1.0))
    out += [gm.mean(), gm.std(), gm[rr < 0.5].mean(), np.percentile(gm, 95), lap.var(), np.abs(lap[rr < 0.5]).mean()]
    for t in (np.percentile(g, 75), np.percentile(g, 90)):  # bright central region
        lab, n = ndi.label(g > t)
        c = lab[H // 2, W // 2]
        reg = (lab == c) if c > 0 else np.zeros_like(g, bool)
        area = reg.mean()
        per = (reg ^ ndi.binary_erosion(reg)).sum() / (np.sqrt(reg.sum()) + 1e-9)
        out += [area, per, (g > t)[rr < 0.5].mean()]
    for a, b in ((0, .25), (.25, .5), (.5, .75), (.75, 1.0)):  # radial profile
        m = (rr >= a) & (rr < b)
        out += [g[m].mean(), g[m].std()]
    return np.array(out, float)


def rank_normal(train, x):
    """Eq. (2): mid-rank ECDF of the training values, standard-normal quantile."""
    s = np.sort(train)
    lo, hi = np.searchsorted(s, x, "left"), np.searchsorted(s, x, "right")
    return norm.ppf(((lo + hi) / 2 + 0.5) / (len(s) + 1))


def ridge_fit(X, y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1]); A[-1, -1] -= lam   # no penalty on intercept
    return np.linalg.solve(A, X.T @ y)


def ridge_cv(X, y, rng, lams=10.0 ** np.arange(-2, 5)):
    idx = rng.permutation(len(y)); parts = np.array_split(idx, 5)
    best, bl = -np.inf, lams[0]
    for lam in lams:
        sc = []
        for p in parts:
            tr = np.setdiff1d(idx, p)
            w = ridge_fit(X[tr], y[tr], lam)
            sc.append(np.corrcoef(rankdata(X[p] @ w), rankdata(y[p]))[0, 1])
        if np.nanmean(sc) > best:
            best, bl = np.nanmean(sc), lam
    return ridge_fit(X, y, bl), bl


for coh in ("LIDC", "Chaksu"):
    proc = Path(PROC[coh])
    df = S.build_manifest(proc)
    assign = json.loads(Path(FOLDS[coh]).read_text())["assignment"]
    cache = Path(HERE) / f"hc_feats_{coh}.pkl"
    if cache.exists():
        F = pickle.load(open(cache, "rb"))
    else:
        F = {}
        for f in df["file"]:
            z = np.load(proc / f, allow_pickle=True)
            if int(z["label"]) >= 0:
                F[f] = (feats(np.asarray(z["image"], float)), np.asarray(z["d_profile"], float))
        pickle.dump(F, open(cache, "wb"))
    runs = {s: R.load_run(d, s) for s, d in COH[coh].items()}
    rng = np.random.default_rng(0)
    point = {"handcrafted": [], CLAIM: []}
    draws = np.zeros(NB)
    lam_used = []
    pick = {a: [] for a in ARMS}
    for k in range(5):
        sp = S.make_fold(df, assign, k, 5)
        fit = [f for f in sp["train"] + sp["val"] if f in F]
        te = list(runs[42][k][CLAIM]["test_files"])
        Xf = np.stack([F[f][0] for f in fit]); Xt = np.stack([F[f][0] for f in te])
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
        Xf = np.c_[(Xf - mu) / sd, np.ones(len(Xf))]; Xt = np.c_[(Xt - mu) / sd, np.ones(len(Xt))]
        Df = np.stack([F[f][1] for f in fit]); Dt = np.stack([F[f][1] for f in te])
        assert np.allclose(Dt, np.asarray(runs[42][k][CLAIM]["test_d_profile"], float), equal_nan=True)
        Ptest = np.zeros_like(Dt)
        for j in range(Dt.shape[1]):
            ok = np.isfinite(Df[:, j])
            y = rank_normal(Df[ok, j], Df[ok, j])
            w, lam = ridge_cv(Xf[ok], y, rng)
            lam_used.append(lam)
            Ptest[:, j] = Xt @ w
        cl = np.asarray(runs[42][k][CLAIM]["test_clusters"])
        groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
        boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(NB)]
        hb = np.array([frho(Ptest[b], Dt[b]) for b in boots])
        hp = frho(Ptest, Dt)
        for s in runs:
            Fc = fields(runs[s][k][CLAIM], "test", Dt.shape[1])
            cb = np.array([frho(Fc[b], Dt[b]) for b in boots])
            point[CLAIM].append(frho(Fc, Dt)); point["handcrafted"].append(hp)
            draws += (cb - hb) / (5 * len(runs))
            for a in ARMS:
                pick[a].append(frho(fields(runs[s][k][a], "test", Dt.shape[1]), Dt))
        print(f"  {coh} fold {k}: handcrafted {hp:.3f}  H_fld s42 {point[CLAIM][-len(runs)]:.3f}", flush=True)
    d = np.array(point[CLAIM]) - np.array(point["handcrafted"])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"\n== {coh}: {Xf.shape[1]-1} features, ridge penalties used {sorted(set(lam_used))}")
    print(f"  handcrafted per-field rho {np.mean(point['handcrafted']):.3f}; H_fld {np.mean(point[CLAIM]):.3f}")
    print(f"  H_fld - handcrafted {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {int((d > 0).sum())}/{len(d)}")
    hc = np.mean(point["handcrafted"])
    print("  arms: " + "  ".join(f"{a} {np.mean(v):.3f}" for a, v in pick.items()))
    print("  arms below the handcrafted baseline: " + ", ".join(a for a, v in pick.items() if np.mean(v) < hc))
