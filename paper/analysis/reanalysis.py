"""Reanalyses on the saved test predictions (no retraining).

  A  field prioritisation: does a method's per-field vector find WHICH fields the
     panel disputes? (top-field hit rate; recall of highly disputed fields at
     equal field-review budgets)
  B  budget curves: the same classifier, each signal referring the same share of
     negative calls; no threshold uses an outcome
  C  deployment-style rule: the study's rule with its certainty cut fixed on the
     OTHER four folds (labels never read on the evaluated fold), applied unchanged

    python reanalysis.py LIDC A      (parts A / B / C; writes rean_<coh>_<part>.pkl)
"""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import ms_subsets_w as MS
R = MS.R
from field_disagreement.triage import conformal as C

HERE = Path(__file__).parent
coh, part = sys.argv[1], sys.argv[2]
SEEDS = (42, 43, 44)
NB = 1000 if part == "A" else 2000
runs = {s: R.load_run(MS.COH[coh][s], s) for s in SEEDS}
folds = sorted(runs[42])
t0 = time.time()


def pct(x):
    """within-fold mid-rank percentile of each column."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        return (rankdata(x) - 0.5) / len(x)
    return np.column_stack([(rankdata(x[:, j]) - 0.5) / len(x) for j in range(x.shape[1])])


def col(r, name):
    return np.asarray(r[f"test_{name}"], float)


def boots_for(k):
    cl = np.asarray(runs[42][k]["H_fld"]["test_clusters"])
    groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    rng = np.random.default_rng(7000 + k)
    return [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            for _ in range(NB)]


# --------------------------------------------------------------------------- A
def ridge_preds():
    """Per-fold test predictions of the hand-crafted ridge baseline (seed-free)."""
    with contextlib.redirect_stdout(io.StringIO()):
        src = open(HERE / "rev_handcrafted.py", encoding="utf8").read()
        head = src[:src.index('for coh in ("LIDC", "Chaksu"):')]
        g = {"__file__": str(HERE / "rev_handcrafted.py"), "__name__": "hc"}
        exec(compile(head, "hc", "exec"), g)
    F = pickle.load(open(HERE / f"hc_feats_{coh}.pkl", "rb"))
    df = g["S"].build_manifest(Path(MS.PROC[coh]))
    assign = json.loads(Path(ROOT + "/" + g["FOLDS"][coh]).read_text())["assignment"]
    rng = np.random.default_rng(0)
    out = {}
    for k in folds:
        sp = g["S"].make_fold(df, assign, k, 5)
        fit = [f for f in sp["train"] + sp["val"] if f in F]
        te = list(runs[42][k]["H_fld"]["test_files"])
        Xf = np.stack([F[f][0] for f in fit]); Xt = np.stack([F[f][0] for f in te])
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
        Xf = np.c_[(Xf - mu) / sd, np.ones(len(Xf))]; Xt = np.c_[(Xt - mu) / sd, np.ones(len(Xt))]
        Df = np.stack([F[f][1] for f in fit])
        P = np.zeros((len(te), Df.shape[1]))
        for j in range(Df.shape[1]):
            ok = np.isfinite(Df[:, j])
            w, _ = g["ridge_cv"](Xf[ok], g["rank_normal"](Df[ok, j], Df[ok, j]), rng)
            P[:, j] = Xt @ w
        out[k] = P
    return out


def part_A():
    os.chdir(ROOT)
    ridge = ridge_preds()
    METHODS = ("H_fld", "H_disp", "P_emd", "ridge", "H_fld_mean", "P_emd_mean", "D_reg", "A_cls")
    BUD = (0.05, 0.10, 0.20)
    res = {}
    for k in folds:
        D = np.asarray(runs[42][k]["H_fld"]["test_d_profile"], float)
        N, P = D.shape
        Dp = pct(D)
        hot = (D > 0) & (D >= np.quantile(D, 0.9, axis=0))
        B = boots_for(k)
        jit = np.random.default_rng(k).random((N, P)) * 1e-9
        for s in SEEDS:
            fr = runs[s][k]
            S = {"H_fld": col(fr["H_fld"], "panel_disp_fields"),
                 "H_disp": col(fr["H_disp"], "panel_disp_fields"),
                 "P_emd": col(fr["P_emd"], "panel_disp_fields"),
                 "ridge": ridge[k],
                 "D_reg": np.repeat(col(fr["D_reg"], R.ships("D_reg"))[:, None], P, 1),
                 "A_cls": np.repeat(col(fr["A_cls"], R.ships("A_cls"))[:, None], P, 1)}
            Sp = {m: pct(v) for m, v in S.items()}
            # the same network's per-field outputs averaged into one level: the
            # overall-disagreement information without the field structure
            Sp["H_fld_mean"] = np.repeat(pct(Sp["H_fld"].mean(1))[:, None], P, 1)
            Sp["P_emd_mean"] = np.repeat(pct(Sp["P_emd"].mean(1))[:, None], P, 1)

            def metrics(idx):
                d = D[idx]; dp = Dp[idx]; h = hot[idx]
                keep = d.max(1) > 0
                T = dp >= dp.max(1, keepdims=True) - 1e-12
                out = {}
                for m in METHODS:
                    sp = Sp[m][idx]
                    Pm = sp >= sp.max(1, keepdims=True) - 1e-12
                    hit = (Pm & T).sum(1) / Pm.sum(1)
                    v = [hit[keep].mean()]
                    sc = (sp + jit[idx]).ravel()
                    order = np.argsort(-sc)
                    for b in BUD:
                        flag = np.zeros(sc.size, bool); flag[order[:int(round(b * sc.size))]] = True
                        v.append((flag & h.ravel()).sum() / max(h.sum(), 1))
                    out[m] = np.array(v)
                out["chance"] = np.array([(T.sum(1) / P)[keep].mean()] + list(BUD))
                return out
            full = metrics(np.arange(N))
            draws = [metrics(b) for b in B]
            res[(s, k)] = {"point": full,
                           "draws": {m: np.stack([x[m] for x in draws]) for m in full}}
        print(f"A {coh} fold {k} {time.time() - t0:.0f}s", flush=True)
    return {"res": res, "methods": METHODS, "budgets": BUD}


# --------------------------------------------------------------------------- B
SIG_B = ("A_cls", "A_cls_mc", "D_reg", "P_emd", "H_disp", "H_fld")
BUDGETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def sig_of(fr, name):
    if name == "A_cls_mc":
        return col(fr["A_cls"], "epistemic")
    return col(fr[name], R.ships(name))


def part_B():
    res = {}
    for k in folds:
        a42 = runs[42][k]["H_fld"]
        y = col(a42, "labels").astype(int); d = col(a42, "disagreement")
        B = boots_for(k)
        for s in SEEDS:
            fr = runs[s][k]
            p = col(fr["A_cls"], "probs")
            neg = p < 0.5
            refer = {}
            for nm in SIG_B:
                x = sig_of(fr, nm)
                for b in BUDGETS:
                    cut = np.quantile(x[neg], 1 - b)        # label-free: this fold's negative calls
                    refer[(nm, b)] = neg & (x >= cut)

            def outc(idx):
                hi = d[idx] >= np.quantile(d[idx], 0.9)
                o = {}
                for (nm, b), rf in refer.items():
                    cleared = neg[idx] & ~rf[idx]
                    o[(nm, b)] = np.array([(cleared & hi).sum() / hi.sum(),
                                           (cleared & (y[idx] == 1)).sum() / len(idx) * 1000])
                for b in BUDGETS:     # random referral of the same share: expectation
                    cl = neg[idx] * (1 - b)
                    o[("random", b)] = np.array([(cl * hi).sum() / hi.sum(),
                                                 (cl * (y[idx] == 1)).sum() / len(idx) * 1000])
                return o
            full = outc(np.arange(len(p)))
            draws = [outc(b) for b in B]
            res[(s, k)] = {"point": full, "draws": {key: np.stack([x[key] for x in draws]) for key in full},
                           "neg_share": float(neg.mean())}
        print(f"B {coh} fold {k} {time.time() - t0:.0f}s", flush=True)
    return {"res": res, "signals": SIG_B + ("random",), "budgets": BUDGETS}


# --------------------------------------------------------------------------- C
SIG_C = ("A_cls", "A_cls_mc", "B_kl", "C_emd", "D_reg", "R_rr", "U_edl", "P_emd", "H_fldp",
         "H_disp", "H_fld")


def safety(conf, p, y, miss):
    pos = y == 1
    cl = np.sort(conf[pos & (p < 0.5)])
    kk = int(np.floor(miss * pos.sum() + 1e-9))
    return float(cl[kk]) if kk < len(cl) else np.inf


def part_C():
    rule = R.study_rule()
    res = {}
    for s in SEEDS:
        cols = {}
        for k in folds:
            a = runs[s][k]["A_cls"]
            p = col(a, "probs"); q = a.get("conf_qhat")
            cols[k] = dict(p=p, u=col(a, "confidence"), y=col(a, "labels").astype(int),
                           d=col(a, "disagreement"),
                           empty=(C.lac_predict(C.binary_probs(p), q).sum(1) == 0) if q is not None
                           else np.zeros(len(p), bool))
        for k in folds:
            other = [j for j in folds if j != k]
            cat = lambda key: np.concatenate([cols[j][key] for j in other])
            cut4 = safety(cat("u"), cat("p"), cat("y"), rule.miss)      # never reads fold k's labels
            cut5 = safety(cat("u"), cat("p"), cat("y"), rule.sota_miss)
            Z = cols[k]; p, u, y, d, emp = Z["p"], Z["u"], Z["y"], Z["d"], Z["empty"]
            negc = p < 0.5
            vet = negc & (p > rule.obvious)
            base = u >= cut4
            fr = runs[s][k]
            autos = {}
            for nm in SIG_C:
                if nm == "A_cls_mc":
                    sg = nt = col(fr["A_cls"], "epistemic")
                else:
                    r = fr[nm]
                    sg = col(r, R.ships(nm)); nt = col(r, R.net_col(r, nm))
                q = lambda x, sh: np.quantile(x[negc], 1 - sh)            # label-free
                refer = (base | (negc & (nt >= q(nt, rule.safety_net)))
                         | (vet & (sg >= q(sg, rule.veto))) | (negc & (sg >= q(sg, rule.extreme))))
                autos[nm] = (~(refer | emp)).astype(float)
            rate = autos["H_fld"].mean()
            mcut = np.quantile(u, rate) if 0 < rate < 1 else cut4
            autos["matched"] = (~((u >= mcut) | emp)).astype(float)
            autos["sota"] = (~((u >= cut5) | emp)).astype(float)
            keep_non = (1 - rule.veto) * (1 - rule.extreme) * (1 - rule.safety_net)
            keep_obv = (1 - rule.extreme) * (1 - rule.safety_net)
            pk = np.where(vet, keep_non, np.where(negc, keep_obv, 1.0))
            autos["randveto"] = (~(base | emp)).astype(float) * pk         # expected over the coin

            def outc(idx):
                hi = d[idx] >= np.quantile(d[idx], 0.9)
                pos = y[idx] == 1; pp = p[idx]; n = len(idx)
                o = {}
                for nm, au in autos.items():
                    A = au[idx]
                    clr = A * (pp < 0.5)
                    o[nm] = np.array([A.mean(), (A * ((pp >= 0.5).astype(int) != y[idx])).sum() / max(A.sum(), 1e-9),
                                      (clr * pos).sum() / n * 1000, (clr * hi).sum() / hi.sum()])
                return o
            B = boots_for(k)
            full = outc(np.arange(len(p)))
            draws = [outc(b) for b in B]
            res[(s, k)] = {"point": full, "draws": {m: np.stack([x[m] for x in draws]) for m in full},
                           "cut4": cut4}
        print(f"C {coh} seed {s} {time.time() - t0:.0f}s", flush=True)
    return {"res": res, "outs": ("auto", "error", "missed_per_1000", "disputed_cleared")}


out = {"A": part_A, "B": part_B, "C": part_C}[part]()
pickle.dump(out, open(HERE / f"rean_{coh}_{part}.pkl", "wb"))
print("DONE", time.time() - t0, flush=True)
