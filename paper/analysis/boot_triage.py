"""Triage and AUC intervals by the primary metric's own bootstrap: patients
resampled within each test fold (draws shared by every signal and every seed),
H calibration/evaluation halvings per draw, folds averaged within a seed, seeds
averaged draw by draw. Writes boot_<cohort>.pkl with, per (seed, fold), the
draw-wise halving-averaged outcome of every rule, and the draw-wise AUCs.

    python boot_triage.py LIDC [B] [H] [folds] [tag]
"""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, pickle, time
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from backend.notebooks.shared import study_results as R
from field_disagreement import metrics as M
from field_disagreement.triage import conformal as C

RUNS = ROOT + "/data/runs"
DIRS = {"LIDC": [f"{RUNS}/lidc_report_weight03"]
                + [f"{RUNS}/lidc_report_seed{s}" for s in (42, 43, 44)],
        "Chaksu": [f"{RUNS}/chaksu_report_weight03",
                   f"{RUNS}/chaksu_report_seed42",
                   f"{RUNS}/chaksu_report_seeds43_44"]}
coh = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
H = int(sys.argv[3]) if len(sys.argv) > 3 else 25
FOLDS = [int(f) for f in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
TAG = sys.argv[5] if len(sys.argv) > 5 else "boot"      # output prefix (stability checks)
SEEDS = (42, 43, 44)
SIGNALS = ("A_cls", "A_cls_mc", "B_kl", "C_emd", "D_reg", "R_rr", "U_edl", "P_emd",
           "H_fldp", "H_disp", "H_fld")
AUC_ARMS = tuple(R.ARMS) + ("H_disp",)
OUTS = ("auto", "error", "missed_per_1000", "false_fast", "high_neg", "obvious_neg_auto")
rule = R.study_rule()
MISS, SOTA, VETO, EXT, NET, OBV = (rule.miss, rule.sota_miss, rule.veto, rule.extreme,
                                   rule.safety_net, rule.obvious)
print("rule", rule, flush=True)


def qcut(x, neg, share):
    return float(np.quantile(x[neg], 1.0 - share)) if neg.any() else np.inf


def safety(conf, p, y, miss):
    pos = y == 1
    cl = np.sort(conf[pos & (p < 0.5)])
    k = int(np.floor(miss * pos.sum() + 1e-9))
    return float(cl[k]) if k < len(cl) else np.inf


def outcomes(auto, p, y, d):
    neg_call = auto & (p < 0.5)
    pos = y == 1
    high = d >= np.quantile(d, 0.9)
    ok = ~pos & (p <= 0.1)
    return (auto.mean(),
            ((p >= 0.5).astype(int) != y)[auto].mean() if auto.any() else np.nan,
            (neg_call & pos).sum() / auto.size * 1000,
            (auto & (p >= 0.5) & ~pos).sum() / max((~pos).sum(), 1),
            (neg_call & high).sum() / max(high.sum(), 1),
            (auto & ok).sum() / ok.sum() if ok.any() else np.nan)


runs = {s: R.load_run(DIRS[coh], s) for s in SEEDS}
folds = [f for f in sorted(runs[42]) if FOLDS is None or f in FOLDS]
store = {"B": B, "H": H, "folds": folds, "seeds": SEEDS, "tri": {}, "auc": {}}
t0 = time.time()
for k in folds:
    ref = runs[42][k]["H_fld"]
    cl = np.asarray(ref["test_clusters"])
    y = np.asarray(ref["test_labels"]).astype(int)
    d = np.asarray(ref["test_disagreement"], float)
    groups = [np.flatnonzero(cl == c) for c in np.unique(cl)]
    G = len(groups)
    rng = np.random.default_rng(1000 + k)         # shared by every seed and signal
    draws = [rng.integers(0, G, G) for _ in range(B)]
    # per seed: classifier columns and every signal's (veto, net)
    S = {}
    for s in SEEDS:
        fr = runs[s][k]
        a = fr["A_cls"]
        assert np.array_equal(np.asarray(a["test_labels"]).astype(int), y)
        p = np.asarray(a["test_probs"], float)
        q = a.get("conf_qhat")
        empty = (C.lac_predict(C.binary_probs(p), q).sum(axis=1) == 0) if q is not None \
            else np.zeros(len(p), bool)
        sig = {}
        for name in SIGNALS:
            if name == "A_cls_mc":
                e = np.asarray(a["test_epistemic"], float)
                sig[name] = (e, e)
                continue
            r = fr[name]
            sig[name] = (np.asarray(r[f"test_{R.ships(name)}"], float),
                         np.asarray(r[f"test_{R.net_col(r, name)}"], float))
        S[s] = dict(p=p, conf=np.asarray(a["test_confidence"], float), empty=empty, sig=sig,
                    auc={n: np.asarray(fr[n]["test_probs"], float) for n in AUC_ARMS if n in fr})
    rows = list(SIGNALS) + ["matched", "randveto", "sota"]
    acc = {s: np.full((B, len(rows), len(OUTS)), np.nan) for s in SEEDS}
    aucs = {s: {n: np.zeros(B) for n in S[s]["auc"]} for s in SEEDS}
    for b, dr in enumerate(draws):
        idx = np.concatenate([groups[i] for i in dr])
        pid = np.concatenate([np.full(len(groups[i]), j) for j, i in enumerate(dr)])
        yb, db = y[idx], d[idx]
        for s in SEEDS:
            for n, pp in S[s]["auc"].items():
                aucs[s][n][b] = M.roc_auc(yb, pp[idx])
        hr = np.random.default_rng((k + 1) * 1_000_003 + b)
        tot = {s: np.zeros((len(rows), len(OUTS))) for s in SEEDS}
        cnt = {s: np.zeros((len(rows), len(OUTS))) for s in SEEDS}
        for h in range(H):
            pick = hr.permutation(G)[:int(round(G / 2))]
            cal = np.isin(pid, pick)
            ev = ~cal
            if (yb[cal] == 1).sum() == 0 or (yb[ev] == 1).sum() == 0:
                continue
            u = hr.random(len(idx)); u2 = hr.random(len(idx))
            for s in SEEDS:
                Z = S[s]
                p, conf, emp = Z["p"][idx], Z["conf"][idx], Z["empty"][idx]
                negc = p[cal] < 0.5
                cut = safety(conf[cal], p[cal], yb[cal], MISS)
                base = conf >= cut
                vet = (p < 0.5) & (p > OBV)
                negall = p < 0.5
                res_rows = []
                claim_refer = None
                for name in SIGNALS:
                    sg, nt = Z["sig"][name]
                    sg, nt = sg[idx], nt[idx]
                    refer = (base | (negall & (nt >= qcut(nt[cal], negc, NET)))
                             | (vet & (sg >= qcut(sg[cal], negc, VETO)))
                             | (negall & (sg >= qcut(sg[cal], negc, EXT))))
                    if name == "H_fld":
                        claim_refer = refer
                    res_rows.append(~(refer | emp))
                rate = float(np.mean(~claim_refer[cal]))
                mcut = float(np.quantile(conf[cal], rate)) if 0 < rate < 1 else cut
                res_rows.append(~((conf >= mcut) | emp))
                rand = (vet & (u < VETO)) | (negall & (u < EXT)) | (negall & (u2 < NET))
                res_rows.append(~(base | emp) & ~rand)
                res_rows.append(~((conf >= safety(conf[cal], p[cal], yb[cal], SOTA)) | emp))
                for i, auto in enumerate(res_rows):
                    o = np.asarray(outcomes(auto[ev], p[ev], yb[ev], db[ev]), float)
                    f = np.isfinite(o)
                    tot[s][i][f] += o[f]; cnt[s][i][f] += 1
        for s in SEEDS:
            with np.errstate(invalid="ignore"):
                acc[s][b] = tot[s] / cnt[s]
        if b % 100 == 0:
            print(f"fold {k} draw {b}  {time.time() - t0:.0f}s", flush=True)
    for s in SEEDS:
        store["tri"][(s, k)] = acc[s]
        store["auc"][(s, k)] = aucs[s]
    store["rows"], store["outs"] = rows, OUTS
    pickle.dump(store, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         f"{TAG}_{coh}_{''.join(map(str, folds))}.pkl"), "wb"))
print("DONE", time.time() - t0, flush=True)
