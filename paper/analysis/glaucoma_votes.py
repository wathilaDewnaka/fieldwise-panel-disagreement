"""Does predicted VCDR disagreement mark split glaucoma decisions? (Chákṣu, no retraining)

Each Chákṣu expert recorded a glaucoma decision (suspect / normal) beside the VCDR.
A panel is SPLIT when its decisions are not unanimous. For every test image this
scores how well each signal separates split from unanimous panels (AUROC), and
how much it still tracks the decision spread once the classifier's uncertainty is
regressed out (partial Spearman, as S2):

    observed VCDR       the experts' recorded VCDR disagreement (reference; no model)
    Field-disp VCDR     Field-disp's VCDR output
    Field-soft VCDR     Field-soft's VCDR output (spread of its VCDR histogram)
    DUP                 DUP's single output (trained on the VCDR disagreement)
    uncertainty         1 - 2|p - 0.5| of the Confidence arm's classifier

Signals are within-fold percentiles; AUROC is pooled over the five test folds per
seed and averaged over seeds. Intervals: 2000 fold-stratified, patient-clustered
bootstrap draws, shared by every signal and seed.

    python glaucoma_votes.py          (writes glaucoma_votes.pkl)
"""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import ms_subsets_w as MS
R = MS.R

HERE = Path(__file__).parent
SEEDS, NB, VCDR = (42, 43, 44), 2000, 1
PROC = Path(ROOT) / MS.PROC["Chaksu"]
SIGNALS = ("observed VCDR", "Field-disp VCDR", "Field-soft VCDR", "DUP", "uncertainty")
runs = {s: R.load_run(MS.COH["Chaksu"][s], s) for s in SEEDS}
folds = sorted(runs[42])


def pct(x):
    return (rankdata(x) - 0.5) / len(x)


def auroc(score, pos):
    r = rankdata(score)
    n1 = pos.sum(); n0 = len(pos) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def partial(x, y, z):
    """Spearman of x and y with z's ranks regressed out of both."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


# per fold: the test items, their votes, and every seed's signals
items = {}
for k in folds:
    h = runs[42][k]["H_fld"]
    files = list(h["test_files"])
    votes = [np.load(PROC / f, allow_pickle=True)["votes"].astype(int) for f in files]
    vsd = np.array([v.std() for v in votes])
    split = np.array([0 < v.sum() < len(v) for v in votes])
    sig = {}
    for s in SEEDS:
        r = runs[s][k]
        assert list(r["H_fld"]["test_files"]) == files
        P = len(r["H_fld"]["test_d_profile"][0])
        p = np.asarray(r["A_cls"]["test_probs"], float)
        sig[s] = {
            "observed VCDR": np.asarray(r["H_fld"]["test_d_profile"], float)[:, VCDR],
            "Field-disp VCDR": R._fields(r["H_fld"], "H_fld", P)[:, VCDR],
            "Field-soft VCDR": R._fields(r["P_emd"], "P_emd", P)[:, VCDR],
            "DUP": R._col(r["D_reg"], R.ships("D_reg")),
            "uncertainty": 1 - 2 * np.abs(p - 0.5),
        }
        sig[s] = {n: pct(v) for n, v in sig[s].items()}
    cl = np.asarray(h["test_clusters"])
    groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    items[k] = dict(split=split, vsd=vsd, sig=sig, groups=groups)


def stats(idx_by_fold):
    """{signal: (auroc, partial)} averaged over seeds, items pooled over folds."""
    out = {}
    split = np.concatenate([items[k]["split"][idx_by_fold[k]] for k in folds])
    vsd = np.concatenate([items[k]["vsd"][idx_by_fold[k]] for k in folds])
    for n in SIGNALS:
        a, pr = [], []
        for s in SEEDS:
            x = np.concatenate([items[k]["sig"][s][n][idx_by_fold[k]] for k in folds])
            u = np.concatenate([items[k]["sig"][s]["uncertainty"][idx_by_fold[k]] for k in folds])
            a.append(auroc(x, split))
            pr.append(np.nan if n == "uncertainty" else partial(x, vsd, u))
        out[n] = (float(np.mean(a)), float(np.mean(pr)))
    return out


full = {k: np.arange(len(items[k]["split"])) for k in folds}
point = stats(full)
rng = np.random.default_rng(2026)
draws = []
for _ in range(NB):
    idx = {}
    for k in folds:
        g = items[k]["groups"]
        idx[k] = np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])
    draws.append(stats(idx))


def ci(f):
    v = np.array([f(d) for d in draws])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


n_split = int(sum(items[k]["split"].sum() for k in folds))
n_all = int(sum(len(items[k]["split"]) for k in folds))
print(f"Chaksu: {n_split} of {n_all} test images have a split glaucoma decision "
      f"({n_split / n_all:.1%})")
print(f"{'signal':18s} {'AUROC split':>12s} {'95% CI':>16s} {'partial rho':>12s} {'95% CI':>16s}")
table = {}
for n in SIGNALS:
    a_ci, p_ci = ci(lambda d: d[n][0]), ci(lambda d: d[n][1])
    table[n] = {"auroc": point[n][0], "auroc_ci": a_ci, "partial": point[n][1], "partial_ci": p_ci}
    ps = "" if n == "uncertainty" else f"{point[n][1]:12.3f} [{p_ci[0]:.3f}, {p_ci[1]:.3f}]"
    print(f"{n:18s} {point[n][0]:12.3f} [{a_ci[0]:.3f}, {a_ci[1]:.3f}] {ps}")
print("\nField-disp VCDR minus each signal (AUROC; partial rho):")
contrasts = {}
for n in SIGNALS:
    if n == "Field-disp VCDR":
        continue
    da = point["Field-disp VCDR"][0] - point[n][0]
    da_ci = ci(lambda d: d["Field-disp VCDR"][0] - d[n][0])
    row = {"auroc": da, "auroc_ci": da_ci}
    line = f"  vs {n:16s} {da:+.3f} [{da_ci[0]:+.3f}, {da_ci[1]:+.3f}]"
    if n != "uncertainty":
        dp = point["Field-disp VCDR"][1] - point[n][1]
        dp_ci = ci(lambda d: d["Field-disp VCDR"][1] - d[n][1])
        row.update(partial=dp, partial_ci=dp_ci)
        line += f"   {dp:+.3f} [{dp_ci[0]:+.3f}, {dp_ci[1]:+.3f}]"
    contrasts[n] = row
    print(line)
# Where each kind of disagreement occurs: split decisions and the predicted /
# recorded VCDR disagreement across fifths of the panel's consensus VCDR.
level = np.concatenate([[float(np.load(PROC / f, allow_pickle=True)["d_consensus"][VCDR])
                         for f in runs[42][k]["H_fld"]["test_files"]] for k in folds])
split_all = np.concatenate([items[k]["split"] for k in folds])
q = np.quantile(level, [0.2, 0.4, 0.6, 0.8])
bins = np.searchsorted(q, level, side="right")
print("\nsplit decisions by fifth of the consensus VCDR (smallest to largest):",
      ", ".join(f"{split_all[bins == b].mean():.0%}" for b in range(5)))
rho_level = {}
for n in ("Field-disp VCDR", "observed VCDR"):
    rho_level[n] = float(np.mean([np.corrcoef(rankdata(np.concatenate(
        [items[k]["sig"][s][n] for k in folds])), rankdata(level))[0, 1] for s in SEEDS]))
    print(f"Spearman({n}, consensus VCDR) = {rho_level[n]:+.2f}  (mean over seeds)")
pickle.dump({"n_split": n_split, "n_all": n_all, "table": table, "contrasts": contrasts,
             "split_by_fifth": [float(split_all[bins == b].mean()) for b in range(5)],
             "rho_level": rho_level},
            open(HERE / "glaucoma_votes.pkl", "wb"))
