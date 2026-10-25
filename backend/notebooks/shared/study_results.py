"""results from a finished run: the primary contrast and the secondary tables.

    python -m backend.notebooks.shared.study_results data/runs/<run folder>

Reads each fold's `result.json` (the per-item test columns), so a run can be
re-scored without a GPU, and the notebook and the command line share one
definition. The metrics are declared in
the study protocol (paper, Methods and Appendix); this module only computes them.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from field_disagreement import metrics as M
from field_disagreement import triage
from field_disagreement.triage import conformal as C
from field_disagreement.engine import selection_summary
from field_disagreement.metrics import paired_comparison
from field_disagreement.triage import ServedRule
from field_disagreement.triage.ladder import (ABLATIONS, ALL_ARMS, CLAIM, CONTROL, DISPLAY_NAMES,
                                    PRIOR_ART, TRAINED_ARMS, ships)
from backend.domains.datasets import DATASETS

# The nine study arms. An arm with no result file is skipped by every table, and a
# file of an arm outside them (an older run's) is never read into one.
ARMS = tuple(ALL_ARMS)
RIVALS = PRIOR_ART + (CONTROL,)
# `d_profile` field names by field count, from the dataset registry (the one
# place a cohort's report fields are named)
FIELD_NAMES = {s.n_fields: s.field_names for s in DATASETS.values() if s.field_names}


def study_rule(config: str = "notebooks/study") -> ServedRule:
    """The study's triage rule, from its yaml's `served_rule:` block (deployed
    defaults for any key it leaves out). Every comparison reads the rule from
    here or from an explicit `rule=` argument, never from a constant."""
    from backend.config import load
    return ServedRule.from_config(getattr(load(config), "served_rule", None))
N_BOOT = 2000
BETAS = (0.05, 0.10, 0.15, 0.20, 0.30)
_TAG = re.compile(r"_([A-Z]_[A-Za-z0-9]+)(?:_s(\d+))?_fold(\d)\.result\.json$")
from .study import BASE_SEED   # the seed whose result files carry no `_s<seed>` suffix


def spearman(x, y) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def _file_seed(m) -> int:
    return int(m.group(2)) if m.group(2) is not None else BASE_SEED


def result_files(run_dir, verbose: bool = False) -> list[Path]:
    """Every per-fold result file of a run. `run_dir` may also be a LIST of run
    folders - seeds trained in separate sessions land in separate dated folders,
    and their files carry distinct `_s<seed>` tags, so they pool by listing them.
    The list is in PRIORITY order: when two folders hold the same (seed, arm,
    fold), the earlier folder's file is read - a rerun of some arms (the first
    folder) replaces those arms of the study it is compared with."""
    dirs = [run_dir] if isinstance(run_dir, (str, Path)) else list(run_dir)
    missing = [str(d) for d in dirs if not Path(d).is_dir()]
    if missing:
        raise FileNotFoundError(f"run folder(s) not found: {', '.join(missing)}")
    seen: dict = {}
    replaced = []
    for d in dirs:
        for f in sorted(Path(d, "checkpoints").glob("*_fold*.result.json")):
            if f.name in seen:
                replaced.append(f"{f.name} ({Path(d).name})")
                continue
            seen[f.name] = d
    if verbose and replaced:
        print(f"  {len(replaced)} result file(s) of later folders replaced by "
              f"{Path(dirs[0]).name}: " + ", ".join(sorted(replaced)[:6])
              + (" ..." if len(replaced) > 6 else ""))
    return [Path(seen[n], "checkpoints", n) for n in sorted(seen)]


def _first(run_dir) -> Path:
    return Path(run_dir if isinstance(run_dir, (str, Path)) else list(run_dir)[0])


def load_run(run_dir, seed: int = BASE_SEED, selection: str = "primary") -> dict:
    """{fold: {arm: result}} for every result file of ONE training seed in the run
    (seed 42: `<arm>_fold<k>`; any other: `<arm>_s<seed>_fold<k>`).

    `selection="sensitivity"` reads each unit as scored at the epoch its SECOND
    selection rule kept (`sel_also`; here: validation AUC only) - same training
    run, other checkpoint. A unit without one is left out."""
    res: dict = {}
    for f in result_files(run_dir):
        m = _TAG.search(f.name)
        if m and _file_seed(m) == int(seed):
            r = json.loads(f.read_text())
            if selection == "sensitivity":
                r = sensitivity_view(r)
                if r is None:
                    continue
            res.setdefault(int(m.group(3)), {})[m.group(1)] = r
    return res


def sensitivity_view(result: dict) -> dict | None:
    """One result as its second selection rule scored it: its own test columns,
    or the primary ones when both rules kept the same epoch."""
    s = result.get("sensitivity")
    if not s:
        return None
    base = {k: v for k, v in result.items() if k != "sensitivity"}
    if s.get("same_epoch"):
        return {**base, "selection": s["selection"]}
    return {**{k: v for k, v in base.items() if not k.startswith("test_")},
            **{k: v for k, v in s.items() if k != "same_epoch"}}


def seeds_in(run_dir, claim: str = CLAIM) -> list[int]:
    """The training seeds with at least one `claim` result in the run, base seed first."""
    found = set()
    for f in result_files(run_dir):
        m = _TAG.search(f.name)
        if m and m.group(1) == claim:
            found.add(_file_seed(m))
    return sorted(found, key=lambda s: (s != BASE_SEED, s))


def _col(result, key):
    return np.asarray(result[f"test_{key}"], dtype=float)


def _profile(fold_res: dict):
    """The panel's disagreement on each report field, (N, P), for one fold's test
    items. Every arm is scored on the same items in the same order (checked in
    `_contrast`), so an arm cached before the column existed borrows it."""
    for r in fold_res.values():
        if r.get("test_d_profile") is not None:
            return np.asarray(r["test_d_profile"], dtype=float)
    return None


def _fields(result: dict, arm: str, n_fields: int) -> np.ndarray:
    """(N, P): a per-field head scores each field with its own output; every
    other arm ranks every field with the one signal it ships."""
    if result.get("test_panel_disp_fields") is not None:
        return np.asarray(result["test_panel_disp_fields"], dtype=float)
    return np.repeat(_col(result, ships(arm))[:, None], n_fields, axis=1)


def field_rho(S: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Spearman on each report field, (P,)."""
    return np.array([spearman(S[:, j], D[:, j]) for j in range(D.shape[1])])


def _score(x, d, b=None) -> float:
    if b is not None:
        x, d = x[b], d[b]
    return float(np.mean(field_rho(x, d))) if d.ndim == 2 else spearman(x, d)


def _signal(fold_res: dict, arm: str, target: str):
    """What a contrast ranks: `macro` = each arm's per-field signal against the
    per-field disagreement; `report` = its shipped signal against report-level d."""
    r = fold_res[arm]
    if target == "report":
        return _col(r, ships(arm)), _col(r, "disagreement")
    D = _profile(fold_res)
    return _fields(r, arm, D.shape[1]), D


def _contrast(res: dict, claim: str, rivals, target: str, n_boot: int, seed: int) -> dict:
    """`claim` minus each rival on each fold's test items, with a paired
    patient-clustered bootstrap. One set of resamples per fold, shared by every
    rival, so every contrast on a fold is read on the same draws."""
    rng = np.random.default_rng(seed)
    folds = sorted(k for k in res if claim in res[k])
    rivals = tuple(r for r in rivals if any(r in res[k] for k in folds))
    rho = {a: [] for a in (claim,) + rivals}
    contrast = {r: [] for r in rivals}
    for k in folds:
        x, d = _signal(res[k], claim, target)
        cl = np.asarray(res[k][claim]["test_clusters"])
        idx = [np.flatnonzero(cl == c) for c in np.unique(cl)]
        boots = [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
                 for _ in range(n_boot)]
        xb = np.array([_score(x, d, b) for b in boots])
        rho[claim].append(_score(x, d))
        for r in rivals:
            if r not in res[k]:
                rho[r].append(np.nan)
                contrast[r].append(None)
                continue
            y, _ = _signal(res[k], r, target)
            if len(y) != len(x):
                raise ValueError(f"fold {k}: {r} and {claim} were scored on different items")
            rho[r].append(_score(y, d))
            dd = xb - np.array([_score(y, d, b) for b in boots])
            lo, hi = np.percentile(dd, [2.5, 97.5])
            contrast[r].append({"delta": rho[claim][-1] - rho[r][-1],
                                "lo": float(lo), "hi": float(hi), "boot": dd})
    return {"folds": folds, "rho": rho, "contrast": contrast, "claim": claim,
            "target": target}


def primary(res: dict, claim: str = CLAIM, rivals=RIVALS,
            n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """PRIMARY: the panel's disagreement on EACH report field, ranked by each arm,
    Spearman averaged over the fields (macro); `claim` minus each rival."""
    return _contrast(res, claim, rivals, "macro", n_boot, seed)


def report_level(res: dict, claim: str = CLAIM, rivals=RIVALS,
                 n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """SECONDARY 1: Spearman(shipped signal, report-level d), the case-level read."""
    return _contrast(res, claim, rivals, "report", n_boot, seed)


def per_field(res: dict, arms=ARMS) -> dict:
    """Each arm's Spearman on each report field, mean over folds, (P,)."""
    out = {}
    for a in arms:
        rows = [field_rho(_fields(res[k][a], a, _profile(res[k]).shape[1]), _profile(res[k]))
                for k in sorted(res) if a in res[k] and _profile(res[k]) is not None]
        if rows:
            out[a] = np.mean(rows, axis=0)
    return out


def pooled(prim: dict) -> dict:
    """PRIMARY, pooled: the mean of the per-fold contrasts, with a fold-stratified
    interval (each fold's resamples averaged draw by draw; folds are
    patient-disjoint, so their draws are independent)."""
    out = {}
    for r, per in prim["contrast"].items():
        if any(t is None for t in per):
            continue
        boot = np.mean([t["boot"] for t in per], axis=0)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[r] = {"delta": float(np.mean([t["delta"] for t in per])),
                  "lo": float(lo), "hi": float(hi)}
    return out


def headline_beta(res: dict, baseline: str = "A_cls"):
    """The smallest beta at which the BASELINE's confidence router certifies
    anything, averaged over folds. The baseline picks the operating point."""
    folds = [k for k in res if baseline in res[k]]
    for b in BETAS:
        cov = [res[k][baseline].get(f"front_confidence_b{b:.2f}_coverage", 0.0)
               for k in folds]
        if folds and np.mean(cov) > 0:
            return b
    return None


def arm_table(res: dict, d0: float, arms=ARMS) -> dict:
    """SECONDARY 2-5, per arm, mean over folds. `d0` is the cohort's declared
    contested threshold (`contested_d`)."""
    beta = headline_beta(res)
    rows = {}
    for a in arms:
        rs = [res[k][a] for k in sorted(res) if a in res[k]]
        if not rs:
            continue
        s = ships(a)
        sep = [M.roc_auc((_col(r, "disagreement") > d0).astype(int), _col(r, s))
               for r in rs]
        rows[a] = {
            "n_folds": len(rs),
            "auc": [r["auc"] for r in rs],
            # a signal that IS confidence residualises to 0 by construction
            "semi_partial": [r.get(f"sp_{s}", 0.0 if s == "confidence" else np.nan)
                             for r in rs],
            "contested_auroc": sep,
            "leakage": [r.get(f"leak_{s}", np.nan) for r in rs],
            "leak_base": [r.get("leak_base", np.nan) for r in rs],
            "certified": ([r.get(f"front_{s}_b{beta:.2f}_coverage", np.nan) for r in rs]
                          if beta is not None else []),
        }
    return {"rows": rows, "beta": beta}


def _served_split(t: dict, cal: np.ndarray, d0: float, rule: ServedRule, rng) -> dict:
    """One calibration/evaluation cut of one fold: every variant's cuts set on
    `cal`, every outcome read on the rest. `t` holds the fold's test columns;
    `signal` is None for a classifier-only rule."""
    ev = ~cal
    p, conf, sig = t["probs"], t["confidence"], t["signal"]
    conf_cut = triage.safety_cut(conf[cal], p[cal], t["labels"][cal], rule.miss)
    same = ~(triage.served_refer(conf, p, conf_cut=conf_cut) | t["empty"])
    sota_cut = triage.safety_cut(conf[cal], p[cal], t["labels"][cal], rule.sota_miss)
    out = {"samecut": same,
           "sota": ~(triage.served_refer(conf, p, conf_cut=sota_cut) | t["empty"])}
    if sig is None:
        out.update(served=same, matched=same)
    else:
        net = t["net"] if rule.safety_net > 0 else None
        refer = triage.served_refer(
            conf, p, sig, conf_cut=conf_cut, obvious=rule.obvious,
            disp_cut=triage.veto_cut(sig[cal], p[cal], rule.veto),
            extreme_cut=triage.veto_cut(sig[cal], p[cal], rule.extreme),
            net=net, net_cut=(triage.veto_cut(net[cal], p[cal], rule.safety_net)
                              if net is not None else None))
        out["served"] = ~(refer | t["empty"])
        rate = float(np.mean(~refer[cal]))
        cut = float(np.quantile(conf[cal], rate)) if 0 < rate < 1 else conf_cut
        out["matched"] = ~(triage.served_refer(conf, p, conf_cut=cut) | t["empty"])
        u = rng.random(len(p))
        rand = ((triage.vetoable(p, rule.obvious) & (u < rule.veto))
                | ((p < 0.5) & (u < rule.extreme)))
        if rule.safety_net > 0:
            rand = rand | ((p < 0.5) & (rng.random(len(p)) < rule.safety_net))
        out["randveto"] = same & ~rand
    return {v: M.served_outcomes(auto[ev], p[ev], t["labels"][ev],
                                 t["disagreement"][ev], d0)
            for v, auto in out.items()}


def served_folds(res: dict, clf: str, sig_arm: str | None, d0: float, *,
                 sig_col: str | None = None, rule: ServedRule | None = None,
                 n_splits: int | None = None, seed: int = 42,
                 net_sig: str | None = None) -> dict:
    """{variant: {outcome: [per fold]}} for the classifier of `clf` and the veto
    signal of `sig_arm` (None: the certainty cut alone), read from banked
    `test_*` columns, under `rule` (default: the deployed `ServedRule()`).
    `sig_col` names the signal column (default: the ladder's `ships(sig_arm)`);
    `net_sig` the safety net's (default `net_col`). Each fold's number is the
    mean over `rule.splits` (or `n_splits`)
    patient-disjoint halvings of its test split: with a few dozen positives per
    half, one halving moves the certainty cut by several points of automation,
    so a single draw cannot separate rules whose difference is smaller than that."""
    from .panel import split_for_calibration
    rule = rule or ServedRule()
    if n_splits is not None:
        rule = replace(rule, splits=int(n_splits))
    rows: dict = {}
    for k in sorted(res):
        r, s = res[k].get(clf), res[k].get(sig_arm) if sig_arm else None
        col = (sig_col or ships(sig_arm)) if sig_arm else None
        need = ("probs", "labels", "disagreement", "confidence", "clusters")
        if r is None or any(f"test_{c}" not in r for c in need):
            continue
        if sig_arm and (s is None or f"test_{col}" not in s):
            continue
        p = _col(r, "probs")
        q = r.get("conf_qhat")
        t = {"labels": _col(r, "labels").astype(int), "probs": p,
             "disagreement": _col(r, "disagreement"), "confidence": _col(r, "confidence"),
             "signal": _col(s, col) if sig_arm else None,
             "net": (_col(s, net_sig or net_col(s, sig_arm))
                     if sig_arm and rule.safety_net > 0 else None),
             "empty": (C.lac_predict(C.binary_probs(p), q).sum(axis=1) == 0
                       if q is not None else np.zeros(len(p), dtype=bool))}
        rng = np.random.default_rng(seed)
        per = []
        for j in range(rule.splits):
            cal, _ = split_for_calibration({"labels": t["labels"],
                                            "idx": np.arange(len(p))},
                                           clusters=r["test_clusters"], seed=seed + j)
            m = np.zeros(len(p), dtype=bool)
            m[cal["idx"]] = True
            per.append(_served_split(t, m, d0, rule, rng))
        for v in per[0]:
            for o in M.SERVED_OUTCOMES:
                rows.setdefault(v, {}).setdefault(o, []).append(
                    float(np.nanmean([x[v][o] for x in per])))
    return rows


def served(res: dict, d0: float, arms=ARMS, classifier: str = "A_cls",
           rule: ServedRule | None = None, seed: int = 42,
           n_splits: int | None = None) -> dict:
    """The triage rule, re-read from the stored test
    columns under the DEPLOYED rule (`rule`, default `ServedRule()`): the certainty
    cut that clears at most `rule.miss` of the positives, plus the veto of the
    `rule.veto` share of negative calls with the highest shipped signal (an
    obvious negative only in the top `rule.extreme` share), no safety net.

    The per-item VALIDATION outputs were not banked, so every cut is set the way
    S5's certificate is: on a patient-disjoint half of each fold's test split
    (`panel.split_for_calibration`), and read on the other half, averaged over
    `n_splits` halvings. Two tables:

      "own"     each arm serving its own classifier and its own signal (an arm
                that ships confidence serves the certainty cut alone)
      "signal"  ONE classifier (`classifier`, the plain one) for every arm, vetoing
                with each arm's signal: only the veto's ranking differs, so this is
                the read of which signal finds the cases that should not be cleared

    Each: {arm: {variant: {outcome: [per fold]}}}, variants as SERVED_VARIANTS."""
    rule = rule or ServedRule()
    if n_splits is not None:
        rule = replace(rule, splits=int(n_splits))
    kw = dict(rule=rule, seed=seed)
    own = {a: served_folds(res, a, None if ships(a) == "confidence" else a, d0, **kw)
           for a in arms}
    sig = {a: served_folds(res, classifier, None if ships(a) == "confidence" else a, d0, **kw)
           for a in arms}
    return {"own": {a: v for a, v in own.items() if v},
            "signal": {a: v for a, v in sig.items() if v},
            "classifier": classifier, "rule": rule, "miss": rule.miss, "veto": rule.veto,
            "extreme": rule.extreme}


_SERVED_COLS = (("auto", "AI decides", True), ("missed_per_1000", "missed/1000", False),
                ("high_neg", "hi-contest neg", False), ("contested_auto", "contested AI", False),
                ("error", "error", False), ("obvious_pos_auto", "obvious pos", True),
                ("obvious_neg_auto", "obvious neg", True))


# --------------------------------------------------------------------------- #
# THE FIELD-PRIORITY VETO (`served_v5`). The rule above (`claim_v4`) ranks a
# negative call by the MEAN of the per-field predictions, every field equal.
# This variant gives each field a weight for how well its
# predicted disagreement flags a CANCER the AI would clear, and sends the call to
# a reader when EITHER lane is high:
#   lane 1 (priority)  the weighted fields; weights are a ridge logistic of
#                      "positive" on the within-fold field ranks, fitted on the
#                      negative calls of the OTHER folds' test items and applied to
#                      this fold (out-of-fold: no item scores itself)
#   lane 2 (overall)   the mean over fields, as in `claim_v4`
# signal = max(rank lane 1, rank lane 2) within the fold. It is printed, never
# judged (only the claim would get learned weights).
#
# THE STUDY'S RULE (the paper's triage analysis; values in
# configs/notebooks/study.yaml `served_rule:`, read by `study_rule`): the veto
# share, plus a SAFETY NET - that share of ALL negative calls (obvious ones
# included) with the highest net signal also go to a reader. Every arm gets the
# same rule from its OWN outputs (`net_col`): its predicted opinion spread when it
# has an opinion head, else the signal it ships. The claim is compared with each
# prior-art rule, paired over folds: disputed cases cleared (lower) and missed
# positives (non-inferior within `rule.miss_margin` per 1000).
PRIORITY_COL = "priority_lanes"


def net_col(result: dict, arm: str) -> str:
    """The safety net's signal for `arm`: its predicted opinion spread if it has an
    opinion head (and the column was banked), else its shipped signal."""
    if (arm in TRAINED_ARMS and TRAINED_ARMS[arm].opinion != "none"
            and result.get("test_panel_spread") is not None):
        return "panel_spread"
    return ships(arm)


MC_DROPOUT = "A_cls_mc"
TRIAGE_RIVALS = PRIOR_ART + (MC_DROPOUT,)


def pool_served_priority(sps) -> dict | None:
    """Several cohorts' `served_priority` outputs as one, their folds concatenated
    (the declared confirmatory read is LIDC + Chakshu pooled, 10 folds); only arms
    present in every cohort are kept."""
    sps = [x for x in sps if x]
    if not sps:
        return None

    def cat(rows):
        return {o: sum((list(r[o]) for r in rows), []) for o in rows[0]}
    common = set.intersection(*(set(x["signal"]) for x in sps))
    return {"rule": sps[0].get("rule"),
            "claim_v4": {"served": cat([x["claim_v4"]["served"] for x in sps])},
            "signal": {a: {"served": cat([x["signal"][a]["served"] for x in sps])}
                       for a in sps[0]["signal"] if a in common}}


def triage_contrasts(sp: dict | None, rule: ServedRule | None = None,
                    rivals=TRIAGE_RIVALS) -> dict:
    """The claim against every prior-art rule, paired over folds
    (`metrics.paired_comparison`): {arm: {"disputed": (diff, lo, hi, lower,
    folds fewer), "missed": (diff, lo, hi, non-inferior, folds not more),
    "auto": (diff, lo, hi), "n"}}. Disputed is `lower` when the interval is wholly
    below 0; missed is non-inferior when its upper bound is below
    `rule.miss_margin` per 1000 (`rule` default: `sp["rule"]`). The claim is
    read on `claim_v4`: it vetoes by the signal it ships, exactly as every rival
    does, with no weights learned for it alone (the learned field-priority row
    is printed, never judged)."""
    if not sp:
        return {}
    rule = rule or sp.get("rule") or study_rule()
    c = sp["claim_v4"]["served"]
    out = {}
    for a, v in sp["signal"].items():
        if a not in rivals:
            continue
        r = v["served"]
        d = paired_comparison(c["high_neg"], r["high_neg"])
        m = paired_comparison(c["missed_per_1000"], r["missed_per_1000"],
                              margin=rule.miss_margin)
        au = paired_comparison(c["auto"], r["auto"])
        out[a] = {"disputed": (d["diff"], d["lo"], d["hi"], d["lower"], d["fewer"]),
                  "missed": (m["diff"], m["lo"], m["hi"], m["noninferior"], m["not_more"]),
                  "auto": (au["diff"], au["lo"], au["hi"]), "n": d["n"]}
    return out


def _within_rank(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (rankdata(x) - 0.5) / len(x) - 0.5


def _ridge_logistic(X, y, l2: float = 1e-2, iters: int = 100):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (q - y) / len(y) + reg * w
        H = (Z * (q * (1 - q))[:, None]).T @ Z / len(y) + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return (lambda X2: np.c_[(X2 - mu) / sd, np.ones(len(X2))] @ w), w[:-1] / sd


def priority_lanes(res: dict, claim: str = CLAIM, classifier: str = "A_cls"):
    """Write v5's two-lane signal into each fold's `claim` result as
    `test_priority_lanes`; return the learned field weights (mean over folds) or
    None when the run has no per-field outputs."""
    folds = [k for k in sorted(res) if claim in res[k] and classifier in res[k]
             and "test_panel_disp_fields" in res[k][claim]
             and "test_panel_disp" in res[k][claim]
             and all(f"test_{c}" in res[k][classifier] for c in ("probs", "labels"))]
    if len(folds) < 2:
        return None

    def fields(k):
        F = np.asarray(res[k][claim]["test_panel_disp_fields"], dtype=float)
        return np.column_stack([_within_rank(F[:, j]) for j in range(F.shape[1])])

    def negatives(k):
        a = res[k][classifier]
        return (np.asarray(a["test_probs"], float) < 0.5,
                (np.asarray(a["test_labels"]).astype(int) == 1).astype(float))

    weights = []
    for k in folds:
        X, y = [], []
        for j in folds:
            if j != k:
                neg, pos = negatives(j)
                X.append(fields(j)[neg])
                y.append(pos[neg])
        score, w = _ridge_logistic(np.concatenate(X), np.concatenate(y))
        weights.append(w)
        r = res[k][claim]
        lane1 = _within_rank(score(fields(k)))
        lane2 = _within_rank(np.asarray(r["test_panel_disp"], float))
        r[f"test_{PRIORITY_COL}"] = np.maximum(lane1, lane2).tolist()
    return np.mean(weights, axis=0)


def served_priority(res: dict, d0: float, arms=ARMS, claim: str = CLAIM,
                    classifier: str = "A_cls", rule: ServedRule | None = None,
                    n_splits: int | None = None) -> dict | None:
    """The study's rule (`rule`, default `study_rule()`) for the claim and for
    every other arm on its own signal and its own safety net (`net_col`), with
    the certainty-only / SOTA references and the learned field-priority row.
    None without per-field outputs. An arm that ships confidence (A_cls, random-
    reader training) vetoes by ITS OWN network's confidence."""
    w = priority_lanes(res, claim, classifier)
    if w is None:
        return None
    rule = rule or study_rule()
    if n_splits is not None:
        rule = replace(rule, splits=int(n_splits))
    P = len(w)
    rows = {"weights": dict(zip(FIELD_NAMES.get(P, [f"field {j}" for j in range(P)]),
                                map(float, w))), "rule": rule, "veto": rule.veto,
            "net": rule.safety_net, "signal": {}}
    kw = dict(rule=rule)
    rows["claim"] = served_folds(res, classifier, claim, d0, sig_col=PRIORITY_COL, **kw)
    rows["claim_v4"] = served_folds(res, classifier, claim, d0, **kw)
    for a in arms:
        if a == claim:
            continue
        v = served_folds(res, classifier, a, d0, **kw)
        if v:
            rows["signal"][a] = v
    # MC dropout (Gal & Ghahramani 2016): the plain classifier's own 20-pass
    # epistemic term as the veto AND the net. No extra network: A_cls is trained
    # with dropout and every result banks `test_epistemic`.
    if MC_DROPOUT not in rows["signal"] and "A_cls" in arms:
        v = served_folds(res, classifier, "A_cls", d0, sig_col="epistemic",
                         net_sig="epistemic", **kw)
        if v:
            rows["signal"][MC_DROPOUT] = v
    return rows


def print_served_priority(sp: dict | None, claim: str = CLAIM) -> None:
    if sp is None:
        return
    print(f"\nTRIAGE RULE  field-priority veto, {sp['veto']:.0%} of negative calls"
          + (f", + safety net: the {sp['net']:.0%} of ALL negative calls with the highest"
             " opinion spread (each arm its own)" if sp.get("net") else ""))
    print("  learned field weights (larger = that field's disagreement flags more "
          "cancers the AI would clear):")
    print("    " + "  ".join(f"{k} {v:+.2f}" for k, v in
                             sorted(sp["weights"].items(), key=lambda kv: -kv[1])))
    head = "".join(f"{h + (' (^)' if up else ' (v)'):>18s}" for _, h, up in _SERVED_COLS[:5])
    print(f"  {'rule':40s}{head}")

    def line(name, rows):
        cells = "".join(f"{_mean(rows[k]):18.1f}" if k == "missed_per_1000"
                        else f"{_mean(rows[k]):18.3f}" for k, _, _ in _SERVED_COLS[:5])
        print(f"  {name:40s}{cells}")
    c = sp["claim"]
    line(f"standard: certainty cut, {triage.SOTA_MISS:.0%} limit", c["sota"])
    line("certainty only, same automation", c["matched"])
    line("random veto, same size", c["randveto"])
    line(f"{claim} own signal (JUDGED)", sp["claim_v4"]["served"])
    line(f"{claim} learned field priority", c["served"])
    for a, v in sp["signal"].items():
        line(f"veto by {a}", v["served"])
    crit = triage_contrasts(sp)
    if not crit:
        return
    print(f"\n  {claim} MINUS EACH PRIOR-ART METHOD  paired over folds, 95% interval;"
          " folds = folds where ours is lower (disputed) / not higher (missed)")
    print(f"  {'vs':10s}{'disputed cleared (points)':>30s}{'folds':>7s}"
          f"{'missed per 1000':>30s}{'folds':>7s}{'AI decides (points)':>26s}")
    for a, r in crit.items():
        d, m, au = r["disputed"], r["missed"], r["auto"]
        print(f"  {a:10s}{d[0]*100:+8.1f} [{d[1]*100:+6.1f}, {d[2]*100:+6.1f}]     "
              f"{d[4]:>5d}/{r['n']}"
              f"{m[0]:+8.1f} [{m[1]:+6.1f}, {m[2]:+6.1f}]     "
              f"{m[4]:>5d}/{r['n']}{au[0]*100:+8.1f} [{au[1]*100:+5.1f}, {au[2]*100:+5.1f}]")


def print_served(srv: dict, claim: str = CLAIM) -> None:
    """The two SERVED tables, with fold counts for the claim arm."""
    widths = [max(len(h) + 5, 11) for _, h, _ in _SERVED_COLS]
    head = "".join(f"{h + (' (^)' if up else ' (v)'):>{wd}s}"
                   for (_, h, up), wd in zip(_SERVED_COLS, widths))

    def line(label, rows):
        return f"  {label:34s}" + "".join(f"{_mean(rows[k]):{wd}.3f}" if k != "missed_per_1000"
                                          else f"{_mean(rows[k]):{wd}.1f}"
                                          for (k, _, _), wd in zip(_SERVED_COLS, widths))

    print(f"\nSERVED RULE  the model alone clears <= {srv['miss']:.0%} of positives (certainty "
          f"cut); the {srv['veto']:.0%} most-disputed negative calls go to a reader,"
          f"\n  an obvious one (p <= {triage.OBVIOUS_NEGATIVE:g}) only in the top "
          f"{srv.get('extreme', triage.EXTREME_VETO):.0%}"
          f"\n  cuts on a patient-disjoint half of test, read on the other half, "
          f"{srv['rule'].splits} halvings per fold; (^) higher / (v) lower is better;"
          f"\n  'neg' = cleared as negative by the system alone, over all such cases")
    print(f"\n  each arm, its own model{'':11s}{head}")
    for a, rows in srv["own"].items():
        print(line(a + (" (certainty only)" if "randveto" not in rows else ""), rows["served"]))
    sig = srv["signal"]
    base = sig.get(srv["classifier"])
    if claim not in sig or base is None:
        return
    print(f"\n  one classifier ({srv['classifier']}), veto by each signal"
          f"{'':1s}{head}")
    print(line(f"SOTA: certainty cut, {triage.SOTA_MISS:.0%} limit", sig[claim]["sota"]))
    print(line(f"certainty only, {srv['miss']:.0%} limit", base["served"]))
    print(line("certainty only, same automation", sig[claim]["matched"]))
    print(line("random veto, same size", sig[claim]["randveto"]))
    for a, rows in sig.items():
        if a != srv["classifier"]:
            print(line(f"veto by {a}" + ("  <- proposed" if a == claim else ""), rows["served"]))

    def folds(x, y, strict=True):
        won = [(u < v) if strict else (u <= v + 1e-12)
               for u, v in zip(x, y) if u == u and v == v]
        return f"{sum(won)}/{len(won)}"
    c = sig[claim]
    hi = c["served"]["high_neg"]
    rivals = ", ".join(f"{a} {folds(hi, sig[a]['served']['high_neg'])}"
                       for a in RIVALS if a in sig and a != srv["classifier"])
    print(f"\n  folds where {claim} clears FEWER highly contested cases as negative:"
          f"\n    vs SOTA {folds(hi, c['sota']['high_neg'])}, same automation "
          f"{folds(hi, c['matched']['high_neg'])}, random veto "
          f"{folds(hi, c['randveto']['high_neg'])}; vs each other signal: {rivals}")
    print(f"  no more errors than SOTA in "
          f"{folds(c['served']['error'], c['sota']['error'], strict=False)} folds; no more "
          f"missed positives in "
          f"{folds(c['served']['missed_per_1000'], c['sota']['missed_per_1000'], strict=False)}")


def overfit_check(res: dict, arms=ARMS) -> dict:
    """Per arm, over folds: the epoch kept, the epoch of the best smoothed val AUC
    (from the result's `history`, when banked), and selected val AUC minus test
    AUC. A kept epoch far past the val peak, or a gap well above the other arms',
    is the sign of an over-fitted checkpoint."""
    out = {}
    for a in arms:
        rows = [selection_summary(res[k][a]) for k in sorted(res)
                if res[k].get(a) and "auc" in res[k][a]]
        if rows:
            # rho_share: the logged disagreement signal (val) at the kept epoch as a
            # share of its best anywhere in the run - did the head finish learning?
            out[a] = {"kept": [r["kept"] for r in rows], "peak": [r["peak"] for r in rows],
                      "gap": [r["gap"] for r in rows],
                      "rho_share": [r["signal_share"] for r in rows]}
    return out


def floor_check(res: dict, floor: int, share: float, max_floor: int, window: int,
                arms=ARMS) -> dict:
    """The floor check: is the selection floor (epoch 10) late enough? For
    every unit of an arm trained on a panel target: the epoch its smoothed
    validation disagreement signal first reached `share` of its best, the kept
    epoch, and the share it had there. Two different findings:

      underfit  kept BEFORE the signal reached `share` of its best - the head had
                not finished learning, which a later floor fixes. Any one stops
                the run; the floor for EVERY arm becomes the latest such epoch,
                capped at `max_floor`.
      drifted   kept after the signal peaked, by which time it had fallen below
                `share` - the head learned and then drifted. A later floor
                cannot fix that, so it is reported, never acted on.

    {"ok", "floor", "underfit": [(arm, fold)], "drifted": [(arm, fold)],
     "units": {arm: [(fold, reached, kept, kept_share)]}}"""
    from field_disagreement.engine import signal_learned_by
    units, late, underfit, drifted = {}, [], [], []
    for a in arms:
        spec = TRAINED_ARMS.get(a)
        if spec is None or (spec.opinion == "none" and not spec.regress):
            continue            # no panel target: nothing that has to finish learning
        if spec.shuffle != "none":
            continue            # a shuffled control has no signal to learn, by design
        for k in sorted(res):
            r = res[k].get(a)
            if not r:
                continue
            v = signal_learned_by(r, share, window)
            kept = (r.get("selection") or {}).get("epoch")
            units.setdefault(a, []).append((k, v["reached"], kept, v["kept_share"]))
            if kept is None or v["kept_share"] != v["kept_share"]:
                continue        # no history logged: nothing to read
            if v["reached"] is None or kept < v["reached"]:
                underfit.append((a, k))
                late.append(v["reached"] or max_floor)
            elif v["kept_share"] < share:
                drifted.append((a, k))
    ok = not underfit
    return {"ok": ok, "floor": floor if ok else min(max(max(late), floor), max_floor),
            "underfit": underfit, "drifted": drifted, "units": units}


def print_floor_check(fc: dict, share: float) -> None:
    print(f"\nFLOOR CHECK  per fold: epoch the smoothed val disagreement signal reached "
          f"{share:.0%} of its best / epoch kept / its share of best there")
    for a, rows in fc["units"].items():
        print(f"  {a:8s} " + "  ".join(
            f"f{k}: ep {r if r is not None else '-'} / kept {e} / {s:.0%}"
            for k, r, e, s in rows))
    if fc["drifted"]:
        print("  drifted (kept after the signal peaked; a floor cannot fix it): "
              + ", ".join(f"{a} f{k}" for a, k in fc["drifted"]))
    print("  -> every head had finished learning by its kept epoch: floor is enough"
          if fc["ok"] else
          f"  -> UNDERFIT ({', '.join(f'{a} f{k}' for a, k in fc['underfit'])}): "
          f"raise the floor to epoch {fc['floor']} for every arm")


def print_overfit_check(oc: dict) -> None:
    if not oc:
        return
    print("\nOVERFITTING CHECK  per fold: epoch kept / epoch of best smoothed val AUC;"
          " selected val AUC minus test AUC")
    for a, v in oc.items():
        ep = "  ".join(f"{k}/{p if p is not None else '-'}" for k, p in zip(v["kept"], v["peak"]))
        sh = v.get("rho_share") or []
        rs = (f"   disagreement signal at kept epoch {_mean(sh):.0%} of its best"
              if any(x == x for x in sh) else "")
        print(f"  {a:8s} gap {_mean(v['gap']):+.3f}   {ep}{rs}")


def auc_parity(res: dict, claim: str = CLAIM, arms=ARMS) -> dict:
    """PRECONDITION: claim minus each arm on test AUC, paired over folds."""
    folds = sorted(k for k in res if claim in res[k])
    out = {}
    for a in arms:
        if a == claim or not all(a in res[k] for k in folds):
            continue
        out[a] = M.paired_fold_test([{"v": res[k][claim]["auc"]} for k in folds],
                                    [{"v": res[k][a]["auc"]} for k in folds], "v")
    return out


def _mean(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def cohort_d0(run_dir) -> float:
    """The declared contested threshold of the cohort this run trained on."""
    from backend.domains.datasets import DATASETS
    name = _first(run_dir).name
    hits = [n for n in DATASETS if f"_{n}_" in f"_{name}_"]
    if not hits:
        raise ValueError(f"cannot tell the cohort from {name!r}; pass d0")
    return float(DATASETS[max(hits, key=len)].contested_d)


# --------------------------------------------------------------------------- #
# TWO-SIDED TRIAGE. The AI decides a
# FIXED share of each fold's cases: it clears the `clear` share with the lowest
# score as negative and flags the `flag` share of the rest with the highest
# score as positive; everything else goes to a reader. Each arm ranks with its
# OWN classifier, blended with its own shipped signal:
#     clear score = (1 - w) rank(p) + w rank(signal)      (low = clear)
#     flag  score = (1 - w) rank(-p) + w rank(signal)     (low = flag)
# so a case the arm expects the panel to dispute is held back from BOTH sides.
# The shares and w are fixed here, never tuned on test. The reference is the
# plain classifier on certainty alone (w = 0), the standard of care.
# --------------------------------------------------------------------------- #
TRIAGE_CLEAR, TRIAGE_FLAG, TRIAGE_W = 0.30, 0.25, 0.2
TRIAGE_SOTA = "A_cls"


def _risk(result: dict, arm: str) -> np.ndarray:
    """The arm's shipped signal, oriented so HIGHER = more likely to be disputed."""
    s = _col(result, ships(arm)).ravel()
    return -s if ships(arm) == "confidence" else s


def triage_fold(result: dict, arm: str, d0: float, clear=TRIAGE_CLEAR,
                flag=TRIAGE_FLAG, w=TRIAGE_W) -> dict:
    """One fold of the two-sided rule: counts per 1000 cases and shares."""
    from scipy.stats import rankdata
    p = np.asarray(result["test_probs"], float).ravel()
    y = np.asarray(result["test_labels"], float).ravel()
    d = np.asarray(result["test_disagreement"], float).ravel()
    n = len(p)
    w = 0.0 if ships(arm) == "confidence" else float(w)
    rs = rankdata(_risk(result, arm)) / n
    neg = np.argsort((1 - w) * rankdata(p) / n + w * rs, kind="stable")[:int(round(clear * n))]
    rest = np.setdiff1d(np.arange(n), neg)
    pos = rest[np.argsort((1 - w) * rankdata(-p[rest]) / n + w * rs[rest],
                          kind="stable")[:int(round(flag * n))]]
    auto = np.r_[neg, pos]
    hi = d >= d0
    return {"automated": len(auto) / n,
            "missed_per_1000": 1000 * y[neg].sum() / n,
            "false_flags_per_1000": 1000 * (1 - y[pos]).sum() / n,
            "contested_cleared": float(hi[neg].mean()) if len(neg) else float("nan"),
            "contested_automated": float(hi[auto].mean()) if len(auto) else float("nan"),
            "missed": int(y[neg].sum()), "n": n}


def two_sided_triage(res: dict, d0: float, arms=ARMS, claim: str = CLAIM,
                     sota: str = TRIAGE_SOTA, **kw) -> dict:
    """{arm: {outcome: [per fold]}} plus the claim's per-fold record against the
    standard of care: fewer-or-equal missed cancers AND a lower contested share
    among the cases the AI decides (cleared or flagged)."""
    folds = [k for k in sorted(res) if claim in res[k] and sota in res[k]]
    need = ("test_probs", "test_labels", "test_disagreement")
    out = {a: {} for a in arms
           if folds and all(a in res[k] and all(c in res[k][a] for c in need)
                            and f"test_{ships(a)}" in res[k][a] for k in folds)}
    for a in out:
        for k in folds:
            for key, v in triage_fold(res[k][a], a, d0, **kw).items():
                out[a].setdefault(key, []).append(v)
    c, s = out.get(claim), out.get(sota)
    wins = ([int(c["missed"][i] <= s["missed"][i]
                 and c["contested_automated"][i] < s["contested_automated"][i])
             for i in range(len(folds))] if c and s else [])
    return {"rows": out, "folds": folds, "sota": sota, "claim_wins": wins,
            "clear": kw.get("clear", TRIAGE_CLEAR), "flag": kw.get("flag", TRIAGE_FLAG),
            "w": kw.get("w", TRIAGE_W)}


def print_two_sided(t: dict, claim: str = CLAIM) -> None:
    if not t or not t["rows"]:
        return
    print(f"\nTWO-SIDED TRIAGE  AI clears {t['clear']:.0%} as negative + flags "
          f"{t['flag']:.0%} as positive; blend w {t['w']:g} (declared); mean over folds")
    print(f"  {'arm':8s} {'automated':>9s} {'missed/1000':>12s} {'false flags/1000':>17s} "
          f"{'contested cleared':>18s} {'contested automated':>20s}")
    for a, r in t["rows"].items():
        tag = "  <- standard of care" if a == t["sota"] else "  <- ours" if a == claim else ""
        print(f"  {a:8s} {_mean(r['automated']):9.0%} {_mean(r['missed_per_1000']):12.1f} "
              f"{_mean(r['false_flags_per_1000']):17.1f} {_mean(r['contested_cleared']):18.3f} "
              f"{_mean(r['contested_automated']):20.3f}{tag}")
    if t["claim_wins"]:
        print(f"  {claim} vs {t['sota']}: no more missed AND fewer contested decided by AI on "
              f"{sum(t['claim_wins'])}/{len(t['claim_wins'])} folds")


def report(run_dir, d0: float | None = None, claim: str = CLAIM,
           n_boot: int = N_BOOT, seed: int = BASE_SEED, verbose: bool = True,
           selection: str = "primary") -> dict:
    """Every table for one training seed of a run, returned as numbers; printed
    too unless `verbose=False` (the notebook shows the plain-words page instead).
    `selection="sensitivity"`: the same tables at the second rule's epochs."""
    if not verbose:
        with contextlib.redirect_stdout(io.StringIO()):
            return report(run_dir, d0, claim, n_boot, seed, verbose=True,
                          selection=selection)
    res = load_run(run_dir, seed, selection)
    if not any(claim in r for r in res.values()):
        print(f"no {claim} result in {run_dir} yet - nothing to report.")
        return None
    if not any(_profile(r) is not None for r in res.values()):
        print(f"no per-field disagreement (test_d_profile) in {run_dir} - nothing to report.")
        return None
    d0 = cohort_d0(run_dir) if d0 is None else d0
    prim = primary(res, claim, n_boot=n_boot)
    folds = prim["folds"]
    n_items = [len(res[k][claim]["test_disagreement"]) for k in folds]
    print(f"Study | {_first(run_dir).name} | seed {seed} | folds {folds} | "
          f"test items per fold {n_items}")

    won, pools = {}, {}
    for title, table in (
            ("PRIMARY  panel disagreement on each report field; Spearman, mean over fields",
             prim),
            ("SECONDARY 1  Spearman(shipped signal, report-level panel disagreement)",
             report_level(res, claim, n_boot=n_boot))):
        print(f"\n{title}, per fold")
        print(f"  {'arm':8s} {'ships':13s}" + "".join(f"{'f' + str(k):>8s}" for k in folds)
              + f"{'mean':>9s}")
        for a, v in table["rho"].items():
            print(f"  {a:8s} {ships(a):13s}" + "".join(f"{x:+8.3f}" for x in v)
                  + f"{_mean(v):+9.3f}")
        pool = pools[table["target"]] = pooled(table)
        print(f"  {claim} minus each arm: per fold, folds won, and pooled over folds "
              f"(95% CI; fold-stratified patient-clustered bootstrap, {n_boot} draws)")
        w = won[table["target"]] = {}
        for r, per in table["contrast"].items():
            ok = [t for t in per if t]
            w[r] = sum(t["delta"] > 0 for t in ok)
            cells = "".join("     n/a" if t is None else f"{t['delta']:+8.3f}" for t in per)
            t = pool.get(r)
            ci = f"   {t['delta']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]" if t else ""
            tag = "  (control)" if r == CONTROL else ""
            print(f"  vs {r:7s}{cells}   won {w[r]}/{len(ok)}{ci}{tag}")

    pf = per_field(res)
    P = len(next(iter(pf.values())))
    print("\nPER FIELD  Spearman on each report field, mean over folds")
    print(f"  {'field':12s}" + "".join(f"{a:>8s}" for a in pf))
    for j, name in enumerate(FIELD_NAMES.get(P, [f"field {j}" for j in range(P)])):
        print(f"  {name[:12]:12s}" + "".join(f"{pf[a][j]:+8.3f}" for a in pf))

    tab = arm_table(res, d0)
    beta = tab["beta"]
    base = _mean(next(iter(tab["rows"].values()))["leak_base"])
    print(f"\nSECONDARY 2-5  mean over folds; contested = d > {d0:g} (declared)")
    print(f"  {'arm':8s} {'semi-partial':>13s} {'contested AUROC':>16s} "
          f"{'leakage@80%':>12s} {'certified' + (f'@b{beta:.2f}' if beta else ''):>15s} "
          f"{'AUC':>7s}")
    for a, row in tab["rows"].items():
        print(f"  {a:8s} {_mean(row['semi_partial']):+13.3f} "
              f"{_mean(row['contested_auroc']):16.3f} {_mean(row['leakage']):12.3f} "
              f"{_mean(row['certified']):15.3f} {_mean(row['auc']):7.3f}")
    print(f"  leakage base rate (no referral model) {base:.3f}; lower is better.")
    if beta is None:
        print("  certified: the confidence router certifies nothing at any beta.")

    srv = served(res, d0)
    print_served(srv, claim)
    srv_v5 = served_priority(res, d0, claim=claim)
    print_served_priority(srv_v5, claim)
    tri = two_sided_triage(res, d0, claim=claim)
    print_two_sided(tri, claim)

    print_overfit_check(overfit_check(res))
    par = auc_parity(res, claim)
    print(f"\nPRECONDITION  AUC parity, {claim} minus each arm, paired over folds "
          f"(95% CI should include 0)")
    for a, t in par.items():
        print(f"  vs {a:7s} {t['delta']:+.4f}  [{t['ci95_lo']:+.4f}, {t['ci95_hi']:+.4f}]")
    return {"primary": prim, "pooled": pools["macro"], "arms": tab, "parity": par,
            "served": srv, "served_v5": srv_v5, "two_sided": tri,
            "won": won["macro"], "report_pooled": pools["report"],
            "report_won": won["report"], "per_field": pf, "d0": d0}


def report_seeds(run_dir, seeds=None, claim: str = CLAIM,
                 n_boot: int = N_BOOT, verbose: bool = True, rivals=RIVALS,
                 target: str = "macro", title: str = "PRIMARY ACROSS SEEDS") -> dict | None:
    """PRIMARY across training seeds: each seed's pooled contrast, their mean, and
    a seed-averaged 95% interval.

    Every seed scores the same test items in the same order, and `_contrast`
    draws its patient resamples from a fixed generator, so draw b is the SAME
    resample in every seed. Averaging the seeds' bootstrap differences draw by
    draw therefore gives the interval of the seed-averaged contrast, with the
    seeds held fixed (variation between the seeds enters through the average,
    not as a separate variance term; the per-seed rows show it directly).
    `rivals` / `target` / `title`: the same read for other arms, or for the
    report-level ranking (`target="report"`)."""
    if not verbose:
        with contextlib.redirect_stdout(io.StringIO()):
            return report_seeds(run_dir, seeds, claim, n_boot, verbose=True,
                                rivals=rivals, target=target, title=title)
    seeds = list(seeds) if seeds is not None else seeds_in(run_dir, claim)
    per_seed, boots = {}, {}
    for s in seeds:
        res = load_run(run_dir, s)
        if not any(claim in r for r in res.values()):
            continue
        prim = _contrast(res, claim, rivals, target, n_boot, 0)
        if len(prim["folds"]) < 1:
            continue
        per_seed[s] = {"pooled": pooled(prim), "folds": prim["folds"],
                       "claim_rho": _mean(prim["rho"][claim]),
                       "won": {r: sum(t["delta"] > 0 for t in per if t)
                               for r, per in prim["contrast"].items()}}
        for r, per in prim["contrast"].items():
            if per and all(t is not None for t in per):
                boots.setdefault(r, {})[s] = np.mean([t["boot"] for t in per], axis=0)
    if not per_seed:
        print(f"no {claim} result for seeds {seeds} in {run_dir}")
        return None
    used = sorted(per_seed, key=lambda s: (s != BASE_SEED, s))
    print(f"\n{title}  {_first(run_dir).name} | seeds {used}")
    print(f"  {claim} {'per-field' if target == 'macro' else 'report-level'} rho by seed: "
          + "  ".join(f"s{s} {per_seed[s]['claim_rho']:.3f}" for s in used))
    print(f"  {'vs':8s}" + "".join(f"{'s' + str(s):>9s}" for s in used)
          + f"{'mean':>9s}  {'seed-averaged 95% CI':>22s}  {'folds won':>9s}  seeds>0")
    pooled_seeds = {}
    for r, by_s in boots.items():
        common = [s for s in used if s in by_s and r in per_seed[s]["pooled"]]
        if not common:
            continue
        deltas = [per_seed[s]["pooled"][r]["delta"] for s in common]
        b = np.mean([by_s[s] for s in common], axis=0)
        lo, hi = np.percentile(b, [2.5, 97.5])
        won = sum(per_seed[s]["won"][r] for s in common)
        n_f = sum(len(per_seed[s]["folds"]) for s in common)
        ci_pos = sum(per_seed[s]["pooled"][r]["lo"] > 0 for s in common)
        pooled_seeds[r] = {"delta": float(np.mean(deltas)), "lo": float(lo), "hi": float(hi),
                           "per_seed": dict(zip(common, map(float, deltas))),
                           "won": won, "n_folds": n_f, "seeds_ci_above_0": ci_pos,
                           "n_seeds": len(common)}
        cells = "".join(f"{per_seed[s]['pooled'][r]['delta']:+9.3f}" if s in common
                        else f"{'n/a':>9s}" for s in used)
        print(f"  {r:8s}{cells}{np.mean(deltas):+9.3f}  [{lo:+.3f}, {hi:+.3f}]"
              f"{'':>6s}{won:>3d}/{n_f:<3d}   {ci_pos}/{len(common)}")
    print("  seeds>0: seeds whose own pooled 95% CI lies above 0.")
    return {"seeds": used, "per_seed": per_seed, "pooled": pooled_seeds}


def ablation_report(run_dir, seeds=None, claim: str = CLAIM, n_boot: int = N_BOOT,
                    reference: str = "P_emd", verbose: bool = True) -> dict | None:
    """The weight-matched ablations (ladder.ABLATIONS) beside `claim` and
    `reference`, for every seed in `run_dir` (a folder or a list of folders: the
    study's folder plus the ablation run's). Per arm: the per-field and
    report-level rankings and test AUC, mean over fold-seed units. Then `claim`
    minus each arm on both rankings, seeds pooled draw by draw (`report_seeds`),
    and on AUC, paired over the units. None when no ablation arm sits beside a
    `claim` result."""
    if not verbose:
        with contextlib.redirect_stdout(io.StringIO()):
            return ablation_report(run_dir, seeds, claim, n_boot, reference, verbose=True)
    seeds = list(seeds) if seeds is not None else seeds_in(run_dir, claim)
    runs = {s: load_run(run_dir, s) for s in seeds}
    present = {a for res in runs.values() for r in res.values() for a in r}
    arms = [a for a in (claim, reference) + ABLATIONS if a in present]
    if claim not in arms or not set(arms) & set(ABLATIONS):
        print(f"no ablation arm beside a {claim} result in {_first(run_dir).name} "
              f"- nothing to report.")
        return None
    units = {a: {} for a in arms}            # {(seed, fold): that fold's results}
    for s, res in runs.items():
        for k, fr in res.items():
            if claim in fr and _profile(fr) is not None:
                for a in arms:
                    if a in fr:
                        units[a][(s, k)] = fr
    print(f"\nWEIGHT-MATCHED ABLATIONS  {_first(run_dir).name} | seeds {seeds}")
    print(f"  {'arm':8s} {'':28s} {'units':>5s} {'per-field':>10s} {'report':>8s} {'AUC':>7s}")
    table = {}
    for a in arms:
        rows = np.array([(_score(*_signal(fr, a, "macro")), _score(*_signal(fr, a, "report")),
                          fr[a]["auc"]) for fr in units[a].values()])
        m = rows.mean(axis=0)
        table[a] = {"units": len(rows), "per_field": float(m[0]), "report": float(m[1]),
                    "auc": float(m[2])}
        print(f"  {a:8s} {DISPLAY_NAMES.get(a, a)[:28]:28s} {len(rows):5d} {m[0]:10.3f} "
              f"{m[1]:8.3f} {m[2]:7.3f}")
    rivals = tuple(a for a in arms if a != claim)
    macro = report_seeds(run_dir, seeds, claim, n_boot, rivals=rivals, target="macro",
                         title="ABLATIONS, per-field ranking")
    rep = report_seeds(run_dir, seeds, claim, n_boot, rivals=rivals, target="report",
                       title="ABLATIONS, report-level ranking")
    print(f"\nABLATIONS, test AUC: {claim} minus each arm, paired over fold-seed units "
          f"(mean, 95% CI)")
    auc = {}
    for a in rivals:
        keys = [u for u in units[a] if u in units[claim]]
        auc[a] = t = M.paired_fold_test([{"v": units[claim][u][claim]["auc"]} for u in keys],
                                        [{"v": units[a][u][a]["auc"]} for u in keys], "v")
        print(f"  vs {a:7s} {t['delta']:+.4f}  [{t['ci95_lo']:+.4f}, {t['ci95_hi']:+.4f}]"
              f"  units {t['n']}")
    return {"arms": table, "per_field": macro, "report": rep, "auc": auc}


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    report(args[0], d0=float(args[1]) if len(args) > 1 else None)
