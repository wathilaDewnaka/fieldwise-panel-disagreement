"""The panel-risk battery: separation, leakage, team and certificate measurements.

One module rather than notebook cells, so the evaluation has one definition of
"split panel" wherever it is called.

What it measures, and why each one is here:

  `panel_signals`      the scores a router may rank on, including the PANEL
                       SPREAD read off the opinion head. That read-out is the
                       prediction of the cohort's own `disagreement` statistic;
                       the entropy is a different functional of the same
                       histogram and cannot see where the mass sits.

  `dispersion_report`  THE PRIMARY. Does the predicted dispersion rank the
                       panel's actual dispersion, over the whole test set, per
                       fold? Everything else here is measured on a subset or
                       pooled across folds, and cannot carry an every-fold
                       claim.

  `separation_report`  agreed vs GENUINELY divided borderline panels, at the
                       cohort's declared `contested_d` - plus the
                       threshold-free rank correlation, so the claim does not
                       rest on any cut-off, and the paired contrast against
                       confidence on the same items.

  `leakage_report`     of the cases the rule auto-decides, how many would the
                       panel have split on. The operational form of the claim
                       and the column a confidence-only router cannot optimise.

  `team_report`        accuracy of the human-AI team, pricing a referral by
                       whether ONE reader could have settled it (the panel
                       supplies that per case).

  `certificate_report` risk control over the auto-decided set: error alone, and
                       error AND panel division together, each at a declared
                       level. This is the intended guarantee; split
                       conformal gives marginal set coverage, which is a
                       different statement (see `triage.risk_control`).

Every function takes arrays and returns a flat dict of floats, so results stay
JSON-serialisable and poolable across folds.
"""
from __future__ import annotations

import numpy as np

from field_disagreement import metrics as M
from field_disagreement import triage
from field_disagreement.triage import conformal as C
from field_disagreement.triage import risk_control as RC

__all__ = ["panel_signals", "dispersion_report", "separation_report",
           "leakage_report", "team_report", "certificate_report",
           "residual_panel_signal",
           "split_for_calibration",
           "panel_battery"]


def panel_signals(pred: dict, q_hat: float | None = None,
                  policy=None) -> dict:
    """{name: defer score} for one split. Higher always means REFER.

    A signal whose head this arm never built is absent rather than constant:
    an arm cannot be scored on a head it does not have, and a column that never
    moves would read as "measured, no effect"."""
    out: dict = {}
    p = np.asarray(pred["probs"], dtype=float).ravel()
    out["confidence"] = triage.confidence_gap(p)
    if pred.get("defer") is not None:
        out["defer_head"] = np.asarray(pred["defer"], dtype=float)[:, 2]
    if pred.get("dispersion") is not None:
        # THE CONTRIBUTION's signal: a direct estimate of E[d | x]. Higher means
        # a more divided panel, which is the same direction as every other
        # column here (higher = REFER).
        disp = np.asarray(pred["dispersion"], dtype=float)
        # a per-field head (N, P) routes the CASE on its mean over the fields
        out["panel_disp"] = (disp.mean(axis=1) if disp.ndim == 2 else disp).ravel()
    if pred.get("opinion") is not None:
        # THE PANEL-RISK READ-OUT: predicted s.d. of the rating histogram, the
        # same functional as the cohort's `disagreement` column, so it is a
        # prediction of the supervised quantity rather than a correlate of it.
        out["panel_spread"] = triage.opinion_spread(pred["opinion"])
    if pred.get("aleatoric") is not None and float(np.std(pred["aleatoric"])) > 0:
        out["opinion_entropy"] = np.asarray(pred["aleatoric"], dtype=float).ravel()
    if pred.get("epistemic") is not None and float(np.std(pred["epistemic"])) > 0:
        out["epistemic"] = np.asarray(pred["epistemic"], dtype=float).ravel()
    if q_hat is not None:
        sets = C.lac_predict(C.binary_probs(p), q_hat)
        out["conformal"] = triage.conformal_ambiguity(sets.sum(axis=1))
    return out


def separation_report(disagreement, signals: dict, tau: float,
                      clusters=None, seed: int = 0) -> dict:
    """The falsification test, at the DECLARED threshold plus threshold-free.

    `tau` is the cohort's `contested_d`, fixed in the cohort file before any of
    this ran. The
    historical `nonzero` split is reported beside it as a sensitivity check, not
    as the headline: with 4 raters on a 1-5 scale one rater one step out already
    counts as "divided", which puts most of the near-agreed middle on the
    divided side and pulls every signal towards 0.5.

    `clusters` (patient ids) makes the intervals cluster bootstraps, which is
    what they must be when one subject contributes several nodules."""
    d = np.asarray(disagreement, dtype=float).ravel()
    out: dict = {"n_s3": int(d.size)}
    if d.size < 8 or not signals:
        return out

    tight = M.disagreement_separation(d, signals, split="threshold", tau=tau,
                                      clusters=clusters, seed=seed)
    loose = M.disagreement_separation(d, signals, split="nonzero",
                                      clusters=clusters, seed=seed)
    out["n_divided"] = tight["n_divided"]
    out["n_united"] = tight["n_united"]
    for name, block in tight["signals"].items():
        out[f"s3_{name}_auc"] = block["auc"]
        out[f"s3_{name}_lo"] = block["ci95_lo"]
        out[f"s3_{name}_hi"] = block["ci95_hi"]
    for name, block in loose["signals"].items():
        # WITH its interval. The historical split is a sensitivity check, and a
        # sensitivity check without a CI cannot say whether it disagrees: on
        # LIDC the loose split lands at 0.556 [0.476, 0.634] (a null) where the
        # declared one lands at 0.683 [0.570, 0.780], and the two bounds are
        # the whole reason the threshold was declared in the cohort file.
        out[f"s3any_{name}_auc"] = block["auc"]
        out[f"s3any_{name}_lo"] = block["ci95_lo"]
        out[f"s3any_{name}_hi"] = block["ci95_hi"]

    # Threshold-free, and paired against confidence on the same items: the form
    # of the result that survives "you chose the cut-off that worked".
    ordered = {"confidence": signals["confidence"], **{k: v for k, v in
                                                       signals.items()
                                                       if k != "confidence"}}
    rho = M.rank_agreement(d, ordered, clusters=clusters, seed=seed)
    for name, block in rho["signals"].items():
        out[f"rho_{name}"] = block.get("rho", float("nan"))
        if "vs_confidence" in block:
            out[f"rho_{name}_vs_conf"] = block["vs_confidence"]
            out[f"rho_{name}_vs_conf_lo"] = block["vs_confidence_lo"]
            out[f"rho_{name}_vs_conf_hi"] = block["vs_confidence_hi"]
    return out


def residual_panel_signal(signals: dict, disagreement, clusters=None,
                          n_boot: int = 2000, seed: int = 0) -> dict:
    """THE CONTROL that replaces the permuted-dispersion arm: does the panel
    signal know anything about panel division that model confidence does not?

    Semi-partial Spearman - rank-transform everything, regress both the signal
    and the panel's dispersion on confidence, and correlate what is left. The
    answer is the part of the panel claim that cannot be re-described as the
    classifier being unsure.

    This exists because the trained control it replaces could not have
    falsified anything. `G_shuffle` permuted the dispersions and then built the
    deferral target as `where(d_permuted > tau, 2, y_true)`, so TWO of the three
    classes stayed genuine and the third logit went on absorbing class
    ambiguity, which is itself correlated with how divided a panel is. Its floor
    was rho ~ 0.30, not chance, and on LIDC fold 0 it scored 0.960 on the
    separation test - higher than the properly supervised arm. A control that
    cannot come back negative is worse than no control, because it gets
    reported as one.

    The confound that control was aimed at is exactly what a semi-partial
    correlation removes, and it removes it without a second training run: if a
    panel signal ranks dispersion ONLY because hard cases are both contested and
    near the decision boundary, then with confidence residualised out it goes to
    zero. If it stays positive, the head is carrying panel information that no
    amount of model-risk estimation supplies - which is the two-risk claim
    stated as a measurement.

    `clusters` (patient ids) makes the interval a cluster bootstrap."""
    from scipy.stats import rankdata

    d = np.asarray(disagreement, dtype=float).ravel()
    out: dict = {}
    if "confidence" not in signals or d.size < 12:
        return out
    conf = np.asarray(signals["confidence"], dtype=float).ravel()

    def semipartial(idx):
        c = rankdata(conf[idx]).astype(float)
        c = c - c.mean()
        den = float(c @ c)
        if den <= 0:
            return {}
        yy = rankdata(d[idx]).astype(float)
        yy = yy - yy.mean()
        yr = yy - c * (float(c @ yy) / den)
        res = {}
        for name, v in signals.items():
            if name == "confidence":
                continue
            x = rankdata(np.asarray(v, dtype=float).ravel()[idx]).astype(float)
            x = x - x.mean()
            xr = x - c * (float(c @ x) / den)
            nx, ny = float(np.linalg.norm(xr)), float(np.linalg.norm(yr))
            res[name] = float(xr @ yr / (nx * ny)) if nx > 0 and ny > 0 else 0.0
        return res

    base = semipartial(np.arange(d.size))
    if not base:
        return out
    rng = np.random.default_rng(seed)
    cl = None if clusters is None else np.asarray(clusters).ravel()
    draws: dict = {k: [] for k in base}
    for _ in range(n_boot):
        if cl is not None and cl.size == d.size:
            uniq = np.unique(cl)
            pick = rng.choice(uniq, size=uniq.size, replace=True)
            idx = np.concatenate([np.flatnonzero(cl == u) for u in pick])
        else:
            idx = rng.integers(0, d.size, d.size)
        if np.unique(d[idx]).size < 3:
            continue
        for k, v in semipartial(idx).items():
            draws[k].append(v)
    for k, v in base.items():
        out[f"sp_{k}"] = float(v)
        arr = np.asarray(draws[k], dtype=float)
        if arr.size >= 50:
            out[f"sp_{k}_lo"] = float(np.percentile(arr, 2.5))
            out[f"sp_{k}_hi"] = float(np.percentile(arr, 97.5))
    return out


def dispersion_report(signals: dict, disagreement) -> dict:
    """THE PRIMARY: does a signal rank the panel's own dispersion, on the whole
    test set rather than on the borderline subset?

    Per fold, which the pooled agreed-vs-divided contrast can never be - that
    one has ~100 items a fold and a dozen united panels among them, so a
    per-fold interval on it is meaningless and the claim has to be pooled. This
    is every test item, so "positive on every fold" is a statement the design
    can actually support.

    Reported against two baselines, both of which are available to a system
    WITHOUT a panel head: the arm's own class-conditional decoder (in
    `opinion_metrics`, as `spread_corr`) and plain confidence, here."""
    d = np.asarray(disagreement, dtype=float).ravel()
    out: dict = {}
    for name, score in signals.items():
        s = np.asarray(score, dtype=float).ravel()
        if s.size == d.size and float(np.std(s)) > 0:
            out[f"rho_test_{name}"] = float(M.spearman(s, d))
    return out


def leakage_report(signals: dict, disagreement, d0: float,
                   coverage: float = 0.8) -> dict:
    """Contested leakage and referral enrichment for every rule, at one coverage.

    Every rule auto-decides the same NUMBER of cases, so the columns compare
    directly: same workload, different cases."""
    out: dict = {}
    for name, score in signals.items():
        block = M.contested_leakage(score, disagreement, d0, coverage)
        out[f"leak_{name}"] = block["leakage"]
        out[f"enrich_{name}"] = block["enrichment"]
    out["leak_base"] = float((np.asarray(disagreement, float).ravel() > d0).mean())
    return out


def team_report(y, p, signals: dict, q_true, neutral: int | None = None,
                coverage: float = 0.8) -> dict:
    """Human-AI team accuracy per rule, with the referral priced by whether one
    reader could have settled the case."""
    out: dict = {}
    if q_true is None:
        return out
    acc = M.expected_reader_accuracy(q_true, y, neutral=neutral)
    out["reader_accuracy_mean"] = float(np.mean(acc))
    for name, score in signals.items():
        block = M.team_accuracy(y, p, score, acc, coverage=coverage)
        out[f"team_{name}"] = block["team_accuracy"]
        out[f"teamref_{name}"] = block["referred_reader_accuracy"]
    return out


def split_for_calibration(test: dict, clusters=None, frac: float = 0.5,
                          seed: int = 0) -> tuple[dict, dict]:
    """Cut the TEST split in two, patient-disjoint: one half calibrates the
    certificate, the other is what it is reported on.

    This replaces calibrating on `val`, which was wrong, and whose wrongness is
    visible in the run it produced. The checkpoint is EARLY-STOPPED on val AUC,
    so val is the split the model was SELECTED on and its error rate is
    optimistically biased; a threshold tuned there does not transfer. On LIDC
    fold 0 the error-only certificate promised risk <= 0.05 and the realised
    error among the cases it auto-decided was 0.096 / 0.087 / 0.128 / 0.091
    across arms - four of six above the bound, where a valid certificate is
    breached at most delta = 5% of the time. That is a broken guarantee, not an
    unlucky fold.

    The old code defended `val` by arguing that temperature is rank-preserving
    and sign-preserving at 0.5. That is true and it is not the problem: model
    selection is. Test is the only split in this protocol the model neither
    trained on nor was chosen on, so the certificate is calibrated inside it and
    reported on the remainder. Nothing else in the battery is touched - the
    dispersion, separation and leakage tables keep the WHOLE test split, because
    none of them fits a threshold and none can leak through one.

    The cut is by PATIENT when `clusters` is given: two nodules from one patient
    landing on opposite sides would leak the calibration half into the
    evaluation half through shared anatomy. A `clusters` of the wrong length
    RAISES rather than falling back to an item-level split - a silent fallback
    here would quietly weaken the guarantee while still printing one, and the
    caller's own `clusters` argument is aligned to the borderline subset, not to
    test, so the mismatch is a live hazard rather than a hypothetical."""
    n = len(np.asarray(test["labels"]).ravel())
    rng = np.random.default_rng(seed)
    if clusters is None:
        m = np.zeros(n, dtype=bool)
        m[rng.permutation(n)[:max(1, int(round(frac * n)))]] = True
    else:
        cl = np.asarray(clusters).ravel()
        if cl.size != n:
            raise ValueError(
                f"clusters has {cl.size} entries for {n} test items - the "
                "calibration split must be cut by patient, and a list aligned "
                "to a different split cannot do it")
        uniq = np.unique(cl)
        pick = rng.permutation(uniq)[:max(1, int(round(frac * len(uniq))))]
        m = np.isin(cl, pick)

    def take(mask):
        out = {}
        for k, v in test.items():
            arr = np.asarray(v)
            out[k] = arr[mask] if arr.ndim and arr.shape[0] == n else v
        return out

    return take(m), take(~m)


def certificate_report(cal: dict, test: dict, *, alpha_err: float = 0.05,
                       betas=None, delta: float = 0.05, d0: float = 1.0,
                       rules=("confidence", "panel_spread", "panel_disp",
                              "epistemic", "defer_head")) -> dict:
    """Calibrate on `cal`, apply to `test`: how much can be automated and under
    what guarantee - reported as a FRONTIER over the panel constraint rather
    than at one declared beta.

    A single beta made the study's headline hostage to a constant that cannot be
    justified from outside the cohort, and on LIDC it was set to 0.10 against a
    contested base rate of 0.194. That asks the router to purify the
    auto-decided set roughly fourfold before the binomial bound is consulted at
    all, so every router - ours and the baseline alike - reported coverage 0.000
    and the table distinguished nothing. Sweeping beta asks what the constant
    was meant to: over the range where automation is possible at all, does
    routing on predicted panel dispersion certify more of it than routing on
    confidence?

    Three things per run:

      error-only          one risk, the classical selective guarantee, ranked by
                          confidence. What a selective-prediction paper reports.
      dual, confidence    error AND panel division, both carried by confidence:
                          the baseline has to satisfy a panel constraint with a
                          signal that ranks panel division near chance.
      dual, panel_spread  error by confidence, panel division by the predicted
                          dispersion. The contribution, stated operationally.

    The headline operating point comes from `RC.headline_beta`: the SMALLEST
    beta at which the BASELINE is feasible. Letting the baseline choose the
    point is what stops the comparison being taken where our own router happens
    to look best, and the whole frontier is reported beside it either way."""
    out: dict = {}
    if not cal or not test:
        return out
    betas = tuple(betas) if betas is not None else RC.DEFAULT_BETAS

    def risks(block):
        y = np.asarray(block["labels"]).astype(int).ravel()
        pr = np.asarray(block["probs"], dtype=float).ravel()
        return (((pr >= 0.5).astype(int) != y).astype(float),
                (np.asarray(block["disagreement"], dtype=float).ravel() > d0
                 ).astype(float))

    err_c, pan_c = risks(cal)
    err_t, pan_t = risks(test)
    cal_sig, test_sig = cal["signals"], test["signals"]
    if "confidence" not in cal_sig:
        return out
    out["cert_n_cal"] = int(len(err_c))
    out["cert_base_contested"] = float(pan_c.mean())
    out["cert_base_error"] = float(err_c.mean())

    single = RC.calibrate(cal_sig["confidence"], err_c, alpha=alpha_err,
                          delta=delta)
    out.update(_cert_row("err_only", single, {"error": test_sig["confidence"]},
                         err_t, pan_t))

    # THE WHOLE SWEEP, for every rule and every beta, applied to the held-out
    # half. Two reasons it is done this way rather than only at a chosen point.
    #
    # (1) `aggregate_folds` reads its key list from FOLD 0 and then indexes every
    #     fold, so a key that exists on one fold and not another either vanishes
    #     from the report or raises. Emitting the certificate only where it was
    #     feasible does exactly that - feasibility varies by fold - so every
    #     beta emits a row, and an infeasible one is coverage 0.0 rather than a
    #     missing key. Coverage 0.0 is also the honest operational reading: the
    #     router automates nothing.
    # (2) The operating point must be ONE number for the study. Re-selecting it
    #     per fold and then averaging is a forking path - five folds, five
    #     chances to land on a flattering beta. The caller picks it from the
    #     POOLED baseline frontier after aggregation; `cert_headline_beta` is
    #     kept per fold for reference only and must not be averaged into a
    #     headline.
    for name in rules:
        if name not in cal_sig or name not in test_sig:
            continue
        rows = RC.beta_frontier(cal_sig["confidence"], cal_sig[name],
                                err_c, pan_c, alpha=alpha_err, betas=betas,
                                delta=delta)
        for r in rows:
            b = r["beta"]
            k = f"front_{name}_b{b:.2f}"
            out[f"{k}_coverage"] = r["coverage"]
            out[f"{k}_feasible"] = r["feasible"]
            cert = RC.dual_calibrate(
                {"error": cal_sig["confidence"], "panel": cal_sig[name]},
                {"error": err_c, "panel": pan_c},
                {"error": alpha_err, "panel": float(b)}, delta=delta)
            applied = _cert_row(f"{k}", cert,
                                {"error": test_sig["confidence"],
                                 "panel": test_sig[name]}, err_t, pan_t)
            out[f"{k}_test_error"] = applied[f"{k}_test_error"]
            out[f"{k}_test_contested"] = applied[f"{k}_test_contested"]
            out[f"{k}_test_coverage"] = applied[f"{k}_coverage"]

    base_rows = [{"beta": b,
                  "feasible": int(out.get(f"front_confidence_b{b:.2f}_feasible", 0)),
                  "coverage": float(out.get(f"front_confidence_b{b:.2f}_coverage", 0.0))}
                 for b in betas]
    hb = RC.headline_beta(base_rows)
    out["cert_headline_beta"] = float(hb) if hb is not None else float("nan")
    return out


def _cert_row(tag: str, cert, test_scores: dict, err, pan) -> dict:
    """One certificate applied to test: what it promised, and what it delivered.

    Both are reported. The promise is the guarantee; the delivery is a sample
    and can be worse than the promise on any single fold without the guarantee
    being wrong - it is a statement about repeated draws, and reporting only the
    realised number would quietly turn it into a claim about this test set."""
    keep = RC.apply_certificate(cert, test_scores)
    n = int(keep.sum())
    nan = float("nan")
    return {f"{tag}_feasible": int(bool(cert.feasible)),
            f"{tag}_cal_coverage": float(cert.coverage),
            f"{tag}_coverage": float(keep.mean()) if keep.size else nan,
            f"{tag}_bound_error": float(cert.bound.get("error", nan)),
            f"{tag}_bound_panel": float(cert.bound.get("panel", nan)),
            f"{tag}_test_error": float(np.asarray(err)[keep].mean()) if n else nan,
            f"{tag}_test_contested": float(np.asarray(pan)[keep].mean()) if n else nan}


def panel_battery(test: dict, s3: dict, *, tau: float, coverage: float = 0.8,
                  q_hat: float | None = None, clusters=None,
                  neutral: int | None = None, cal: dict | None = None,
                  alpha_err: float = 0.05, betas=None,
                  delta: float = 0.05, seed: int = 0,
                  test_clusters=None) -> dict:
    """The whole panel-risk report for one trained fold, as a flat dict.

    `tau` is the cohort's declared `contested_d`, used for BOTH the
    agreed-vs-divided contrast and the definition of a contested case in the
    leakage and certificate columns - one number, declared in the cohort file,
    so the three tables cannot silently disagree about what "split" means.

    Two cluster arguments, because they index different splits and confusing
    them is silent: `clusters` is the patient id of each BORDERLINE item (it
    makes the separation bootstrap a cluster bootstrap), `test_clusters` is the
    patient id of each TEST item (it makes the certificate's calibration cut
    patient-disjoint)."""
    y = np.asarray(test["labels"]).astype(int).ravel()
    p = np.asarray(test["probs"], dtype=float).ravel()
    sig = panel_signals(test, q_hat=q_hat)
    d = np.asarray(test["disagreement"], dtype=float).ravel()

    out: dict = {}
    out.update(dispersion_report(sig, d))          # THE PRIMARY, per fold
    # A DIAGNOSTIC, not a claim, and no bar is set on it: how much daylight is
    # there between the two estimators of the same quantity? A tie on `rho_test`
    # is NOT evidence that they are one signal - on the LIDC CPU probe these
    # correlate 0.187 with each other while both rank `d`. Without this number a
    # reader has no way to tell "equivalent" from "different, scoring alike".
    if "panel_disp" in sig and "panel_spread" in sig:
        out["rho_readout_pair"] = float(M.spearman(sig["panel_disp"],
                                                   sig["panel_spread"]))
    # THE CONTROL: the same ranking with confidence residualised out, so a
    # panel signal cannot claim credit for the classifier being unsure.
    out.update(residual_panel_signal(sig, d, clusters=test_clusters,
                                     seed=seed))
    out.update(leakage_report(sig, d, tau, coverage))
    out.update(team_report(y, p, sig, test.get("q_true"), neutral=neutral,
                           coverage=coverage))
    # THE SAME SEPARATION TEST, ON THE FULL TEST SPLIT.
    #
    # `separation_report` below runs on the BORDERLINE (s3) subset, which is
    # defined by a NEUTRAL median rating - so it is EMPTY on any K=2 cohort and
    # the falsification bar has simply never been computable on Chákṣu. That
    # makes a cross-cohort claim impossible on the one test the study declares as its
    # falsifier, which is a gap in the report and not a property of the data.
    #
    # Same metric (`M.disagreement_separation`), same declared `tau` from the
    # cohort file, same cluster bootstrap - only the split differs, so nothing
    # here is a new measure and nothing is tuned. Keys are prefixed `sep_` to
    # keep them distinct from the `s3_` block, which is retained unchanged.
    if d.size >= 8 and sig:
        _full = M.disagreement_separation(d, sig, split="threshold", tau=tau,
                                          clusters=test_clusters, seed=seed)
        out["sep_n_divided"] = _full.get("n_divided")
        out["sep_n_united"] = _full.get("n_united")
        for _nm, _blk in _full.get("signals", {}).items():
            out[f"sep_{_nm}_auc"] = _blk["auc"]
            out[f"sep_{_nm}_lo"] = _blk["ci95_lo"]
            out[f"sep_{_nm}_hi"] = _blk["ci95_hi"]
    if s3 and s3.get("disagreement") is not None and len(s3["disagreement"]):
        out.update(separation_report(s3["disagreement"],
                                     panel_signals(s3, q_hat=q_hat),
                                     tau=tau, clusters=clusters, seed=seed))
    # THE CERTIFICATE calibrates inside TEST and reports on the half it did
    # not see. `cal` (the val split) is deliberately ignored: the checkpoint
    # was early-stopped on it, so it is not exchangeable with test and the
    # guarantee quoted from it was not being met - see `split_for_calibration`.
    # NB `clusters` is aligned to the BORDERLINE subset, not to test, so the
    # calibration cut takes `test_clusters` and nothing else.
    c_blk, t_blk = split_for_calibration(test, clusters=test_clusters,
                                         seed=seed)
    out.update(certificate_report(
        {**c_blk, "signals": panel_signals(c_blk, q_hat=q_hat)},
        {**t_blk, "signals": panel_signals(t_blk, q_hat=q_hat)},
        alpha_err=alpha_err, betas=betas, delta=delta, d0=tau))
    return out
