"""The study's per-fold unit of work: build an arm, train it, evaluate it.

    run_fold       one arm on one fold, through the shared `FoldRunner`
    build_arm      the arm's network and its `field_disagreement.Objective`
    evaluate_arm   everything the study reports, measured on one trained fold
    serving_block  the claim arm's fitted policy, averaged over folds (for reference)

The arms are `field_disagreement.triage.ladder`; the tables are `study_results`. Per-item test
arrays are kept in each result, so every table is rebuilt from the run folder
without a GPU. Nothing here reads a notebook global.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch.utils.data import DataLoader

from field_disagreement import Calibrator, Objective, build_model, decide_batch
from field_disagreement import metrics as M
from field_disagreement import triage
from field_disagreement.triage import conformal as C
from field_disagreement.triage.ladder import TRAINED_ARMS, ladder_config, ships

from . import study_prior_art as X
from . import panel as PANEL
from .evaluate import panel_stats, predict_full
from .report import flatten
from .study import BASE_SEED
from .training import balanced_sampling

__all__ = ["STUDY_SCHEMA", "fold_tag", "run_fold", "build_arm", "evaluate_arm",
           "serving_block"]

# Bump when `evaluate_arm` produces metrics an older result.json cannot contain:
# a cached fold under an older schema is retrained rather than served with
# missing columns. 9 is the schema every arm of the final runs was evaluated under.
STUDY_SCHEMA = 9


def fold_tag(arm: str, fold: int, seed: int = BASE_SEED) -> str:
    """The per-fold cache tag. A replication at another seed gets its own tags;
    seed 42 keeps the plain ones."""
    sd = "" if int(seed) == BASE_SEED else f"_s{int(seed)}"
    return f"{arm}{sd}_fold{fold}"


def run_fold(runner, cfg, experiment, arm: str, fold: int, *, clusters: dict, device):
    """Train (or load from the run folder) and evaluate `arm` on `fold`.

    One stopping floor for EVERY arm: early stopping cannot fire before the
    panel term reaches full weight (defer_warmup x epochs), so the arms differ
    in their objective and in nothing else. Selection is the config's rule
    (`selection:`, here: the smoothed val-AUC band) and reads val AUC only."""
    return runner.run_fold(
        fold, fold_tag(arm, fold, cfg.seed),
        build=lambda ctx: build_arm(ctx, cfg, experiment, arm),
        train_step=lambda ctx: ctx.extras["objective"].train_epoch(ctx),
        evaluate=lambda ctx: evaluate_arm(ctx, cfg, arm, clusters=clusters,
                                          device=device),
        selection=experiment.training.selection,
        min_epochs=int(cfg.defer_warmup * cfg.epochs))


def val_monitor(arm: str, cfg, device):
    """A per-epoch LOG of how well the arm's shipped signal ranks the panel's
    disagreement on VALIDATION: `val_rho_fields` (Spearman per report field,
    macro mean) and `val_rho_d` (with the report-level d). Written into the
    epoch's history row; never read by selection or stopping, which stay on val
    AUC so no arm is picked on the outcome it is scored on. The RNG is restored
    afterwards, so logging cannot change the training draw."""
    from scipy.stats import spearmanr

    from field_disagreement.engine.runner import predict_full as _pf

    def rho(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
            return float("nan")
        return float(spearmanr(a[ok], b[ok])[0])

    def monitor(ctx) -> dict:
        state = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            pred = _pf(ctx.model, ctx.val, device, 0, cfg.seed,
                       extra_keys=("d_profile", "disagreement"))
        finally:
            torch.set_rng_state(state)
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)
        spec = TRAINED_ARMS[arm]
        if spec.target == "opinion_fields" and pred.get("dispersion") is not None:
            pred["dispersion"] = _field_spreads(pred["dispersion"], cfg.n_ratings)
        _drop_aux(pred, arm, cfg)
        col = ships(arm)
        if col == "panel_disp" and pred.get("dispersion") is not None:
            S = np.asarray(pred["dispersion"], float).reshape(len(pred["probs"]), -1)
        elif col == "panel_spread" and pred.get("opinion") is not None:
            S = np.asarray(triage.opinion_spread(pred["opinion"]), float)[:, None]
        elif col == "epistemic" and pred.get("dispersion") is not None:
            S = np.asarray(pred["dispersion"], float).reshape(len(pred["probs"]), -1)[:, :1]
        else:
            p = np.asarray(pred["probs"], float).ravel()
            S = (1 - 2 * np.abs(p - 0.5))[:, None]
        out = {"val_rho_d": rho(S.mean(1), pred.get("disagreement"))}
        D = pred.get("d_profile")
        if D is not None:
            D = np.asarray(D, float).reshape(len(S), -1)
            out["val_rho_fields"] = float(np.nanmean(
                [rho(S[:, j if S.shape[1] == D.shape[1] else 0], D[:, j])
                 for j in range(D.shape[1])]))
        return out
    return monitor


def build_arm(ctx, cfg, experiment, arm: str):
    """The arm's network, with its objective put in `ctx.extras["objective"]`."""
    spec = TRAINED_ARMS[arm]
    # ONE pass over train, then one over val, before the model is built. Every
    # loader pass draws from the RNG, so these two passes fix the draw the model
    # starts from: keep them, or a retrain no longer reproduces the paper's runs.
    # The train pass
    # collects the training targets `rank_gauss` scores against.
    trv = {}
    balanced = balanced_sampling(cfg)
    # Under the balanced sampler the train pass is a class-reweighted DRAW; the
    # rank reference must be the training SPLIT, so it is read once in file order.
    ref_pass = (DataLoader(ctx.train.dataset, batch_size=ctx.train.batch_size,
                           shuffle=False) if balanced else ctx.train)
    for b in ref_pass:
        for k in ("disagreement", "d_single", "d_profile", "d_diag", "opinion_fields", "label"):
            if k in b:
                trv.setdefault(k, []).append(b[k].numpy())
    trv = {k: np.concatenate(v) for k, v in trv.items()}
    for _ in ctx.val:
        pass
    # sorted per column, so a per-field target (n, P) ranks each field on its own
    r_ref = (np.sort(trv[spec.target], axis=0).astype(np.float32)
             if spec.transform != "none" else None)
    arm_cfg = ladder_config(experiment, arm)
    # #neg / #pos over the training items: the consensus loss's own pos_weight,
    # which the extension arms' losses reuse
    y_tr = trv["label"]
    pos_weight = (1.0 if balanced else
                  float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0))
    if spec.cls_target == "random_reader":
        ctx.extras["objective"] = X.RandomReaderObjective(
            arm_cfg, cls_loss=X.soft_target_bce(pos_weight), r_ref=r_ref)
    elif spec.head == "evidential":
        ctx.extras["objective"] = X.EvidentialObjective(
            arm_cfg, cls_loss=None, r_ref=r_ref, pos_weight=pos_weight)
    else:
        ctx.extras["objective"] = Objective(arm_cfg, cls_loss=ctx.loss_fn, r_ref=r_ref)

    # Heads are built PER ARM: an arm without an opinion objective has no opinion
    # head, so it can never be scored on an untrained one.
    # the per-field arm gets one dispersion output per report field
    width = (trv["d_profile"].shape[1] if spec.target == "d_profile" else
             int(np.prod(trv["opinion_fields"].shape[1:])) if spec.target == "opinion_fields"
             else spec.regress)
    if spec.aux_fields:                 # + the P x K auxiliary field histograms
        width += int(np.prod(trv["opinion_fields"].shape[1:]))
    if spec.head == "evidential":
        width = 2                       # the two evidences, read by EvidentialNet
    model = build_model(cfg.model, pretrained=cfg.pretrained,
                        opinion_head=spec.opinion != "none",
                        n_ratings=cfg.n_ratings, dispersion_head=width)
    if spec.head == "evidential":
        model = X.EvidentialNet(model)
    ctx.extras["val_monitor"] = val_monitor(arm, cfg, ctx.device)
    if cfg.mc_passes > 0 and not triage.has_dropout(model):
        print(f"  [warn] {cfg.model} has no active dropout - MC sampling will give "
              f"T identical passes and an epistemic term of exactly 0.")
    return model


def _signals(pred, qhat, keep=None):
    """A split's predictions -> the (N, F) policy input. `keep` pins the columns
    to a FITTED policy's own `signal_names`."""
    return triage.prediction_signals(
        prob=pred["probs"], q_hat=qhat,
        defer_score=pred["defer"][:, 2] if pred["defer"] is not None else None,
        opinion_entropy=pred["aleatoric"], epistemic=pred["epistemic"],
        drop_constant=True, keep=keep)


def _field_spreads(logits, k: int) -> np.ndarray:
    """(N, P*K) per-field logits -> (N, P) s.d. of each field's predicted histogram."""
    z = np.asarray(logits, float)
    z = z.reshape(z.shape[0], -1, int(k))
    q = np.exp(z - z.max(-1, keepdims=True))
    q /= q.sum(-1, keepdims=True)
    n, p, _ = q.shape
    return np.asarray(triage.opinion_spread(q.reshape(n * p, -1)), float).reshape(n, p)


def _drop_aux(pred, arm: str, cfg) -> None:
    """An `aux_fields` arm's head is [P dispersions | P x K field histograms]; it
    routes on, and is scored by, the P dispersions alone, so the rest is cut."""
    d = pred.get("dispersion")
    if TRAINED_ARMS[arm].aux_fields and d is not None:
        d = np.asarray(d)
        pred["dispersion"] = d[..., : d.shape[-1] // (1 + int(cfg.n_ratings))]


def _centred(ctx, model, device, cfg, arm: str) -> dict:
    """The test split again with NO translation: the robustness read of the
    evaluation geometry (the primary reads the fixed per-file offset)."""
    import copy
    ds = copy.copy(ctx.test.dataset)
    if float(getattr(ds, "shift_frac", 0.0)) <= 0.0:
        return {}
    ds.shift_frac = 0.0
    dl = DataLoader(ds, batch_size=ctx.test.batch_size, shuffle=False,
                    num_workers=0)
    pred = predict_full(model, dl, device, 0, cfg.seed)
    if TRAINED_ARMS[arm].head == "evidential" and pred.get("dispersion") is not None:
        pred["epistemic"] = np.asarray(pred.pop("dispersion"), float).ravel()
        pred["dispersion"] = None
    if TRAINED_ARMS[arm].target == "opinion_fields" and pred.get("dispersion") is not None:
        pred["dispersion"] = _field_spreads(pred["dispersion"], cfg.n_ratings)
    _drop_aux(pred, arm, cfg)
    return pred


def evaluate_arm(ctx, cfg, arm: str, *, clusters: dict, device) -> dict:
    """Everything the study reports, on one trained fold.

    Three things are fitted on VALIDATION, frozen and applied to test, for every
    arm alike: the temperature (rank-preserving - only ECE / NLL / Brier move),
    the conformal LAC threshold, and the learned deferral policy (fitted against
    the model's own errors, never against annotator disagreement)."""
    model = ctx.model
    val = predict_full(model, ctx.val, device, cfg.mc_passes, cfg.seed)
    test = predict_full(model, ctx.test, device, cfg.mc_passes, cfg.seed)
    s3 = predict_full(model, ctx.loaders["score3"], device, cfg.mc_passes, cfg.seed)
    if TRAINED_ARMS[arm].head == "evidential":
        # the evidential arm's vacuity travels in `dispersion`; it is its
        # model-uncertainty term, so it is scored as `epistemic`
        for split in (val, test, s3):
            if split.get("dispersion") is not None:
                split["epistemic"] = np.asarray(split.pop("dispersion"), float).ravel()
                split["dispersion"] = None
    if TRAINED_ARMS[arm].target == "opinion_fields":
        # the per-field soft-label arm's head is P x K logits: its per-field
        # signal is each field's predicted SPREAD, as B_kl / C_emd route on theirs
        for split in (val, test, s3):
            if split.get("dispersion") is not None:
                split["dispersion"] = _field_spreads(split["dispersion"], cfg.n_ratings)
    for split in (val, test, s3):
        _drop_aux(split, arm, cfg)
    # PATIENT CLUSTERS, in loader order: every interval resamples patients.
    test_cl = [clusters.get(str(f)) for f in ctx.loaders["splits"]["test"]]
    s3_cl = [clusters.get(str(f)) for f in ctx.loaders["splits"]["score3"]]
    d0 = float(cfg.contested_d)
    neutral = getattr(cfg, "neutral_rating", None)

    temperature = M.fit_temperature(val["labels"], val["probs"])
    for split in (val, test, s3):
        split["probs"] = M.apply_temperature(split["probs"], temperature)

    y, p = test["labels"], test["probs"]
    qhat = C.lac_calibrate(C.binary_probs(val["probs"]), val["labels"], cfg.alpha)
    sig_val, names = _signals(val, qhat)
    policy = triage.fit_deferral_policy(sig_val, names, val["probs"], val["labels"],
                                        coverage=cfg.target_coverage)

    # classification + calibration
    res = {"arm": arm, "schema": STUDY_SCHEMA, "temperature": float(temperature),
           "arm_spec": dataclasses.asdict(TRAINED_ARMS[arm]),
           "auc": M.roc_auc(y, p), "ece": M.ece(y, p),
           "brier": M.brier(y, p), "nll": M.nll(y, p)}

    # the contested panels (d > d0) and the rest
    hard = np.asarray(test["disagreement"], dtype=float) > cfg.contested_d
    res["hard_n"] = int(hard.sum())
    res.update(flatten("hard", M.subgroup_metrics(y, p, hard)))
    res.update(flatten("easy", M.subgroup_metrics(y, p, ~hard)))

    # opinion distribution: placement (EMD), shape (KL), spread (entropy)
    if test["opinion"] is not None:
        res.update(M.opinion_metrics(test["opinion"], test["q_true"],
                                     test["disagreement"]))

    # what a model with NO opinion head could emit instead, fitted on train panels
    q_tr, y_tr = panel_stats(ctx.train)
    if len(q_tr):
        fitted = M.fit_opinion_baselines(q_tr, y_tr)
        for kind in ("prior", "class_conditional"):
            Qb = M.predict_opinion_baseline(p, fitted, kind)
            res.update(flatten(f"base_{kind}", M.opinion_metrics(
                Qb, test["q_true"], test["disagreement"])))
        Qcc = M.predict_opinion_baseline(p, fitted, "class_conditional")
        res["test_decoder_spread"] = np.asarray(
            triage.opinion_spread(Qcc), float).ravel().tolist()

    # conformal
    conf = C.evaluate(C.binary_probs(p), y,
                      C.binary_probs(val["probs"]), val["labels"], alpha=cfg.alpha)
    res["conf_coverage"] = conf["empirical_coverage"]
    res["conf_set_size"] = conf["avg_set_size"]
    res["conf_qhat"] = qhat

    # the SERVED decision path
    dep = decide_batch(test, policy=policy,
                       calibrator=Calibrator(q_hat=qhat, alpha=cfg.alpha))
    res["deployed_rule"] = dep["rule"]
    res["deployed_rate"] = float(dep["deferred"].mean())
    res["deployed_forced_empty"] = int(dep["forced_by_empty_set"].sum())
    res.update(flatten("deployed", M.error_concentration(y, p, dep["deferred"])))

    # model risk: every deferral rule against the confidence baseline
    rules = triage.candidate_rules(
        prob=p, q_hat=qhat,
        defer_score=test["defer"][:, 2] if test["defer"] is not None else None,
        opinion_entropy=test["aleatoric"], epistemic=test["epistemic"],
        policy=policy)
    for name, metrics in M.compare_selectors(y, p, rules).items():
        res.update(flatten(name, metrics))
    sig_test, _ = _signals(test, qhat, keep=policy.signal_names)
    res.update(flatten("policy", M.error_concentration(y, p, policy.decide(sig_test))))

    # per-item test arrays: every table is rebuilt from these
    res["test_probs"] = np.asarray(p, float).ravel().tolist()
    res["test_labels"] = np.asarray(y, float).ravel().tolist()
    res["test_disagreement"] = np.asarray(test["disagreement"], float).ravel().tolist()
    res["test_clusters"] = [str(c) for c in test_cl]
    res["test_files"] = [str(f) for f in ctx.loaders["splits"]["test"]]
    # the panel's disagreement on each report field, and a per-field head's own
    # prediction of it: the per-field primary is rebuilt from these
    prof = [b["d_profile"].numpy() for b in ctx.test if "d_profile" in b]
    if prof:
        res["test_d_profile"] = np.concatenate(prof).astype(float).tolist()
    if test.get("dispersion") is not None and np.ndim(test["dispersion"]) == 2:
        res["test_panel_disp_fields"] = np.asarray(test["dispersion"], float).tolist()
    for name, score in PANEL.panel_signals(test, q_hat=qhat).items():
        res[f"test_{name}"] = np.asarray(score, float).ravel().tolist()
    if test.get("q_true") is not None:
        res["test_reader_accuracy"] = M.expected_reader_accuracy(
            test["q_true"], y, neutral=neutral).ravel().tolist()

    # the same test items with no translation (evaluation-geometry robustness);
    # the temperature is rank-preserving, so it is applied as on the primary pass
    cen = _centred(ctx, model, device, cfg, arm)
    if cen:
        res["test_centred_probs"] = np.asarray(
            M.apply_temperature(cen["probs"], temperature), float).ravel().tolist()
        for name, score in PANEL.panel_signals(cen).items():
            res[f"test_centred_{name}"] = np.asarray(score, float).ravel().tolist()
        if cen.get("dispersion") is not None and np.ndim(cen["dispersion"]) == 2:
            res["test_centred_panel_disp_fields"] = np.asarray(cen["dispersion"], float).tolist()

    # the borderline (score-3) items, where the cohort has them
    if s3["disagreement"] is not None and len(s3["disagreement"]):
        res["s3_disagreement"] = np.asarray(s3["disagreement"], float).ravel().tolist()
        # per field: the median-3 nodules are scored on the per-field primary too
        s3_prof = [b["d_profile"].numpy() for b in ctx.loaders["score3"] if "d_profile" in b]
        if s3_prof:
            res["s3_d_profile"] = np.concatenate(s3_prof).astype(float).tolist()
        if s3.get("dispersion") is not None and np.ndim(s3["dispersion"]) == 2:
            res["s3_panel_disp_fields"] = np.asarray(s3["dispersion"], float).tolist()
        res["s3_clusters"] = [str(c) for c in s3_cl]
        for name, vals in PANEL.panel_signals(s3).items():
            res[f"s3_{name}"] = np.asarray(vals, float).ravel().tolist()
        res.setdefault("s3_panel_spread", None)

    # the fitted policy, printed by `serving_block` for reference
    res["policy"] = {**policy.to_dict(round_to=None),
                     "coefficients": policy.coefficients()}

    # panel risk: ranking, semi-partial, leakage, team accuracy, and the
    # certificate, which calibrates on a patient-disjoint half of TEST (the
    # checkpoint is early-stopped on val, so val is not exchangeable)
    res.update(PANEL.panel_battery(
        test, s3, tau=d0, coverage=cfg.target_coverage, q_hat=qhat,
        clusters=s3_cl, neutral=neutral,
        alpha_err=float(getattr(cfg, "cert_alpha", 0.10)),
        betas=tuple(getattr(cfg, "cert_betas", (0.05, 0.10, 0.15, 0.20, 0.30))),
        delta=float(getattr(cfg, "cert_delta", 0.05)),
        seed=cfg.seed, test_clusters=test_cl))
    return res


def serving_block(per_fold: list[dict], cfg, arm: str) -> None:
    """Print `arm`'s fitted deferral policy, averaged over folds, FOR REFERENCE.
    Coefficients are on the standardised scale, so their magnitudes read as what
    drove the deferrals. It is not the triage rule the paper evaluates: it never
    reads the disagreement head. Skipped, with the reason, when the folds fitted
    different signal sets."""
    policies = [triage.DeferralPolicy.from_dict(f["policy"]) for f in per_fold]
    try:
        block = triage.DeferralPolicy.average(policies)
    except ValueError as exc:
        print(f"no serving block - the folds did not fit the same signals: {exc}")
        return
    if block.weights is not None:
        return
    print(f"what drove the deferrals ({arm}, standardised coefficients, + = defer):")
    for name, w in block.coefficients().items():
        print(f"  {name:18s} {w:+.3f}")
    print("\n--- reference only: not the triage rule the paper evaluates ---")
    print(f"conformal_qhat: {np.mean([f['conf_qhat'] for f in per_fold]):.6f}")
    print(f"alpha: {cfg.alpha}")
    print(f"mc_passes: {cfg.mc_passes}")
    print("defer_policy:")
    for k, v in block.to_dict().items():
        print(f"  {k}: {v}")
