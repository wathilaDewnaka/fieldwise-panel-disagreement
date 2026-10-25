"""The DOMAIN half of a training run: everything the framework loop must not know.

`field_disagreement.engine.ExperimentRunner` owns the loop - resume, epochs, validation,
model selection, early stopping, checkpointing. It is handed a model and two
loaders and never asks where they came from, which is what keeps it free of the
manifest, the patient-keyed fold assignment and the .npz loader factory that
`backend/tests/test_core_purity.py` holds the framework to knowing nothing about.

Building those IS this file. `FoldRunner` adds two things to the framework loop:
`run_fold`, the cached unit of work the notebook calls.

    runner = FoldRunner(cfg, CKPT, device, df, patient_fold, schema=STUDY_SCHEMA)
    result = runner.run_fold(fold, tag,
                             build=..., train_step=..., evaluate=...,
                             selection="auc_band",
                             min_epochs=int(cfg.defer_warmup * cfg.epochs))

`build`, `train_step` and `evaluate` are the study's own - the model it wants,
the objective it composes, and what it measures once the best weights are back.
All three are handed the same `Fold` context, and `Fold.extras` is where `build`
leaves anything the other two need (the arm's objective).
"""
from __future__ import annotations

import time

import torch

from field_disagreement.engine import ExperimentRunner, Fold, set_seed

from .session import check_budget, record_unit_time

from .training import (
    amp_components,
    build_cls_loss,
    build_optimizer,
    build_scheduler,
    loaders_for,
    pos_weight_for,
)



__all__ = ["CohortFolds", "Fold", "FoldRunner"]


class CohortFolds:
    """This project's `field_disagreement.FoldSource`, for `TriageExperiment`.

    The façade owns dataset -> folds -> model -> objective -> training ->
    evaluation, but not the first arrow: which folder the processed cohort is
    in, which subjects the frozen split puts in each fold, and a fold's class
    balance are facts about THIS cohort.

        folds = CohortFolds(cfg, df, patient_fold)
        experiment = TriageExperiment(experiment_cfg, folds,
                                        store=CKPT, device=device)

    The last fold's loaders are held: `pos_weight` is asked for the fold
    `loaders` was just built for, and rebuilding would re-read the whole split.
    """

    def __init__(self, cfg, df, patient_fold, *, n_folds: int | None = None):
        self.cfg = cfg
        self.df = df
        self.patient_fold = patient_fold
        self.n_folds = int(getattr(cfg, "n_splits", 5)
                           if n_folds is None else n_folds)
        self._cached: tuple[int, dict] | None = None

    def __len__(self) -> int:
        return self.n_folds

    def loaders(self, fold: int) -> dict:
        if self._cached is None or self._cached[0] != fold:
            self._cached = (
                fold, loaders_for(self.cfg, self.df, self.patient_fold, fold))
        return self._cached[1]

    def pos_weight(self, fold: int) -> float:
        """#neg / #pos over this fold's TRAIN split, for the weighted BCE."""
        return pos_weight_for(self.df, self.loaders(fold)["splits"]["train"], self.cfg)


def fold_seed(base: int, fold: int) -> int:
    """This fold's seed. Depends on the FOLD only - never on run order.

    THE BUG THIS FIXES. `set_seed` used to be called once, in the notebook,
    before any training, so the RNG stream was global and sequential: what a
    fold was initialised with depended on how many folds had ALREADY trained in
    that session. A cached fold trains nothing and consumes nothing, so resuming
    a run silently handed everything after it a different draw. Re-ordering the
    loop does NOT fix this - fold-major has the same problem, because the cause
    is the skipped consumption, not the sequence.

    WHY THE ARMS SHARE IT. Every arm gets this same seed, so within a fold the
    arms start from IDENTICAL weights and see the same augmentation and shuffle
    order. The only thing that differs is the loss, which is what the contrast
    claims to measure - this is the common-random-numbers trick, and it removes
    initialisation from the contrasts instead of leaving it in as noise. Folds still differ from each other, so the
    fold-to-fold spread keeps reporting init variance honestly.

    `1000 *` rather than `+ fold` so seed 42/fold 4 and seed 44/fold 2 cannot
    collide on 46."""
    return int(base) + 1000 * int(fold)


class FoldRunner(ExperimentRunner):
    """One fold, from a cache check to a cached result.

    `schema` is the stale-cache guard `CheckpointStore.cached_result` takes -
    bump it in the notebook whenever the evaluation starts producing metrics an
    older result.json cannot contain, or a re-run silently reloads the old
    numbers and every new column prints as "-".
    """

    def __init__(self, cfg, store, device, df, patient_fold, *, schema=None,
                 experiment=None):
        super().__init__(cfg, store, device)
        self.df = df
        self.patient_fold = patient_fold
        self.schema = schema
        # The typed `ExperimentConfig` this run trains, written into every
        # result. Weights and the settings that produced them then travel
        # together, instead of the settings being whatever the YAML said the
        # next time somebody looks.
        self.experiment = experiment

    # Set True to re-read kept weights instead of training. Default False:
    # every existing study trains exactly as before.
    evaluate_only = False

    def run_fold(self, fold, tag, *, build, train_step, evaluate,
                 selection: str = "auc", min_epochs: int = 0,
                 is_fresh=None) -> dict:
        """Cache check -> build -> fit -> best weights -> evaluate -> cache.

        `is_fresh(result) -> bool` is an OPTIONAL extra check on a cached
        result, for a study whose freshness question the `schema` counter does
        not answer. It is only consulted for results that already passed the
        schema check, and returning False re-runs the fold.
        """
        cached = self.store.cached_result(tag, self.schema)
        if cached is not None and (is_fresh is None or is_fresh(cached)):
            print(f"[{tag}] already complete -> using cached result")
            return cached

        # RE-SCORE, no training. For the one case that keeps recurring: the
        # weights are right and the EVALUATION was wrong or incomplete, so the
        # honest thing is to read the kept epoch again rather than train a new
        # model and report it as the same run. Resuming will not do this -
        # `save_ckpt` fires on improvement, so an early-stopped fold resumes
        # `patience` epochs short of done and retrains them, which can move the
        # kept epoch and change arms whose numbers were already valid.
        if self.evaluate_only:
            if not self.store.path(tag, "best.pt").exists():
                raise FileNotFoundError(
                    f"[{tag}] evaluate_only=True but there are no kept weights "
                    f"to re-score. Train this unit first, or turn it off.")
            stale = self.store.cached_result(tag) or {}
            ctx = self.build_fold(fold, tag, build)
            self.store.load_best(tag, ctx.model)
            print(f"[{tag}] EVAL-ONLY -> kept weights restored, no training")
            result = {"fold": fold,
                      "best_val_auc": stale.get("best_val_auc", float("nan")),
                      "selection": stale.get("selection"),
                      # The re-score is recorded IN the result: a number that
                      # came from a re-read must never be indistinguishable
                      # from one that came from a fresh training run.
                      "rescored": True,
                      **evaluate(ctx)}
            if self.schema is not None:
                result.setdefault("schema", self.schema)
            if self.experiment is not None:
                result.setdefault("config", self.experiment.to_dict())
            self.store.cache_result(tag, result)
            self.release(ctx)
            return result

        # A hosted session's time limit (Kaggle): never START a unit that cannot
        # finish inside it. A no-op unless a deadline was set by the stager.
        check_budget(self.store, tag)
        started = time.monotonic()

        ctx = self.build_fold(fold, tag, build)
        best_auc = self.fit(ctx, train_step, selection=selection,
                            min_epochs=min_epochs)
        self.store.load_best(tag, ctx.model)

        # `selection` travels with the numbers. Without it, "was this arm's
        # checkpoint ever trained under the objective the arm is named for" can
        # only be answered by reconstructing the run from its stdout, which is
        # gone the moment a Colab session ends.
        result = {"fold": fold, "best_val_auc": best_auc,
                  "selection": (ctx.selection.to_dict()
                                if ctx.selection is not None else None),
                  "history": list(getattr(ctx, "history", None) or []),
                  **evaluate(ctx)}
        # The SECOND selection rule tracked in the same run (`sel_also`), scored
        # the same way from its own kept weights - or noted as the same epoch,
        # in which case its numbers ARE the primary ones.
        alt = getattr(ctx, "alt_selection", None)
        if alt is not None and alt.epoch:
            sens = {"selection": alt.to_dict()}
            if alt.epoch == (result["selection"] or {}).get("epoch"):
                sens["same_epoch"] = True
            else:
                self.store.load_best(tag, ctx.model, kind="band.pt")
                sens.update(evaluate(ctx))
            result["sensitivity"] = sens
        if self.schema is not None:
            result.setdefault("schema", self.schema)
        if self.experiment is not None:
            result.setdefault("config", self.experiment.to_dict())
        self.store.cache_result(tag, result)
        record_unit_time(self.store, tag, time.monotonic() - started)
        self.release(ctx)
        return result

    def build_fold(self, fold, tag, build) -> Fold:
        """This fold's loaders, class-weighted loss, model, optimiser, schedule.

        The two domain-coupled steps live here and nowhere else: `loaders_for`
        reads the processed .npz folder through the frozen split, and
        `pos_weight_for` reads the manifest for this fold's class balance.
        """
        cfg = self.cfg
        # Reseed PER FOLD, so this fold is the same draw whether it runs
        # first, last, or after a resume, and so that the arms share it within
        # a fold. See `fold_seed`.
        set_seed(fold_seed(getattr(cfg, "seed", 0), fold))
        loaders = loaders_for(cfg, self.df, self.patient_fold, fold)
        pos_weight = torch.tensor(
            pos_weight_for(self.df, loaders["splits"]["train"], cfg), device=self.device)

        ctx = Fold(cfg=cfg, fold=fold, tag=tag, device=self.device,
                   loaders=loaders, loss_fn=build_cls_loss(cfg, pos_weight))
        ctx.model = build(ctx).to(self.device)
        ctx.opt = build_optimizer(ctx.model, cfg)
        ctx.sched = build_scheduler(ctx.opt, cfg)
        ctx.use_amp, ctx.scaler = amp_components(self.device, cfg)
        return ctx
