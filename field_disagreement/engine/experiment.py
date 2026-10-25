"""One study, end to end: folds in, per-fold results out.

`ExperimentRunner` closed the "reusable components, no reusable training" gap
for a SINGLE fold: hand it a model and two loaders and it owns the epoch loop.
What it does not own is everything around that loop - building the model the
config describes, composing the objective the config switches on, constructing
the optimiser and schedule, doing it again for every fold, and loading the best
weights back to evaluate. This class does all of it:

    experiment = TriageExperiment(config, folds, store=store, device=device)
    results    = experiment.fit()          # one dict per fold
    experiment.calibrate()                 # q_hat + the deferral policy, on val

so the caller never touches a loss or the conformal arithmetic.

`folds` is the one thing this cannot supply, for the same reason `fit` does not
take a dataset: the cohort, its storage layout and the frozen per-subject split
are DOMAIN facts, and a framework that knew them would only work on the cohort
it was written against. So a caller passes any object with

    len(folds)              how many folds
    folds.loaders(i)        {"train": ..., "val": ..., "test": ...}
    folds.pos_weight(i)     optional; class balance over fold i's train split

`backend/notebooks/shared/runner.py` is this project's implementation of it.

`train_step` and `evaluate` are overridable because a study that adds a term of
its own is a real thing this project does: replacing the step is how a
contribution stays readable in the notebook that argues for it. Everything else
follows from the config, which is the point - an arm is a
`with_contributions(...)` and nothing else. Subclass to change the rest.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from ..config import ExperimentConfig
from ..metrics import roc_auc
from ..objectives import Objective
from ..registry import build_model
from ..triage import fit_deferral_policy, prediction_signals
from .calibration import DEFAULT_CLASS_NAMES, Calibrator
from .optim import (amp_components, build_cls_loss, build_optimizer,
                    build_scheduler, set_seed)
from .runner import ExperimentRunner, Fold, predict, predict_full

__all__ = ["TriageExperiment", "FoldSource"]


@runtime_checkable
class FoldSource(Protocol):
    """The domain half of a study: how many folds, and what each one loads."""

    def __len__(self) -> int: ...

    def loaders(self, fold: int) -> dict: ...


class TriageExperiment:
    """The composed study: config + folds -> trained, evaluated, servable."""

    def __init__(self, config, folds, *, store, device, tag: str = "run",
                 train_step=None, evaluate=None, tau_d: float = 0.5,
                 selection: str = "auc", min_epochs: int = 0):
        self.config = ExperimentConfig.of(config)
        self.folds = folds
        self.store = store
        self.device = device
        self.tag = tag
        self.tau_d = tau_d
        self.selection = selection
        self.min_epochs = min_epochs
        self._train_step = train_step
        self._evaluate = evaluate
        self.runner = ExperimentRunner(self.config, store, device)
        # Written by `calibrate`, read by `pipeline`. Unfitted until then, which
        # is the honest state: a `Calibrator` with no `q_hat` collapses C(x) to
        # the argmax singleton and reports `alpha=None` rather than implying a
        # coverage guarantee nobody measured.
        self.calibrator = Calibrator()
        self.policy = None

    def fold_tag(self, fold: int) -> str:
        """The checkpoint key for one fold - resume state, best weights, result."""
        return f"{self.tag}_f{fold}"

    def build_model(self, fold: int):
        """The network this config describes.

        The auxiliary heads follow from the contributions rather than from a
        separate switch: an arm with no opinion term must not carry an opinion
        head, or the ablation compares two networks as well as two objectives.
        """
        c = self.config
        return build_model(c.model.backbone, pretrained=c.model.pretrained,
                           feat_stage=c.model.feat_stage,
                           n_ratings=c.model.n_ratings,
                           opinion_head=c.opinion.enabled,
                           defer_head=c.deferral.enabled)

    def train_step(self, ctx: Fold):
        """One epoch. The standard contribution set needs no study-specific step."""
        if self._train_step is not None:
            return self._train_step(ctx)
        return ctx.extras["objective"].train_epoch(ctx)

    def evaluate_fold(self, ctx: Fold) -> dict:
        """What this fold scored, once the best weights are back.

        The default is deliberately thin, because what a study REPORTS is the
        study's own claim.
        """
        if self._evaluate is not None:
            return self._evaluate(ctx)
        probs, labels, files = predict(ctx.model, ctx.test, self.device)
        return {"test_auc": roc_auc(labels, probs), "n_test": len(files)}

    def build_fold(self, fold: int) -> Fold:
        """This fold's loaders, loss, model, optimiser, schedule and objective."""
        set_seed(self.config.seed)
        loaders = self.folds.loaders(fold)

        pos_weight = getattr(self.folds, "pos_weight", None)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight(fold), device=self.device)

        ctx = Fold(cfg=self.config, fold=fold, tag=self.fold_tag(fold),
                   device=self.device, loaders=loaders,
                   loss_fn=build_cls_loss(self.config, pos_weight))
        ctx.model = self.build_model(fold).to(self.device)
        ctx.opt = build_optimizer(ctx.model, self.config)
        ctx.sched = build_scheduler(ctx.opt, self.config)
        ctx.use_amp, ctx.scaler = amp_components(self.device, self.config)
        ctx.extras["objective"] = Objective(self.config, cls_loss=ctx.loss_fn,
                                            tau_d=self.tau_d)
        return ctx

    def fit_fold(self, fold: int) -> dict:
        """Build, train, restore the best weights, evaluate, release."""
        ctx = self.build_fold(fold)
        best_auc = self.runner.fit(ctx, self.train_step,
                                   selection=self.selection,
                                   min_epochs=self.min_epochs)
        self.store.load_best(ctx.tag, ctx.model)

        # The settings travel with the numbers, or a result read later is scored
        # against whatever the YAML says by then.
        result = {"fold": fold, "best_val_auc": best_auc,
                  "selection": (ctx.selection.to_dict()
                                if ctx.selection is not None else None),
                  **self.evaluate_fold(ctx), "config": self.config.to_dict()}
        self.runner.release(ctx)
        return result

    def fit(self) -> list[dict]:
        """Every fold, in order. Returns one result per fold."""
        return [self.fit_fold(fold) for fold in range(len(self.folds))]

    def evaluate(self) -> list[dict]:
        """Re-run the evaluation on the stored best weights, without training.

        The operation you want after adding a metric: the weights are the ones
        selection already chose, so re-scoring must not cost another run and
        must not be able to change them.
        """
        out = []
        for fold in range(len(self.folds)):
            ctx = self.build_fold(fold)
            self.store.load_best(ctx.tag, ctx.model)
            out.append({"fold": fold, **self.evaluate_fold(ctx)})
            self.runner.release(ctx)
        return out

    # -- the step between training and serving ------------------------------ #
    def calibrate(self, *, split: str = "val",
                  class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
                  coverage: float | None = None) -> tuple[Calibrator, object]:
        """Fit the conformal threshold and the deferral policy on held-out data.

        The step between `fit` (weights) and `pipeline` (which consumes a
        calibrator): collect held-out predictions, fit `q_hat`, fit the policy,
        so nothing reaches the pipeline as a float pasted by hand.

        Every fold contributes its OWN validation split, scored by its own
        member, which is what keeps the calibration honest: no point here was
        trained on by the member that predicted it. Because the folds are
        subject-disjoint, concatenating them covers the cohort once.

        One conservatism is deliberate and worth stating rather than hiding: the
        calibration probabilities come from SINGLE members, while the served
        pipeline pools every member. An ensemble is generally better separated
        than its members, so a threshold fitted on member predictions errs
        towards larger prediction sets — coverage lands at or above 1 - alpha,
        never below it. The alternative, calibrating the pooled ensemble, has no
        split that is held out from all of its members at once.

        No temperature scaling is applied. Rescaling logits would move `q_hat`
        and the served path does not rescale, so fitting it here would calibrate
        one system and deploy another.
        """
        c = self.config
        mc = c.uncertainty.samples if c.uncertainty.mc_dropout else 0
        parts = []
        for fold in range(len(self.folds)):
            ctx = self.build_fold(fold)
            self.store.load_best(ctx.tag, ctx.model)
            loader = ctx.loaders[split]
            parts.append(predict_full(ctx.model, loader, self.device,
                                      mc_passes=mc, seed=c.seed))
            self.runner.release(ctx)

        def stack(key):
            """Concatenate a signal only if EVERY fold produced it — an arm that
            built no opinion head must not be calibrated on one fold's stub."""
            vals = [p[key] for p in parts]
            return None if any(v is None for v in vals) else np.concatenate(vals)

        probs, labels = stack("probs"), stack("labels")
        if probs is None or not len(probs):
            raise ValueError(
                f"the {split!r} split produced no predictions to calibrate on")

        self.calibrator = Calibrator.fit(
            probs, labels, alpha=c.conformal.alpha, class_names=class_names,
            split=split, n_folds=len(self.folds), mc_passes=mc)

        # The policy's input, built through `deferral_inputs`, so every caller
        # builds the same columns.
        defer = stack("defer")
        signals, names = prediction_signals(
            prob=probs, q_hat=self.calibrator.q_hat,
            defer_score=None if defer is None else defer[:, 2],
            opinion_entropy=stack("aleatoric"), epistemic=stack("epistemic"),
            drop_constant=True)
        # Fitted against the model's OWN errors, never against annotator
        # disagreement — that is what separates it from the supervised head.
        self.policy = fit_deferral_policy(
            signals, names, probs, labels,
            coverage=(c.deferral.target_coverage if coverage is None
                      else coverage))
        return self.calibrator, self.policy
