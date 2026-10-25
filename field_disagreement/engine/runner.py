"""The training loop, as part of the framework.

What makes a loop framework-safe is what it is handed rather than what it looks
up. `fit` receives a model and two loaders and never asks where they came from,
so it needs no manifest, no fold assignment and no loader factory - the domain
concepts `backend/tests/test_core_purity.py` keeps out of this package. Building
those is the caller's job, and `backend/notebooks/shared/runner.py` is the
caller that does it for this project's cohorts.

What `fit` deliberately does NOT own is the training step. The composition of
losses IS each contribution, it genuinely differs per study, and hiding it
behind a config switch would make the work harder to read and the ablations
harder to trust. It is passed in, and handed the same `Fold` context every time.

    runner = ExperimentRunner(cfg, store, device)
    best   = runner.fit(fold_ctx, train_step,
                        selection="auc_band", min_epochs=10)

Two settings are arguments because they are the ones that silently diverged when
five copies of this loop existed side by side:

    selection   "auc"      keep the best val-AUC epoch. Arm-neutral, but it can
                          latch onto a single lucky epoch, before the arm's own
                          objective has reached full weight.
                "auc_band" keep the LAST epoch whose SMOOTHED val AUC is within
                           `sel_tol` of the best. This is the rule the study uses.

The band is a PARITY constraint - epochs inside it are classifiers of equal
quality - and taking its LAST member gives each arm the most exposure to its own
objective that its classifier can carry. Nothing but val AUC is ever consulted,
so no outcome metric and no per-arm objective can reach the choice. An epoch
whose val AUC has fallen more than `sel_tol` below the arm's own best is OUT of
the band and cannot be kept.

Both the band and patience read the SMOOTHED series. Anchored on the raw series,
a single spike sets a bar nothing later clears and the band collapses onto it.

    min_epochs  the epoch from which the objective is at FULL weight. 0 = no ramp.

`min_epochs` gates all three of stopping, patience and SELECTION. The last is
the one that was missing: an arm is named for its objective, so a checkpoint
from inside the ramp is a baseline that the result file then labels a treatment.

`fit` records which epoch it kept on `ctx.selection`, and the callers write it
into every result - that provenance did not exist, so the above was only
diagnosable from a training log that outlives no Colab session.

Everything else the loop needs is `field_disagreement.TrainingConfig`: `epochs`,
`patience` and `ckpt_every`. `ExperimentConfig.of` resolves that from
either the typed config or the flat YAML namespace the notebook loads, so
the loop reads one validated block rather than four loose attributes.

`store` is anything with the checkpoint protocol - `resume`, `save_best`,
`save_ckpt` - which is duck-typed rather than imported, because a resumable
store is deployment infrastructure and differs between a lab box, a Colab VM and
a cluster.
"""
from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import torch

from ..config import ExperimentConfig
from ..metrics import roc_auc

SELECTIONS = ("auc", "auc_band", "auc_band_signal")


@dataclass(frozen=True)
class Selection:
    """Which epoch's weights `fit` kept, and what it kept them on.

    `post_warmup` is the question worth asking of every arm before its numbers
    are read: a run whose objective never reached the weights it saved looks,
    in a result.json, exactly like one where it did.

    `epoch = 0` means nothing has been selected yet; `val_auc = -1.0` is below
    every real AUC, so the first epoch always displaces it.
    """

    rule: str = "auc"
    epoch: int = 0                       # 1-based, matching the log line
    val_auc: float = -1.0
    val_auc_smooth: float | None = None
    min_epochs: int = 0                  # 0-based index of the first full epoch
    signal: float | None = None          # smoothed val `sel_signal` (auc_band_signal)

    @property
    def post_warmup(self) -> bool:
        """Were these weights trained with the objective at FULL weight?"""
        return self.epoch > self.min_epochs

    def to_dict(self) -> dict:
        return {**asdict(self), "post_warmup": self.post_warmup}


@dataclass
class Fold:
    """One unit of training, handed to the study's own step and evaluation.

    `extras` is free scratch space shared between them: it is where a caller
    puts anything the step needs that the loop has no business knowing about -
    a calibrated deferral threshold, a preflight gradient norm.
    """

    cfg: Any
    fold: int
    tag: str
    device: Any
    loaders: dict
    loss_fn: Callable
    model: Any = None
    opt: Any = None
    sched: Any = None
    scaler: Any = None
    use_amp: bool = False
    epoch: int = 0
    extras: dict = field(default_factory=dict)
    # Written by `fit`; read by the caller once it returns. NOT in `extras`,
    # which `release` clears and which belongs to the study rather than the loop.
    selection: Selection | None = None
    # One row per trained epoch (loss, val AUC, smoothed val AUC), written by `fit`
    # into every result, so where a run peaked and which epoch it kept can be read
    # after the session is gone.
    history: list = field(default_factory=list)

    @property
    def train(self):
        return self.loaders["train"]

    @property
    def val(self):
        return self.loaders["val"]

    @property
    def test(self):
        return self.loaders["test"]


# --------------------------------------------------------------------------- #
# the two passes the loop itself needs
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, loader, device):
    """Sigmoid probabilities, labels and item ids over a loader.

    Classification only - the auxiliary heads are not read here, because this is
    what model selection and early stopping are scored on.
    """
    model.eval()
    probs, labels, files = [], [], []
    for batch in loader:
        p = torch.sigmoid(model(batch["image"].to(device)))
        probs.append(p.cpu().numpy())
        labels.append(batch["label"].numpy())
        files.extend(batch["file"])
    if not probs:
        return np.array([]), np.array([]), []
    return np.concatenate(probs), np.concatenate(labels), files


@torch.no_grad()
def predict_full(model, loader, device, mc_passes: int = 0, seed: int = 0,
                 extra_keys: tuple[str, ...] = ()) -> dict:
    """Every predicted signal the triage path reads, over a loader.

    `predict` above is deliberately classification-only, because that is what
    model selection is scored on. Calibration needs more: the conformal
    threshold needs the probabilities, and the deferral policy needs the panel
    entropy and the epistemic term beside them. Collecting those was previously
    only possible outside the package, which meant the framework could train a
    system and serve one but not calibrate the step between.

    `mc_passes > 0` runs T stochastic passes with dropout left on (BatchNorm
    stays in eval, so batch composition cannot leak into the "uncertainty") and
    splits their mutual information into

        aleatoric   spread each sample predicts INSIDE the panel -> irreducible
        epistemic   disagreement BETWEEN the samples             -> model unsure

    With `mc_passes = 0` there is one deterministic pass, aleatoric is the
    opinion head's own entropy and epistemic is exactly 0 — honest rather than
    missing, since one deterministic model has no spread to measure.

    An arm with NO opinion head fills the same two columns from its classifier
    instead (the binary split under MC, zeros without), so the columns change
    meaning across arms. `uncertainty_source` in the result names which one it
    is: mc_opinion | mc_class | opinion_entropy | none.

    A head the arm never built is absent from the result rather than zero-filled,
    so a thinner arm is never silently scored on an untrained head. Only
    `image` and `label` are required of a batch; `file` is used when present.

    `extra_keys` names further batch fields to carry through unchanged — the
    ANNOTATION targets an evaluation scores against (`opinion_dist`,
    `disagreement`), which the framework itself never reads but which a caller
    would otherwise have to re-iterate the loader to collect. Nothing here
    interprets them; they are concatenated and returned under their own names.
    """
    from ..triage import mc_dropout_predict, opinion_entropy, summarise

    model.eval()
    acc = {k: [] for k in ("probs", "labels", "opinion", "defer",
                           "dispersion", "aleatoric", "epistemic",
                           *extra_keys)}
    files = []
    for batch in loader:
        x = batch["image"].to(device)
        if mc_passes > 0:
            u = summarise(mc_dropout_predict(model, x, n_passes=mc_passes,
                                             seed=seed))
            acc["probs"].append(u["prob"])
            if "opinion" in u:
                acc["opinion"].append(u["opinion"])
                acc["aleatoric"].append(u["opinion_aleatoric"])
                acc["epistemic"].append(u["opinion_epistemic"])
            else:
                # No opinion head: fall back to the binary head's own MI split,
                # so the epistemic signal still exists for the thinner arms.
                acc["aleatoric"].append(u["binary_total"] - u["binary_epistemic"])
                acc["epistemic"].append(u["binary_epistemic"])
            if "defer" in u:
                acc["defer"].append(u["defer"])
            if "dispersion" in u:
                acc["dispersion"].append(u["dispersion"])
        else:
            out = model(x, return_all=True)
            p = torch.sigmoid(out["logit"]).cpu().numpy()
            acc["probs"].append(p)
            acc["epistemic"].append(np.zeros_like(p))
            if "opinion" in out:
                op = out["opinion"].softmax(1).cpu().numpy()
                acc["opinion"].append(op)
                acc["aleatoric"].append(opinion_entropy(op))
            else:
                acc["aleatoric"].append(np.zeros_like(p))
            if "defer" in out:
                acc["defer"].append(out["defer"].softmax(1).cpu().numpy())
            if "dispersion" in out:
                acc["dispersion"].append(out["dispersion"].cpu().numpy())
        acc["labels"].append(batch["label"].numpy())
        for key in extra_keys:
            acc[key].append(batch[key].numpy())
        if "file" in batch:
            files.extend(batch["file"])

    out = {k: (np.concatenate(v) if v else None) for k, v in acc.items()}
    out["files"] = files
    # WHICH distribution `aleatoric` / `epistemic` were split over. They are only
    # comparable between two arms with the same source: under MC sampling an arm
    # with no opinion head falls back to the CLASSIFIER's binary split, so its
    # `aleatoric` is not a panel quantity even though it fills the same column.
    has_opinion = out["opinion"] is not None
    out["uncertainty_source"] = (
        ("mc_opinion" if has_opinion else "mc_class") if mc_passes > 0
        else ("opinion_entropy" if has_opinion else "none"))
    return out


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
class ExperimentRunner:
    """Resume, epochs, validation, selection, early stopping, checkpointing."""

    def __init__(self, cfg, store, device):
        self.cfg = cfg
        # The loop reads the TYPED training block and nothing else, so the
        # settings it runs under are validated once here rather than read as
        # four loose attributes only ever checked by the run failing. `of`
        # accepts the flat namespace the notebook loads.
        self.training = ExperimentConfig.of(cfg).training
        self.store = store
        self.device = device

    def fit(self, ctx: Fold, train_step, *, selection: str = "auc",
            min_epochs: int = 0) -> float:
        """Train one unit to completion. Returns the best validation AUC.

        Best-so-far weights are written by `store.save_best` as they are found,
        so the caller restores them with `store.load_best` once this returns.
        """
        train = self.training
        if selection not in SELECTIONS:
            raise ValueError(
                f"unknown selection {selection!r} (known: {list(SELECTIONS)})")

        tag = ctx.tag
        start, best_auc, since, extra = self.store.resume(
            tag, ctx.model, ctx.opt, ctx.sched, ctx.scaler)
        keep, stall_ref = self._resume_selection(
            extra, selection, best_auc, start, min_epochs)
        ctx.selection = keep
        # The val-AUC smoothing window, restored from the checkpoint when there is
        # one. A resume that lost it would re-arm patience and silently train
        # longer, so it rides in `extra` alongside `keep`.
        sm = (extra or {}).get("smooth", {}) if isinstance(extra, dict) else {}
        win_auc = list(sm.get("win_auc", []))
        # The signal rule's state, and the SECOND rule tracked alongside
        # (`sel_also`, weights in band.pt), restored the same way.
        win_sig = list(sm.get("win_sig", []))
        sig_ref = float(sm.get("sig_ref", float("-inf")))
        also = train.sel_also
        alt = None
        if also is not None:
            a0 = (extra or {}).get("alt") if isinstance(extra, dict) else None
            alt = Selection(rule=also, min_epochs=min_epochs)
            if a0:
                alt = replace(alt, epoch=a0["epoch"], val_auc=a0["val_auc"],
                              val_auc_smooth=a0.get("val_auc_smooth"),
                              signal=a0.get("signal"))
        ctx.alt_selection = alt
        uses_sig = "auc_band_signal" in (selection, also)
        ctx.history = (list(extra.get("history", [])) if isinstance(extra, dict)
                       else [])

        # A UNIT THAT ALREADY FINISHED MUST NOT TRAIN AGAIN. The patience test
        # sits at the END of the loop body, so a resumed run whose `since` was
        # already at the limit used to train exactly ONE more epoch before
        # stopping - and that epoch can displace `keep`. Resume was therefore not
        # idempotent, and re-running a completed study to pick up a NEW EVAL
        # METRIC (which is the normal reason to re-run one) silently moved the
        # selected checkpoint on every early-stopped fold while leaving the folds
        # that ran to the cap alone. Returning here makes that re-run a pure
        # re-score: `run_fold` goes straight to `load_best` and `evaluate`.
        if start >= train.epochs or (since >= train.patience and start > min_epochs):
            print(f"[{tag}] already trained to completion "
                  f"(epoch {start}/{train.epochs}, patience {since}/"
                  f"{train.patience}) -> restoring the kept epoch and "
                  f"re-evaluating only")
            return best_auc

        for ep in range(start, train.epochs):
            ctx.epoch = ep
            loss, note = _step_result(train_step(ctx))
            ctx.sched.step()

            vp, vy, _ = predict(ctx.model, ctx.val, self.device)
            vauc = roc_auc(vy, vp)
            # SMOOTHING. The val-AUC series is smoothed over `sel_window` before
            # it is used for ANYTHING - patience or the selection band - so a single
            # lucky epoch can neither stop the run nor anchor the band. On the raw
            # series one early spike sets a bar no later epoch can clear, and the
            # band closes after a few full-weight epochs.
            w = max(1, self.training.sel_window)
            win_auc = (win_auc + [vauc])[-w:]
            s_auc = sum(win_auc) / len(win_auc)

            row = {"epoch": ep + 1, "loss": float(loss), "val_auc": float(vauc),
                   "val_auc_smooth": float(s_auc)}
            # An optional per-epoch LOG (here: how well the arm's signal ranks the
            # panel's disagreement on val). Recorded only; selection and stopping
            # never read it. Printed on the epoch's one line.
            monitor = ctx.extras.get("val_monitor")
            if callable(monitor):
                row.update(monitor(ctx))
            extra = "".join(f"  {k} {v:.4f}" for k, v in row.items() if k.startswith("val_rho"))
            s_sig = None
            if uses_sig:
                if train.sel_signal not in row:
                    raise ValueError(
                        f"selection 'auc_band_signal' reads {train.sel_signal!r} "
                        f"from the val monitor, which did not log it")
                v = float(row[train.sel_signal])
                win_sig = (win_sig + [v if v == v else 0.0])[-w:]
                s_sig = sum(win_sig) / len(win_sig)
            print(f"[{tag}] ep {ep + 1:02d}/{train.epochs}  loss {loss:.4f}  "
                  f"{note}val_auc {vauc:.4f}  smooth {s_auc:.4f}{extra}")
            ctx.history = [h for h in ctx.history if h["epoch"] <= ep] + [row]

            cand = Selection(rule=selection, epoch=ep + 1, val_auc=vauc,
                             val_auc_smooth=s_auc, min_epochs=min_epochs, signal=s_sig)
            best_auc = max(best_auc, vauc)
            if cand.post_warmup:
                # ONE anchor for two decisions, and it is the SMOOTHED series: the
                # best smoothed val AUC reached with the objective at full weight.
                # Patience asks whether the classifier is still improving; the band
                # asks which epochs are within `sel_tol` of it. A peak from inside
                # the ramp answers neither, which is why `post_warmup` gates both.
                since = 0 if s_auc >= stall_ref else since + 1
                stall_ref = max(stall_ref, s_auc)
                # The signal rule also waits while the signal is still rising
                # inside the band, so a slow head is not stopped mid-learning.
                if selection == "auc_band_signal" and s_sig is not None:
                    if s_sig > sig_ref and s_auc >= stall_ref - train.sel_tol:
                        since = 0
                    sig_ref = max(sig_ref, s_sig)

            if alt is not None:     # the second rule, on the same epoch
                cand2 = replace(cand, rule=also)
                if self._keep(alt, cand2, stall_ref, min_epochs > 0 and not alt.post_warmup):
                    alt = cand2
                    self.store.save_best(tag, ctx.model, kind="band.pt")
                ctx.alt_selection = alt
            improved = self._keep(keep, cand, stall_ref,
                                  min_epochs > 0 and not keep.post_warmup)
            if improved:
                keep = cand
                self.store.save_best(tag, ctx.model)

            if improved or (ep + 1) % train.ckpt_every == 0:
                self.store.save_ckpt(tag, ctx.model, ctx.opt, ctx.sched,
                                     ctx.scaler, ep + 1, best_auc, since,
                                     extra={"keep": keep.to_dict(),
                                            "alt": alt.to_dict() if alt else None,
                                            "stall_ref": stall_ref,
                                            "smooth": {"win_auc": win_auc,
                                                       "win_sig": win_sig,
                                                       "sig_ref": sig_ref},
                                            "history": ctx.history})
            if since >= train.patience and ep >= min_epochs:
                print(f"[{tag}] early stop @ ep {ep + 1}")
                break

        ctx.selection = keep
        if min_epochs and not keep.post_warmup:
            # Only reachable when the run ENDED inside the ramp, since the first
            # full-weight epoch takes the checkpoint over unconditionally.
            print(f"[{tag}] SELECTION WARNING: the run ended at epoch "
                  f"{ctx.epoch + 1}, before the objective reached full weight "
                  f"(epoch {min_epochs + 1}). The kept epoch {keep.epoch} is a "
                  f"provisional baseline, not a measurement of this arm.")
        return best_auc

    # -- model selection ----------------------------------------------------- #
    def _keep(self, keep: Selection, cand: Selection, anchor: float,
              handover: bool) -> bool:
        """Does `cand` displace the epoch being kept?

        Only full-weight epochs are eligible. Inside the ramp the best-AUC epoch
        is held PROVISIONALLY - so a run cut short still has weights to restore -
        and the first full-weight epoch takes over from it (`handover`).

        "auc_band" keeps the LAST epoch whose smoothed val AUC is still within
        `sel_tol` of `anchor` (the best smoothed value so far). Two lines, and
        they are the two things the rule has to do: the band is a parity
        constraint on the classifier, and taking the LAST member of it gives
        each arm the most training its own classifier can carry.

        The band is recomputed every epoch against the running best, so a kept
        epoch that a later peak pushes out of tolerance is displaced rather than
        stranded. There is always at least one member - the epoch that set
        `anchor` sits at the top of its own band.
        """
        if not cand.post_warmup:
            return cand.val_auc >= keep.val_auc          # provisional hold
        if handover:
            return True
        if cand.rule == "auc":
            return cand.val_auc >= keep.val_auc
        tol = self.training.sel_tol

        def in_band(x):
            return x.val_auc_smooth is not None and x.val_auc_smooth >= anchor - tol
        if cand.rule == "auc_band":
            return in_band(cand)
        # auc_band_signal: only in-band epochs; a kept epoch the rising anchor has
        # pushed out of the band gives way; otherwise the higher smoothed signal.
        if not in_band(cand):
            return False
        if not in_band(keep) or keep.signal is None:
            return True
        return (cand.signal if cand.signal is not None else float("-inf")) >= keep.signal

    def _resume_selection(self, extra, selection: str, best_auc: float,
                          start: int, min_epochs: int):
        """`(Selection, stall_ref)` to carry on from, for any vintage of `extra`.

        Mid-run checkpoints written before this shape exist (a bare float, or
        None). Under "auc" that float IS the val AUC, and keeping it is what
        stops the first epoch after a resume from overwriting good weights with
        worse ones; under any other rule the next epoch re-earns the slot.
        """
        keep = Selection(rule=selection, min_epochs=min_epochs)
        if isinstance(extra, dict):
            k = extra["keep"]
            return replace(keep, epoch=k["epoch"], val_auc=k["val_auc"],
                           val_auc_smooth=k.get("val_auc_smooth"),
                           signal=k.get("signal")), extra["stall_ref"]
        if selection == "auc":
            keep = replace(keep, val_auc=float(extra)
                           if isinstance(extra, (int, float)) else best_auc)
        # A resume landing before the ramp ends re-earns its anchor; one landing
        # after it has no record of the anchor, and the global best stands in.
        return keep, (-1.0 if start <= min_epochs else best_auc)

    def release(self, ctx: Fold) -> None:
        """Drop this unit's GPU memory before the next one is built.

        The references must go before `empty_cache`, or the allocator fragments
        and a long multi-arm run OOMs partway through.
        """
        ctx.model = ctx.opt = ctx.sched = ctx.scaler = None
        ctx.loaders = ctx.loss_fn = None
        ctx.extras.clear()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _step_result(out):
    """A training step returns its loss, and optionally a note for the log line.

    The note is where a study puts the schedule values a reader watches during a
    run - a loss ramp, a calibrated threshold. It
    keeps those visible without the loop having to know which objective is on.
    """
    if isinstance(out, tuple):
        loss, note = out
        return float(loss), (f"{note}  " if note else "")
    return float(out), ""
