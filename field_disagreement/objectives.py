"""Training objectives: the classification losses, and the composed one.

`focal_loss` is the plain per-sample alternative to class-weighted BCE.

`Objective` is the whole training loss, assembled from the contributions an
`ExperimentConfig` switches on. Every arm composes its loss here, from one
frozen config, so two arms meant to differ by one term cannot differ by anything
else.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import triage
from .config import ExperimentConfig

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# baseline classification loss (class-weighted BCE / focal)
# --------------------------------------------------------------------------- #
def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Binary focal loss (Lin et al., gamma=2 default).

        L = -alpha_t (1 - p_t)^gamma log p_t

    `logits` and `targets` are (N,) with targets in {0,1}. alpha weights the
    (minority) positive class; set alpha<0 to disable the alpha term."""
    targets = targets.float()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)              # prob of true class
    loss = ce * (1.0 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


# --------------------------------------------------------------------------- #
# the composed training loss
# --------------------------------------------------------------------------- #
@dataclass
class Objective:
    """The training loss for one run: L_cls plus whatever the config enables.

        obj = Objective(arm_config, cls_loss=bce)
        loss, note = obj(model, batch, out, epoch, device, rng=rng)

    `cls_loss` is passed in because class balance is a property of the split,
    which is a domain concept. `tau_d` is the disagreement threshold a model
    WITH the 3-way deferral head is trained against; no arm of the study has
    that head, so for them it is never read.

    The panel term (`deferral`) ramps in from step 0 over a fraction of total
    training; the opinion term has a constant weight.

    `batch` is the `field_disagreement.spectrum` contract, so nothing here knows what the
    images are.
    """

    cfg: ExperimentConfig
    cls_loss: object
    tau_d: float = 0.5
    # this fold's sorted TRAINING regression targets, the reference
    # `deferral.regress_transform: rankgauss` scores against: (n,) or, for the
    # per-field target, (n, P) sorted column by column. None otherwise.
    r_ref: object = None

    def lambdas(self, epoch: int) -> dict:
        """Every auxiliary term's weight at this epoch; 0 when switched off."""
        c = self.cfg
        return {
            "opinion": c.opinion.weight if c.opinion.enabled else 0.0,
            "defer": (triage.defer_lambda(
                epoch, c.deferral.weight, c.deferral.warmup_frac, c.epochs)
                if c.deferral.enabled else 0.0),
        }

    def note(self, lam: dict) -> str:
        """The schedule values worth watching, for the runner's log line.

        Only the terms in play, so a baseline arm's line stays readable rather
        than carrying a zero."""
        c, parts = self.cfg, []
        if c.deferral.enabled:
            parts.append(f"lam_d {lam['defer']:.2f}")
        return " ".join(parts)

    def __call__(self, model, batch, out, epoch: int, device,
                 rng: np.random.Generator | None = None):
        """(total loss, log note) for one batch.

        `out` is the model's `return_all=True` dict, so the caller runs the
        forward pass inside its own autocast context and this only composes."""
        c, lam = self.cfg, self.lambdas(epoch)
        y = batch["label"].to(device)
        loss = self.cls_loss(out["logit"], y.float())
        # Each term's UNWEIGHTED value, detached, for the log only. Recording
        # them touches neither the graph nor any RNG, so the total and its
        # gradients are exactly what they were before the terms were logged.
        terms = {"cls": loss.detach()}

        # THE CIRCULARITY CONTROL (`opinion.panel_shuffle = "all"`). One
        # permutation per batch, applied to EVERY panel target - the histogram
        # and the dispersions together - so each image is trained on another
        # image's whole panel, internally consistent, just not its own.
        panel_perm = None
        if c.opinion.panel_shuffle != "none":
            if rng is None:
                raise ValueError(
                    "opinion.panel_shuffle permutes the batch's panels "
                    "and needs `rng`; pass the epoch-seeded generator.")
            panel_perm = torch.as_tensor(rng.permutation(y.shape[0]),
                                         device=y.device, dtype=torch.long)

        if c.opinion.enabled:
            q_target = batch["opinion_dist"].to(device)
            # "disp" leaves each image its OWN histogram; only the dispersions move
            if panel_perm is not None and c.opinion.panel_shuffle == "all":
                q_target = q_target[panel_perm]
            op_loss = triage.opinion_loss(c.opinion.loss, out["opinion"], q_target)
            terms["opinion"] = op_loss.detach()
            loss = loss + lam["opinion"] * op_loss

        if c.deferral.enabled:
            d = batch["disagreement"].to(device)
            rt = c.deferral.regress_target
            r_tgt = d if rt == "disagreement" else batch[rt].to(device)
            if panel_perm is not None:
                d = d[panel_perm]
                r_tgt = r_tgt[panel_perm]
            if c.deferral.regress_dispersion and rt == "opinion_fields":
                # THE PER-FIELD SOFT-LABEL BASELINE: one rating histogram per
                # field, each under the opinion loss, averaged over the fields.
                B, P, K = r_tgt.shape
                logits = out["dispersion"].reshape(B * P, K)
                disp_loss = triage.opinion_loss(c.opinion.loss, logits,
                                                r_tgt.reshape(B * P, K))
                terms["dispersion"] = disp_loss.detach()
                loss = loss + lam["defer"] * disp_loss
            elif c.deferral.regress_dispersion:
                # MSE on the dispersion itself: the head estimates E[d | x],
                # a consistent ranker of the quantity the router sorts by. No
                # threshold, so `tau_d` is not used here.
                disp_out = out["dispersion"]
                if c.deferral.aux_fields:
                    # the head is [P dispersions | P x K field histograms]; the
                    # histograms are an auxiliary target under the opinion loss.
                    # Only the whole-panel shuffle moves them (as `opinion_dist`).
                    B, P = r_tgt.shape
                    h_tgt = batch["opinion_fields"].to(device)
                    if panel_perm is not None and c.opinion.panel_shuffle == "all":
                        h_tgt = h_tgt[panel_perm]
                    K = h_tgt.shape[-1]
                    aux_loss = triage.opinion_loss(
                        c.opinion.loss, disp_out[:, P:].reshape(B * P, K),
                        h_tgt.reshape(B * P, K))
                    terms["fields"] = aux_loss.detach()
                    loss = loss + lam["defer"] * aux_loss
                    disp_out = disp_out[:, :P]
                if c.deferral.regress_transform == "rankgauss":
                    if self.r_ref is None:
                        raise ValueError(
                            "deferral.regress_transform: rankgauss needs the "
                            "fold's training targets; build the Objective "
                            "with r_ref=")
                    ref = torch.as_tensor(self.r_ref)
                    r_tgt = (torch.stack([triage.rank_gauss(r_tgt[:, j], ref[:, j].contiguous())
                                          for j in range(r_tgt.shape[1])], 1)
                             if r_tgt.dim() == 2 else triage.rank_gauss(r_tgt, ref))
                disp_loss = F.mse_loss(disp_out, r_tgt)
                terms["dispersion"] = disp_loss.detach()
                loss = loss + lam["defer"] * disp_loss
            if out.get("defer") is not None:
                target = triage.build_deferral_targets(d, y, self.tau_d)
                defer_loss = triage.deferral_ce_loss(out["defer"], target)
                terms["defer"] = defer_loss.detach()
                loss = loss + lam["defer"] * defer_loss

        self.last_terms = terms
        return loss, self.note(lam)

    def train_epoch(self, ctx) -> tuple[float, str]:
        """One epoch over `ctx.train`, as the runner's `train_step`.

        This is the whole training step for a study whose objective is the
        standard contribution set. A study that adds a term of its own writes
        its own step and calls `self(...)` for the shared part."""
        ctx.model.train()
        rng = np.random.default_rng(self.cfg.seed + 1000 * ctx.epoch)
        total = 0.0
        sums: dict = {}               # per-term sums, kept on device until the end
        for batch in ctx.train:
            x = batch["image"].to(ctx.device)
            ctx.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=ctx.device.type,
                                    enabled=ctx.use_amp):
                out = ctx.model(x, return_all=True)
                loss, note = self(ctx.model, batch, out, ctx.epoch, ctx.device,
                                  rng=rng)
            if ctx.use_amp:
                ctx.scaler.scale(loss).backward()
                ctx.scaler.step(ctx.opt)
                ctx.scaler.update()
            else:
                loss.backward()
                ctx.opt.step()
            total += loss.item() * x.size(0)
            for k, v in getattr(self, "last_terms", {}).items():
                sums[k] = sums.get(k, 0.0) + v.float() * x.size(0)
        n = len(ctx.train.dataset)
        # The epoch means of every term in play, unweighted, after the schedule
        # values: which term dominates the total is not visible from the total.
        self.epoch_terms = {k: float(v) / n for k, v in sums.items()}
        parts = " ".join(f"{k} {v:.4f}" for k, v in self.epoch_terms.items())
        return total / n, (f"{note} | {parts}" if note else parts)
