"""The training stack: the typed config, the loss it assembles, and the loop.

What is worth testing is the correspondence: a term is in the loss when
its block is enabled and only then, an arm differs from the base by exactly the
components named, and the loop stops and selects where the config says.

The model in the loop tests is scripted rather than trained: `fit` is
orchestration, so what matters is which epochs it keeps given a val-AUC
sequence, not that a real network improves.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from field_disagreement import (
    TriageExperiment,
    ConformalConfig,
    DataLoaderConfig,
    DeferralConfig,
    ExperimentConfig,
    InvalidConfig,
    ModelConfig,
    Objective,
    OpinionConfig,
    OptimizerConfig,
    TrainingConfig,
    build_model,
)
from field_disagreement.engine import Fold, Selection
from field_disagreement.engine.runner import ExperimentRunner, _step_result
from backend.config import load
from backend.notebooks.shared import FoldRunner
from backend.notebooks.shared.checkpoints import CheckpointStore


def with_panel(cfg, on: bool):
    """`cfg` with the panel terms (opinion and deferral) switched on or off."""
    return replace(cfg, opinion=replace(cfg.opinion, enabled=on),
                   deferral=replace(cfg.deferral, enabled=on))


DEVICE = torch.device("cpu")
K = 5


# =========================================================================== #
# the typed config — invariants that used to fail silently
# =========================================================================== #
def test_a_miscoverage_rate_outside_zero_and_one_is_rejected():
    """alpha is a MISCOVERAGE rate; target coverage is 1 - alpha. Passing 0.9
    meaning "90% coverage" gives 10% coverage and no error."""
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(InvalidConfig, match="alpha"):
            ConformalConfig(alpha=bad)


def test_the_ablation_switches_reject_a_typo():
    """A misspelt mode silently selecting neither arm is the failure this
    prevents; `n_ratings` below 2 trains a head against a wider target."""
    with pytest.raises(InvalidConfig, match="opinion.loss"):
        OpinionConfig(loss="emdd")
    with pytest.raises(InvalidConfig, match="n_ratings"):
        ModelConfig(n_ratings=1)


def test_a_deferral_head_needs_the_panel_not_the_opinion_head():
    """Its target is built from how divided the panel was - the batch's
    disagreement - never from the opinion head's output. Deferral without the
    opinion head is therefore a real arm (learn WHEN to defer, not WHAT the panel
    said); what it needs is a cohort WITH a panel, which is a capability the
    domain checks when it binds the cohort."""
    arm = ExperimentConfig(opinion=OpinionConfig(enabled=False),
                           deferral=DeferralConfig(enabled=True))
    assert "opinion_distribution" in arm.needed_capabilities()


def test_an_arm_differs_from_the_base_by_exactly_its_components():
    """The point of running every arm from one config: two arms cannot differ
    by anything except the contributions named, because everything else is the
    same frozen object. `panel` carries the deferral head it supervises."""
    base = ExperimentConfig()
    arm = with_panel(base, False)

    assert (arm.opinion.enabled, arm.deferral.enabled) == (False, False)
    assert arm.model == base.model and arm.seed == base.seed
    assert base.opinion.enabled is True            # the base is untouched


def test_the_capabilities_a_run_needs_follow_from_what_it_switches_on():
    """What the domain checks the cohort against, so an arm that never trains an
    opinion head is not blocked from a cohort with no panel."""
    full = ExperimentConfig()
    assert "opinion_distribution" in full.needed_capabilities()
    assert with_panel(full, False).needed_capabilities() == \
        ("classification",)


def test_a_config_survives_the_round_trip_it_is_stored_through():
    """It is written into every fold result so weights and settings travel
    together; a lossy round-trip makes that record wrong rather than absent. A
    key from another version is named rather than ignored."""
    cfg = ExperimentConfig(model=ModelConfig(backbone="cnn", n_ratings=2),
                           opinion=OpinionConfig(loss="kl", weight=0.25))
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg

    with pytest.raises(InvalidConfig, match="unknown config keys"):
        ExperimentConfig.from_dict({**cfg.to_dict(), "not_a_block": 0.5})


@pytest.mark.parametrize("name", ["notebooks/base", "notebooks/study"])
def test_the_projects_own_yaml_produces_a_valid_config(name):
    """The bridge has to work on the real files, not just on defaults — they are
    the locked protocol and this is only a validated view of them."""
    flat = load(name)
    flat.n_ratings = 2                      # normally set by datasets.bind
    experiment = ExperimentConfig.from_namespace(flat)

    assert experiment.epochs == flat.epochs
    assert experiment.conformal.alpha == flat.alpha
    assert experiment.deferral.weight == flat.lam_def


def test_the_flat_configs_way_of_switching_the_panel_terms_off_is_understood():
    """`opinion_loss: none` is how the yaml says "no opinion head"; read as a
    loss name it would be an unknown-objective error instead."""
    experiment = ExperimentConfig.from_namespace(
        SimpleNamespace(opinion_loss="none", n_ratings=5, alpha=0.1))
    assert (experiment.opinion.enabled, experiment.deferral.enabled) == \
        (False, False)


# =========================================================================== #
# the loss the config assembles
# =========================================================================== #
@pytest.fixture
def batch():
    """One batch of the `field_disagreement.spectrum` contract."""
    torch.manual_seed(0)
    n, size = 4, 64
    mask = (torch.rand(n, size, size) > 0.5).float()
    return {"image": torch.rand(n, 3, size, size),
            "label": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "agreement_map": torch.rand(n, size, size),
            # THE NORMALISED surface as well as the raw count: a proportion,
            # because the binomial precision of a count depends on how many
            # readers there were. The domain adapter has supplied both since the format note in
            # backend/domains/adapter.py; the fixture had drifted from it.
            "agreement_frac": torch.rand(n, size, size),
            "binary_mask": mask,
            "opinion_dist": torch.full((n, K), 1.0 / K),
            "disagreement": torch.tensor([0.0, 1.2, 0.3, 1.8])}


@pytest.fixture
def net():
    """Eval mode on purpose: these tests compare losses ACROSS configs, and an
    active dropout mask would make two forwards of one input differ for a reason
    unrelated to which terms are enabled."""
    return build_model("cnn", pretrained=False, opinion_head=True,
                       defer_head=True, n_ratings=K).eval()


def loss_for(cfg, net, batch, epoch=99):
    obj = Objective(cfg, cls_loss=nn.BCEWithLogitsLoss(), tau_d=0.5)
    out = net(batch["image"], return_all=True)
    loss, _ = obj(net, batch, out, epoch, DEVICE, rng=np.random.default_rng(0))
    # Detached because these tests only COMPARE loss values across configs and
    # never backward through them; reading a grad-carrying tensor as a scalar
    # keeps the graph alive for a number nothing differentiates.
    return float(loss.detach())


BARE = with_panel(ExperimentConfig(), False)


def test_a_classifier_only_config_is_just_the_classification_loss(net, batch):
    out = net(batch["image"], return_all=True)
    expected = nn.BCEWithLogitsLoss()(out["logit"], batch["label"]).detach()
    assert loss_for(BARE, net, batch) == pytest.approx(float(expected), rel=1e-5)


def test_switching_the_panel_terms_on_changes_the_loss(net, batch):
    """And switching them off returns to the control EXACTLY — which is what makes
    an arm difference attributable to those terms and nothing else."""
    arm = with_panel(BARE, True)
    control = loss_for(BARE, net, batch)

    assert loss_for(arm, net, batch) != pytest.approx(control, rel=1e-6)
    assert loss_for(with_panel(arm, False), net, batch) == pytest.approx(control, rel=1e-6)


def test_auc_band_keeps_the_last_epoch_at_parity_and_no_worse_one():
    """Both halves of the rule, as a unit. Inside the band the LATER epoch wins
    however much lower its AUC (that is the exposure fix); outside it, nothing
    wins (that is the overfitting guarantee)."""
    cfg = SimpleNamespace(epochs=10, patience=1, ckpt_every=100, sel_tol=0.01,
                          sel_window=5, amp=False, loss="bce", focal_alpha=0.25,
                          focal_gamma=2.0)
    runner = ExperimentRunner(cfg, CheckpointStore(Path("."), "m", DEVICE), DEVICE)
    mk = lambda n, sm: Selection(rule="auc_band", epoch=n, val_auc=sm,
                                 val_auc_smooth=sm, min_epochs=0)

    kept = mk(30, 0.9000)                              # anchor 0.9000, band 0.8900
    assert runner._keep(kept, mk(50, 0.8910), 0.9000, False)   # in band, later
    assert runner._keep(kept, mk(50, 0.8900), 0.9000, False)   # exactly on the edge
    assert not runner._keep(kept, mk(50, 0.8899), 0.9000, False)  # drifted out
    # An epoch with no smoothed value cannot be judged, so it cannot be kept.
    assert not runner._keep(kept, Selection(rule="auc_band", epoch=51,
                                            val_auc=0.99, min_epochs=0),
                            0.9000, False)


def test_a_single_spike_cannot_collapse_the_band(tmp_path):
    """`band_latest` died of this: one 0.9213 epoch anchored the band so high that
    nothing later cleared it, and one arm was scored on 12 full-weight
    epochs against another's 50. The anchor is the SMOOTHED series now, so the same
    spike moves it by a fifth as much and the plateau stays in band."""
    cfg = SimpleNamespace(epochs=10, patience=1, ckpt_every=100, sel_tol=0.01,
                          sel_window=5, amp=False, loss="bce", focal_alpha=0.25,
                          focal_gamma=2.0)
    runner = ExperimentRunner(cfg, CheckpointStore(Path("."), "m", DEVICE), DEVICE)
    plateau = [0.8850] * 4 + [0.9213]                 # the spike, then the window
    smoothed = sum(plateau) / len(plateau)            # 0.8923, not 0.9213
    mk = lambda n, sm: Selection(rule="auc_band", epoch=n, val_auc=sm,
                                 val_auc_smooth=sm, min_epochs=0)

    # Raw-anchored, a 0.8850 plateau epoch is 0.036 below the spike -> shut out.
    assert not runner._keep(mk(22, 0.9213), mk(52, 0.8850), 0.9213, False)
    # Smoothed, the anchor is 0.8923 and the same epoch is inside the band.
    assert runner._keep(mk(22, smoothed), mk(52, 0.8850), smoothed, False)


def test_the_selection_rule_is_config_driven_and_checked():
    """The checkpoint-selection rule comes from the config and is checked there."""
    assert TrainingConfig().selection == "auc_band"
    assert TrainingConfig(selection="auc").selection == "auc"
    with pytest.raises(InvalidConfig, match="sel_window"):
        TrainingConfig(sel_window=0)
    with pytest.raises(InvalidConfig, match="training.sel_tol"):
        TrainingConfig(sel_tol=0.0)
    # An unknown rule must RAISE, never map onto something else: a config that
    # silently changes selection rule is how two protocols end up in one results
    # folder.
    for gone in ("plateau", "converged", "val_loss"):
        with pytest.raises(InvalidConfig, match="training.selection"):
            TrainingConfig(selection=gone)


def test_every_auxiliary_term_ramps_in_from_zero():
    """At epoch 0 the panel term is still near zero, so an arm starts close to
    its own control rather than fighting the classifier when the classification
    signal is weakest. A disabled term is 0 at every epoch."""
    full = ExperimentConfig()
    start = Objective(full, cls_loss=None).lambdas(0)
    late = Objective(full, cls_loss=None).lambdas(full.epochs - 1)

    assert start["defer"] < 0.05 * full.deferral.weight
    assert late["defer"] == pytest.approx(full.deferral.weight)

    off = Objective(BARE, cls_loss=None)
    for epoch in (0, 5, 99):
        assert set(off.lambdas(epoch).values()) == {0.0}


def test_the_log_note_names_only_the_terms_in_play():
    """A baseline arm's line should not carry a zero for a term it does not
    have."""
    off = Objective(BARE, cls_loss=None)
    assert off.note(off.lambdas(9)) == ""

    full = Objective(with_panel(ExperimentConfig(), True),
                     cls_loss=None, tau_d=0.45)
    assert "lam_d" in full.note(full.lambdas(99))


# =========================================================================== #
# the loop
# =========================================================================== #
LABELS = torch.tensor([0.0, 0.0, 1.0, 1.0])
PERFECT = [0.1, 0.2, 0.8, 0.9]      # AUC 1.00 against LABELS
MIDDLING = [0.1, 0.5, 0.4, 0.9]     # AUC 0.75
CHANCE = [0.5, 0.5, 0.5, 0.5]       # AUC 0.50


def stepping(loss=0.5):
    """A stand-in train step that takes a REAL optimiser step.

    The loop steps the schedule once the step returns, so a step that never
    touched the optimiser makes PyTorch report a schedule stepped before its
    optimiser — a warning about this fixture rather than about `fit`, and one
    that would mask the same warning if the loop ever did earn it.
    """
    def train_step(ctx):
        ctx.opt.step()
        return loss
    return train_step


class ScriptedNet(nn.Module):
    """Returns a scripted probability vector, one entry per forward pass.

    The parameter is real so the optimiser, scheduler and checkpoint round-trip
    have something to act on; it does not affect the output."""

    def __init__(self, script):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.script = [torch.tensor(p) for p in script]
        self.calls = 0

    def forward(self, x):
        probs = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return torch.logit(probs.clamp(1e-6, 1 - 1e-6)) + 0.0 * self.w


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)


class SpyStore(CheckpointStore):
    """A real store that also records which epoch each `save_best` came from,
    and the selection state the last `save_ckpt` carried."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_at = []
        self.saved_extra = None

    def save_best(self, tag, model, kind="best.pt"):
        self.best_at.append(getattr(model, "calls", None))
        super().save_best(tag, model, kind=kind)

    def save_ckpt(self, *args, extra=None, **kwargs):
        self.saved_extra = extra
        super().save_ckpt(*args, extra=extra, **kwargs)


@pytest.fixture
def loop_cfg():
    return SimpleNamespace(epochs=10, patience=1, ckpt_every=100, sel_window=5,
                           warmup=1, lr_vit=1e-4, lr_head=3e-4, weight_decay=0.0,
                           no_amp=True, loss="bce")


def scripted(loop_cfg, store, script):
    """A runner and a hand-built `Fold` around a scripted net."""
    from backend.notebooks.shared.training import amp_components, build_optimizer, build_scheduler
    net = ScriptedNet(script)
    batch = {"image": torch.zeros(4, 1), "label": LABELS,
             "file": [f"f{i}" for i in range(4)]}
    ctx = Fold(cfg=loop_cfg, fold=0, tag="unit", device=DEVICE,
               loaders={"train": [], "val": [batch], "test": []},
               loss_fn=lambda logit, y: logit.sum())
    ctx.model = net
    ctx.opt = build_optimizer(net, loop_cfg)
    ctx.sched = build_scheduler(ctx.opt, loop_cfg)
    ctx.use_amp, ctx.scaler = amp_components(DEVICE, loop_cfg)
    return ExperimentRunner(loop_cfg, store, DEVICE), ctx, net


def test_early_stopping_waits_for_a_warm_up_to_finish(loop_cfg, tmp_path):
    """A panel term that ramps in over the first epochs, with a shorter patience,
    could otherwise stop an arm before it ever trained at full weight the head the
    study measures. The default (no
    floor) has to stay the historical behaviour.

    `min_epochs` is a floor on WHEN stopping may fire and on what may spend the
    patience budget. Here the only good epoch is the first, from inside the ramp:
    with the floor its peak is not something later epochs can stall against, so
    the run continues; without one it ends the run immediately."""
    # A store each: both runs use the tag `unit`, and the first now trains to the
    # end of its schedule, so a shared directory would leave the second resuming
    # from a finished run instead of starting one.
    store = SpyStore(tmp_path / "floored", "m", DEVICE, use_drive=False)
    runner, ctx, net = scripted(loop_cfg, store, [PERFECT] + [CHANCE] * 20)
    runner.fit(ctx, stepping(), selection="auc", min_epochs=5)
    assert net.calls == 10          # ran the schedule out, not stopped at 1

    store = SpyStore(tmp_path / "unfloored", "m", DEVICE, use_drive=False)
    runner, ctx, net = scripted(loop_cfg, store, [PERFECT] + [CHANCE] * 20)
    runner.fit(ctx, stepping(), selection="auc", min_epochs=0)
    assert net.calls == 2


def test_the_ramp_does_not_spend_the_patience_budget(loop_cfg, tmp_path):
    """Patience is counted against the best AUC reached at FULL objective weight.

    The failure this prevents: with the anchor on the global best, a val-AUC peak
    from the lam=0 warm-up sets the bar, the panel term legitimately costs a
    little AUC, and the arm stops a few epochs after the ramp ends having barely
    trained under the objective it is named for.

    Stopping must still FIRE - the floor is a re-anchoring, not a suppression."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    runner, ctx, net = scripted(loop_cfg, store, [PERFECT] * 6 + [MIDDLING] * 20)
    runner.fit(ctx, stepping(), selection="auc", min_epochs=5)

    # ep 6 (index 5) is the first at full weight and sets the anchor at 1.0;
    # ep 7 misses it, patience (1) runs out, and the run stops there.
    assert net.calls == 7


def test_the_best_auc_epoch_is_the_one_saved(loop_cfg, tmp_path):
    """`selection="auc"`: weights are written only while val AUC improves."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    loop_cfg.patience = 99                             # let the whole script run
    runner, ctx, _ = scripted(loop_cfg, store, [MIDDLING, PERFECT, CHANCE, CHANCE])

    best_auc = runner.fit(ctx, stepping(), selection="auc")

    assert store.best_at == [1, 2]                     # improved, then peaked
    assert best_auc == pytest.approx(1.0)


def test_every_epoch_is_logged_and_the_monitor_never_selects(loop_cfg, tmp_path):
    """`history` gets one row per trained epoch, with the optional val monitor's
    numbers; a monitor that prefers a different epoch does not move selection,
    which stays on val AUC."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    loop_cfg.patience = 99
    runner, ctx, _ = scripted(loop_cfg, store, [MIDDLING, PERFECT, CHANCE, CHANCE])
    seen = iter([0.1, 0.2, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    ctx.extras["val_monitor"] = lambda c: {"val_rho_fields": next(seen)}
    runner.fit(ctx, stepping(), selection="auc")
    assert [h["epoch"] for h in ctx.history] == list(range(1, 11))
    assert ctx.history[2]["val_rho_fields"] == 0.9
    assert ctx.selection.epoch == 2                    # the AUC peak, not the rho peak


def test_resuming_an_old_checkpoint_does_not_overwrite_better_weights(loop_cfg,
                                                                      tmp_path):
    """A checkpoint written before the loop carried a selection score has
    `extra=None`. Read as "no best yet", the first epoch after a resume always
    counts as an improvement and replaces good weights with worse ones. Under
    plain-AUC selection the resumed `best_auc` IS the selection score."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    runner, ctx, _ = scripted(loop_cfg, store, [CHANCE] * 5)
    store.save_ckpt("unit", ctx.model, ctx.opt, ctx.sched, ctx.scaler,
                    next_epoch=3, best_auc=1.0, since=0, extra=None)

    runner.fit(ctx, stepping(), selection="auc", min_epochs=0)

    assert store.best_at == []          # chance-level epochs never displaced it


# --------------------------------------------------------------------------- #
# the ramp gate, and what a resume restores
# --------------------------------------------------------------------------- #
def bare_runner(loop_cfg, tmp_path):
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    return ExperimentRunner(loop_cfg, store, DEVICE)


def epoch(n, auc, ramp=0, rule="auc"):
    return Selection(rule=rule, epoch=n, val_auc=auc, min_epochs=ramp)


def test_only_full_weight_epochs_are_eligible_for_selection(loop_cfg, tmp_path):
    """A ramp epoch can look excellent and still
    not be a measurement of the arm, so it is held only until a full-weight epoch
    exists to take over."""
    runner = bare_runner(loop_cfg, tmp_path)
    kept = epoch(7, 0.94, ramp=10)

    # Still inside the ramp: the hold is on AUC alone.
    assert runner._keep(kept, epoch(8, 0.96, ramp=10), 0.96, True)
    assert not runner._keep(kept, epoch(8, 0.92, ramp=10), 0.96, True)

    # The first full-weight epoch takes over unconditionally - worse AUC does not
    # matter. The ramp epoch was a placeholder, never a candidate.
    assert runner._keep(kept, epoch(11, 0.80, ramp=10), 0.80, True)


def test_a_bare_float_checkpoint_resumes_its_auc(loop_cfg, tmp_path):
    """A mid-run checkpoint holding a bare float: under "auc" it IS the val AUC
    and is kept, which is what stops a resume from overwriting good weights with
    worse ones."""
    runner = bare_runner(loop_cfg, tmp_path)
    keep, _ = runner._resume_selection(0.87, "auc", 0.87, start=4, min_epochs=0)
    assert keep.val_auc == pytest.approx(0.87)


def test_a_ramp_epoch_is_handed_over_however_good_it_looked(loop_cfg, tmp_path):
    """End to end: the best epoch of the whole run is inside the ramp, and the
    run still ends holding a full-weight epoch, so no arm's row measures a model
    its objective had barely touched."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    loop_cfg.patience = 99
    runner, ctx, _ = scripted(loop_cfg, store, [MIDDLING, PERFECT] + [CHANCE] * 8)

    runner.fit(ctx, stepping(), selection="auc", min_epochs=5)

    assert ctx.selection.epoch >= 6                      # a full-weight epoch
    assert ctx.selection.post_warmup is True
    assert ctx.selection.val_auc == pytest.approx(0.5)   # ...worse, and kept


def test_a_run_that_never_reaches_full_weight_says_so(loop_cfg, tmp_path):
    """The one case that can still end on a ramp epoch: the run stopped inside
    the ramp. The weights are honest to restore and dishonest to label, so the
    record marks them and `fit` says so on the way out."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    loop_cfg.epochs, loop_cfg.patience = 3, 99
    runner, ctx, _ = scripted(loop_cfg, store, [MIDDLING, PERFECT, CHANCE])

    runner.fit(ctx, stepping(), selection="auc", min_epochs=5)

    assert ctx.selection.epoch == 2                      # the provisional hold
    assert ctx.selection.post_warmup is False
    assert ctx.selection.to_dict()["post_warmup"] is False


def test_the_auc_band_rule_round_trips_through_a_checkpoint(loop_cfg, tmp_path):
    """End to end on the "auc_band" path, across a resume.

    The selection state is a record, not a packed float, so it has to survive
    `save_ckpt` -> `resume` intact - otherwise every disconnect on a free GPU
    silently restarts model selection, which is the runtime this store exists for.
    """
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    loop_cfg.patience, loop_cfg.ckpt_every, loop_cfg.epochs = 99, 1, 4
    # Four items give AUCs in steps of 0.25, so the band has to be wide enough to
    # straddle one step. 0.3 puts MIDDLING (0.75) inside PERFECT's (1.00) band and
    # CHANCE (0.50) outside it.
    loop_cfg.sel_window, loop_cfg.sel_tol = 1, 0.3      # no smoothing: exact
    runner, ctx, _ = scripted(loop_cfg, store,
                              [CHANCE, PERFECT, MIDDLING, CHANCE])

    runner.fit(ctx, stepping(), selection="auc_band", min_epochs=0)

    # ep2 is the PEAK and ep3 is worse - but ep3 is still at parity, so the LAST
    # in-band epoch wins. ep4 has drifted out of the band and is refused, which is
    # the overfitting half of the rule.
    assert ctx.selection.epoch == 3
    assert ctx.selection.val_auc == pytest.approx(0.75)
    assert ctx.selection.post_warmup is True

    # The checkpoint written at epoch 4 carries the epoch SELECTION kept, and that
    # is what a resume starts from.
    resumed, _ = runner._resume_selection(
        store.saved_extra, "auc_band", 1.0, start=4, min_epochs=0)
    assert resumed.epoch == 3
    assert resumed.val_auc_smooth == pytest.approx(0.75)


def test_the_step_contract_and_the_selection_rules_are_enforced(loop_cfg,
                                                                tmp_path):
    """A step returns its loss, optionally with a note for the log line. An
    unknown selection rule raises rather than falling back to plain AUC, which
    would silently undo the configured rule."""
    assert _step_result(0.25) == (0.25, "")
    assert _step_result((0.25, "lam 0.50")) == (0.25, "lam 0.50  ")

    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    runner, ctx, _ = scripted(loop_cfg, store, [CHANCE])
    with pytest.raises(ValueError, match="unknown selection"):
        runner.fit(ctx, stepping(), selection="best")


# --------------------------------------------------------------------------- #
# the whole unit of work, the way a notebook calls it
# --------------------------------------------------------------------------- #
@pytest.fixture
def fold_runner(loop_cfg, tmp_path, monkeypatch):
    """A `FoldRunner` whose domain half — loaders and class balance — is stubbed,
    so the cached-unit-of-work path can be exercised without a dataset."""
    from backend.notebooks.shared import runner as R

    batch = {"image": torch.randn(8, 4),
             "label": torch.tensor([0.0] * 4 + [1.0] * 4),
             "file": [f"f{i}" for i in range(8)]}
    monkeypatch.setattr(R, "loaders_for", lambda *a, **k: {
        "train": [batch], "val": [batch], "test": [batch],
        "splits": {"train": ["f0"]}})
    monkeypatch.setattr(R, "pos_weight_for", lambda *a, **k: 1.0)
    loop_cfg.epochs, loop_cfg.patience = 2, 99
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    return FoldRunner(loop_cfg, store, DEVICE, df=None, patient_fold=None,
                      schema=7)


def test_run_fold_builds_trains_evaluates_and_then_caches(fold_runner):
    """The path the notebook takes: cache miss -> build -> fit -> best weights
    -> evaluate -> cached result, then a cache HIT that skips all of it.

    `build` also stashes something in `extras`, which is how a study carries a
    value computed at build time through to its step."""
    seen = []

    def build(ctx):
        ctx.extras["tau_d"] = 0.42
        return TinyNet()

    def train_step(ctx):
        for b in ctx.train:
            ctx.opt.zero_grad(set_to_none=True)
            loss = ctx.loss_fn(ctx.model(b["image"]), b["label"])
            loss.backward()
            ctx.opt.step()
        return float(loss.detach()), f"tau_d {ctx.extras['tau_d']:.2f}"

    def evaluate(ctx):
        seen.append(ctx.extras["tau_d"])     # extras survive from build to here
        return {"metric": 1.0}

    kwargs = {"build": build, "train_step": train_step, "evaluate": evaluate}
    result = fold_runner.run_fold(0, "unit", **kwargs)

    assert result["fold"] == 0 and result["metric"] == 1.0
    assert result["schema"] == 7 and "best_val_auc" in result
    assert seen == [0.42]

    assert fold_runner.run_fold(0, "unit", **kwargs) == result
    assert seen == [0.42]                    # nothing was rebuilt


def test_a_cached_result_the_study_calls_stale_is_re_run(fold_runner):
    """`is_fresh` is a study's own freshness question, which the schema counter
    does not answer: a cached result missing a metric the study now reports would
    print as "-" and read like a failed arm rather than stale numbers."""
    runs = []

    def evaluate(ctx):
        runs.append(ctx.fold)
        return {"new_metric": 1.0}

    kwargs = {"build": lambda ctx: TinyNet(), "train_step": stepping(),
              "evaluate": evaluate}

    fold_runner.run_fold(0, "unit", **kwargs)
    assert runs == [0]

    fold_runner.run_fold(0, "unit", **kwargs, is_fresh=lambda d: "new_metric" in d)
    assert runs == [0]                                   # the metric is there

    fold_runner.run_fold(0, "unit", **kwargs, is_fresh=lambda d: "later" in d)
    assert runs == [0, 0]                                # it is not — re-run


# =========================================================================== #
# the typed TRAINING protocol — the half that must not differ between arms
# =========================================================================== #
def test_the_training_protocol_rejects_a_run_that_cannot_happen():
    """Getting one of these wrong must cost an exception, not a training run."""
    with pytest.raises(InvalidConfig, match="training.loss"):
        TrainingConfig(loss="bcee")
    with pytest.raises(InvalidConfig, match="training.sel_window"):
        TrainingConfig(sel_window=0)
    with pytest.raises(InvalidConfig, match="training.epochs"):
        TrainingConfig(epochs=0)


def test_the_optimiser_and_loader_blocks_reject_their_own_dead_ends():
    with pytest.raises(InvalidConfig, match="optimizer.lr_head"):
        OptimizerConfig(lr_head=-1.0)
    with pytest.raises(InvalidConfig, match="data.batch_size"):
        DataLoaderConfig(batch_size=0)
    with pytest.raises(InvalidConfig, match="data.shift_frac"):
        DataLoaderConfig(shift_frac=1.0)


@pytest.mark.parametrize("block, field, expected", [
    ("training", "patience", 15),
    ("optimizer", "lr_trunk", 1.0e-4),
    ("data", "batch_size", 32),
])
def test_the_typed_blocks_resolve_from_the_flat_yaml(block, field, expected):
    """The project has two configuration layers in flight. One resolver is what
    lets the runner, the optimiser and the façade all be written against the
    typed config without breaking the notebook that passes the flat one."""
    flat = load("notebooks/base")
    flat.n_ratings = 2

    typed = ExperimentConfig.of(flat)
    assert getattr(getattr(typed, block), field) == expected
    assert ExperimentConfig.of(typed) is typed        # already typed, untouched


def test_the_flat_configs_negative_amp_switch_is_read_correctly():
    """The YAML states it as `no_amp`. Read straight through as `amp` it would
    turn mixed precision off exactly when the config asked for it on."""
    assert ExperimentConfig.of(SimpleNamespace(no_amp=False)).training.amp is True
    assert ExperimentConfig.of(SimpleNamespace(no_amp=True)).training.amp is False


def test_the_run_length_has_one_home_and_a_stored_run_still_reads():
    """`epochs` moved into the training block. The deferral ramp is a fraction
    of TOTAL training, so the objective reads it — through the property, so
    there is no second copy to fall out of step. A result.json written before
    the move is still the record of a real run and is read, not rejected."""
    cfg = ExperimentConfig(training=TrainingConfig(epochs=30))
    assert cfg.epochs == 30 and ExperimentConfig.from_dict(cfg.to_dict()) == cfg

    legacy = ExperimentConfig.from_dict({"seed": 7, "epochs": 30})
    assert legacy.epochs == 30 and legacy.training.epochs == 30


def test_the_loop_reads_the_flat_namespace_the_notebooks_still_pass(loop_cfg,
                                                                    tmp_path):
    """The bridge has to hold both ways round, or the notebook breaks."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    runner, _, _ = scripted(loop_cfg, store, [CHANCE])
    assert runner.training.epochs == loop_cfg.epochs
    assert runner.training.patience == loop_cfg.patience
    assert runner.training.sel_window == loop_cfg.sel_window


# =========================================================================== #
# the whole study, composed from the config
# =========================================================================== #
class Loader(list):
    """A loader stand-in: the batches, plus the `.dataset` an epoch mean needs."""

    @property
    def dataset(self):
        return [None] * sum(int(b["label"].numel()) for b in self)


class FakeFolds:
    """A `FoldSource` with no cohort behind it.

    The point of the protocol is that the façade never learns what the images
    are, so the test of that is a source made of noise: if anything in
    `TriageExperiment` needed a real dataset, this could not drive it.
    """

    def __init__(self, n_folds=2, n=4, size=64):
        self.n_folds, self.n, self.size = n_folds, n, size

    def __len__(self):
        return self.n_folds

    def loaders(self, fold):
        torch.manual_seed(100 + fold)
        n, size = self.n, self.size
        batch = {"image": torch.rand(n, 3, size, size),
                 "label": torch.tensor([0.0, 1.0] * (n // 2)),
                 "file": [f"f{fold}_{i}" for i in range(n)],
                 "agreement_map": torch.rand(n, size, size),
                 "binary_mask": (torch.rand(n, size, size) > 0.5).float(),
                 "opinion_dist": torch.full((n, K), 1.0 / K),
                 "disagreement": torch.rand(n)}
        return {k: Loader([batch]) for k in ("train", "val", "test")}

    def pos_weight(self, fold):
        return 1.0


@pytest.fixture
def study():
    """A two-fold, two-epoch run with the panel terms switched on."""
    from field_disagreement import UncertaintyConfig
    return with_panel(ExperimentConfig(
        model=ModelConfig(backbone="cnn", pretrained=False, n_ratings=K),
        training=TrainingConfig(epochs=2, warmup=1, patience=99),
        uncertainty=UncertaintyConfig(mc_dropout=False, samples=0),
    ), True)


def experiment_for(study, store, **kw):
    return TriageExperiment(study, FakeFolds(), store=store, device=DEVICE,
                              tag="unit", **kw)


def test_the_experiment_trains_every_fold_and_records_what_produced_it(study,
                                                                       tmp_path):
    """dataset -> folds -> model -> objective -> training -> evaluation, from
    one config and one call."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    results = experiment_for(study, store).fit()

    assert [r["fold"] for r in results] == [0, 1]
    assert all("best_val_auc" in r and "test_auc" in r for r in results)
    # The settings travel with the numbers, or a result read later is scored
    # against whatever the YAML says by then.
    assert results[0]["config"] == study.to_dict()


def test_the_network_carries_only_the_heads_its_arm_supervises(study, tmp_path):
    """The heads follow from the contributions rather than from a separate
    switch. An arm with no opinion term that still built an opinion head would
    differ from its control by a network as well as by an objective."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)

    full = experiment_for(study, store).build_fold(0)
    out = full.model(next(iter(full.train))["image"], return_all=True)
    assert "opinion" in out and "defer" in out

    bare = experiment_for(with_panel(study, False), store).build_fold(0)
    out = bare.model(next(iter(bare.train))["image"], return_all=True)
    assert "opinion" not in out and "defer" not in out


def test_evaluating_again_rescores_the_selected_weights_and_cannot_move_them(
        study, tmp_path):
    """The operation you want after adding a metric: re-scoring the epoch
    selection already chose must not cost another run, and must not be able to
    quietly train further and report a different model's numbers."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    experiment = experiment_for(study, store)

    trained = experiment.fit()
    rescored = experiment.evaluate()

    assert [r["fold"] for r in rescored] == [0, 1]
    for was, now in zip(trained, rescored):
        assert now["test_auc"] == pytest.approx(was["test_auc"])


def test_a_study_that_adds_a_term_keeps_the_rest_of_the_composition(study,
                                                                    tmp_path):
    """Replacing the step is how a contribution stays readable in the notebook
    that argues for it. Taking the default is how the studies that add no term
    stop maintaining a copy of the same one."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    seen = []

    def train_step(ctx):
        seen.append(ctx.epoch)
        ctx.opt.step()
        return 0.5

    experiment_for(study, store, train_step=train_step).fit()
    assert seen == [0, 1, 0, 1]              # two epochs, twice, and nothing else


def test_calibration_does_not_disturb_the_weights_selection_chose(study, tmp_path):
    """It is a read of held-out data, not a training pass. If it could move the
    weights, every number reported before it would describe another model."""
    store = SpyStore(tmp_path, "m", DEVICE, use_drive=False)
    experiment = experiment_for(study, store)
    experiment.fit()

    before = experiment.evaluate()
    experiment.calibrate()
    after = experiment.evaluate()

    for was, now in zip(before, after):
        assert now["test_auc"] == pytest.approx(was["test_auc"])


def test_this_projects_domain_half_satisfies_the_protocol():
    """`CohortFolds` is what makes the façade usable on a real cohort; the
    protocol is the whole contract between them."""
    from field_disagreement import FoldSource
    from backend.notebooks.shared import CohortFolds

    folds = CohortFolds(SimpleNamespace(n_splits=5), df=None, patient_fold=None)
    assert isinstance(folds, FoldSource) and len(folds) == 5
    assert hasattr(folds, "pos_weight")


# =========================================================================== #
# the optimiser and schedule, now shipped with the loop that reads them
# =========================================================================== #
def test_the_differential_learning_rate_survives_the_rename(loop_cfg):
    """The flat config names the pretrained trunk's rate `lr_vit` after the
    backbone it was written for; the typed block names it `lr_trunk`. Read from
    the wrong key this would not raise — it would train a pretrained trunk at
    the heads' rate for a whole run and report the result as the arm.
    """
    from field_disagreement.engine import build_optimizer

    class Split(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk, self.head = nn.Linear(4, 4), nn.Linear(4, 1)

        def param_groups(self, lr_trunk, lr_head):
            return [{"params": list(self.trunk.parameters()), "lr": lr_trunk},
                    {"params": list(self.head.parameters()), "lr": lr_head}]

    opt = build_optimizer(Split(), loop_cfg)
    assert [g["lr"] for g in opt.param_groups] == [loop_cfg.lr_vit,
                                                   loop_cfg.lr_head]

    # A from-scratch net has no such split, and training all of it at lr_head is
    # the right answer rather than a missing feature.
    plain = build_optimizer(TinyNet(), loop_cfg)
    assert [g["lr"] for g in plain.param_groups] == [loop_cfg.lr_head]


def test_the_schedule_warms_up_then_decays_to_nothing(loop_cfg):
    """It moved packages with the optimiser, and a schedule that decayed over
    the wrong span would be invisible in any single epoch's log line."""
    from field_disagreement.engine import build_optimizer, build_scheduler

    opt = build_optimizer(TinyNet(), loop_cfg)
    sched = build_scheduler(opt, loop_cfg)

    lrs = []
    for _ in range(loop_cfg.epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    assert lrs[0] == pytest.approx(loop_cfg.lr_head)     # warmup=1: full at ep 0
    assert lrs == sorted(lrs, reverse=True)              # and never rises again
    assert lrs[-1] < 0.05 * loop_cfg.lr_head             # decayed to ~nothing


# ---------------------------------------------------------------------------
# Per-(arm, fold) seeding: which folds come from cache must not change the model
# a fold trains.
# ---------------------------------------------------------------------------

def test_a_folds_seed_does_not_depend_on_what_ran_before_it():
    """THE REGRESSION. `set_seed` used to be called once per run, so the RNG
    stream was global and sequential and a fold's init depended on how many
    folds had already trained. A cached fold trains nothing, so resuming a run
    silently handed the remaining folds a different draw. Re-ordering the loop
    does not help - fold-major skips consumption exactly the same way."""
    from backend.notebooks.shared.runner import fold_seed
    assert fold_seed(42, 0) == fold_seed(42, 0)


def test_the_three_arms_share_a_folds_seed():
    """Common random numbers: every arm starts from IDENTICAL weights within a
    fold, so a contrast between arms measures the LOSS and not the two arms'
    different initialisations. Removing that noise is the point."""
    from backend.notebooks.shared.runner import fold_seed
    assert fold_seed(42, 3) == fold_seed(42, 3)      # seed depends on fold only


def test_folds_differ_so_fold_spread_still_reports_init_variance():
    from backend.notebooks.shared.runner import fold_seed
    seeds = [fold_seed(42, f) for f in range(5)]
    assert len(set(seeds)) == 5


def test_the_run_seed_still_controls_the_draw():
    """Changing cfg.seed must still change every fold, or the seed sweep that
    produced the seed-42/44 replication would be measuring nothing."""
    from backend.notebooks.shared.runner import fold_seed
    assert fold_seed(42, 0) != fold_seed(44, 0)


def test_seed_and_fold_cannot_collide():
    """`+ fold` would put seed 42/fold 4 and seed 44/fold 2 both on 46."""
    from backend.notebooks.shared.runner import fold_seed
    pairs = {(b, f): fold_seed(b, f) for b in (42, 43, 44) for f in range(5)}
    assert len(set(pairs.values())) == len(pairs)


# --------------------------------------------------------------------------- #
# The decentring condition - pinned, because a drift would not be visible in
# any run's output.
# --------------------------------------------------------------------------- #
def test_no_notebook_sets_the_condition_behind_binds_back():
    """One place decides it. A notebook that assigns `shift_frac` itself can
    silently disagree with the cohort file."""
    import json
    from pathlib import Path

    for nb_path in sorted(Path("backend/notebooks").glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        src = "".join("".join(c["source"]) for c in nb["cells"]
                      if c["cell_type"] == "code")
        assert "cfg.shift_frac = " not in src, (
            f"{nb_path.name} assigns cfg.shift_frac itself; decentring is "
            f"applied once, in backend/domains/datasets.py:bind")


# --------------------------------------------------------------------------- #
# The arm list and screens
# --------------------------------------------------------------------------- #


def test_the_screen_can_only_ever_be_a_subset_of_the_study():
    """`screen_arms` runs fewer arms for a cheap decision. It must not be able to
    introduce an arm the study did not declare, or a screen becomes a second,
    undeclared experiment."""
    from types import SimpleNamespace
    from backend.notebooks.shared import screen_arms
    from field_disagreement.triage.ladder import ABLATIONS, ALL_ARMS

    declared = list(ALL_ARMS)
    assert screen_arms(SimpleNamespace(), declared) == declared
    assert screen_arms(SimpleNamespace(screen_arms=["A_cls", "H_fld"]),
                       declared) == ["A_cls", "H_fld"]
    with pytest.raises(ValueError, match="are not declared arms of this study"):
        screen_arms(SimpleNamespace(screen_arms=["H_new"]), declared)
    # an ablation enters only when named, and only if it is declared as one
    assert screen_arms(SimpleNamespace(screen_arms=[*declared, *ABLATIONS]),
                       declared, optional=ABLATIONS) == [*declared, *ABLATIONS]
    with pytest.raises(ValueError, match="are not declared arms of this study"):
        screen_arms(SimpleNamespace(screen_arms=list(ABLATIONS)), declared)


# --------------------------------------------------------------------------- #
# EVAL-ONLY re-score. The path exists for one situation: the weights are right
# and the EVALUATION was wrong. That happened twice (a head the MC loop
# never stacked, then a signal the borderline split never exported), and each
# time the fix was to re-read `best.pt` rather than train a new model and report
# it under the same run's name.
#
# It has to be tested here because on Colab it runs once, unattended, against
# hours of banked weights.
# --------------------------------------------------------------------------- #
class _Store:
    def __init__(self, tmp, has_best=True, stale=None):
        self.tmp, self.has_best, self.stale = tmp, has_best, stale
        self.cached = None
        self.loaded = False

    def path(self, tag, name):
        p = self.tmp / f"{tag}_{name}"
        if name == "best.pt" and self.has_best:
            p.write_text("weights")
        return p

    def cached_result(self, tag, schema=None):
        # schema=None is the "give me the stale one" call the re-score makes to
        # carry `best_val_auc` forward; with a schema it must miss, or the fold
        # returns early and never re-scores at all.
        return self.stale if schema is None else None

    def load_best(self, tag, model):
        self.loaded = True

    def cache_result(self, tag, result):
        self.cached = result


def _runner(store):
    from backend.notebooks.shared.runner import FoldRunner
    r = object.__new__(FoldRunner)
    r.store, r.schema, r.experiment = store, 9, None
    r.evaluate_only = True
    r.build_fold = lambda fold, tag, build: SimpleNamespace(model=object())
    r.release = lambda ctx: None
    return r


def _boom(ctx):
    raise AssertionError("evaluate_only must NOT train")


def test_eval_only_rescores_from_kept_weights_without_training(tmp_path):
    store = _Store(tmp_path, stale={"best_val_auc": 0.91, "selection": {"epoch": 7}})
    out = _runner(store).run_fold(
        3, "D_reg_fold3", build=lambda ctx: None, train_step=_boom,
        evaluate=lambda ctx: {"rho_test_panel_disp": 0.42})

    assert store.loaded, "the kept epoch's weights were never restored"
    assert out["rho_test_panel_disp"] == 0.42
    assert out["fold"] == 3
    # the run that produced the weights is carried forward, not recomputed from
    # a val pass that never happened
    assert out["best_val_auc"] == 0.91
    assert out["selection"] == {"epoch": 7}
    assert out["schema"] == 9
    # a re-scored number must never be mistakable for a freshly trained one
    assert out["rescored"] is True
    assert store.cached is out


def test_eval_only_refuses_when_there_are_no_kept_weights(tmp_path):
    """Silently training instead would be the worst outcome: hours of GPU, and
    a 'rescore' that is actually a different model."""
    with pytest.raises(FileNotFoundError, match="no kept weights"):
        _runner(_Store(tmp_path, has_best=False)).run_fold(
            0, "D_reg_fold0", build=lambda ctx: None, train_step=_boom,
            evaluate=lambda ctx: {})


def test_the_default_is_still_to_train(tmp_path):
    from backend.notebooks.shared.runner import FoldRunner
    assert FoldRunner.evaluate_only is False, (
        "every existing study must train exactly as before")


def test_logged_loss_terms_reconstruct_the_total_exactly(net, batch):
    """Every term the objective adds is logged UNWEIGHTED, so the weighted sum of
    the logged terms must equal the total it returned - otherwise the log line
    describes a different loss from the one that was optimised. Logging is
    read-only: the detached copies cannot change the total or its gradient."""
    cfg = with_panel(ExperimentConfig(), True)
    obj = Objective(cfg, cls_loss=nn.BCEWithLogitsLoss(), tau_d=0.5)
    out = net(batch["image"], return_all=True)
    loss, _ = obj(net, batch, out, 99, DEVICE, rng=np.random.default_rng(0))
    lam, t = obj.lambdas(99), obj.last_terms
    weight = {"cls": 1.0, "opinion": lam["opinion"],
              "defer": lam["defer"], "dispersion": lam["defer"]}
    assert {"cls", "opinion"} <= set(t)
    rebuilt = sum(weight[k] * float(v) for k, v in t.items())
    assert rebuilt == pytest.approx(float(loss.detach()), rel=1e-6)
    assert all(not v.requires_grad for v in t.values())


def test_predict_full_names_the_distribution_its_uncertainty_came_from(batch):
    """Without an opinion head the `aleatoric` / `epistemic` columns come from the
    classifier, which is not comparable with an opinion arm's columns - so the
    result must say which source filled them."""
    from field_disagreement.engine.runner import predict_full
    loader = [{"image": batch["image"], "label": batch["label"]}]
    with_op = build_model("cnn", pretrained=False, opinion_head=True, n_ratings=K)
    without = build_model("cnn", pretrained=False)
    assert predict_full(with_op, loader, DEVICE)["uncertainty_source"] == "opinion_entropy"
    assert predict_full(without, loader, DEVICE)["uncertainty_source"] == "none"
    assert predict_full(without, loader, DEVICE, mc_passes=2)["uncertainty_source"] == "mc_class"
    assert predict_full(with_op, loader, DEVICE, mc_passes=2)["uncertainty_source"] == "mc_opinion"


def test_signal_selection_keeps_the_best_validation_signal_inside_the_auc_band(loop_cfg, tmp_path):
    """The study's rule: every epoch here is in the val-AUC band (AUC 1.0 throughout), so
    the kept epoch is the one whose smoothed VALIDATION signal is highest, and the
    second rule (`sel_also: auc_band`, val AUC only) keeps the band's last epoch
    into band.pt from the same run."""
    from dataclasses import replace as _r
    cfg = SimpleNamespace(**{**vars(loop_cfg), "patience": 100, "sel_window": 1,
                             "selection": "auc_band_signal", "sel_also": "auc_band",
                             "sel_signal": "val_rho_fields"})
    store = SpyStore(tmp_path / "sig", "m", DEVICE, use_drive=False)
    runner, ctx, net = scripted(cfg, store, [PERFECT] * 10)
    sig = [0.1, 0.2, 0.5, 0.9, 0.3, 0.2, 0.4, 0.3, 0.2, 0.1]
    ctx.extras["val_monitor"] = lambda c: {"val_rho_fields": sig[c.epoch]}
    runner.fit(ctx, stepping(), selection="auc_band_signal", min_epochs=0)
    assert ctx.selection.epoch == 4 and ctx.selection.signal == 0.9
    assert ctx.alt_selection.epoch == 10 and ctx.alt_selection.rule == "auc_band"
    assert store.path("unit", "band.pt").exists()
    # an epoch outside the band can never be kept, however high its signal
    keep = _r(ctx.selection, val_auc_smooth=0.95, signal=0.1)
    out = _r(keep, val_auc_smooth=0.90, signal=0.99, epoch=11)
    assert not runner._keep(keep, out, anchor=0.95, handover=False)
    # and a kept epoch the rising anchor pushed out of the band gives way
    assert runner._keep(keep, _r(keep, val_auc_smooth=0.99, signal=0.0, epoch=12),
                        anchor=1.0, handover=False)
