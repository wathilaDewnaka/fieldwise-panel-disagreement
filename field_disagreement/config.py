"""Typed experiment configuration: what a run is, checked before it starts.

Without checks, a misconfiguration surfaces as a result rather than an error:
`alpha: 1.5` gives a conformal set that is always everything, and an opinion head
on a cohort with a constant panel trains against a constant and reports a clean
null. Each of those costs a training run to discover and none of them raises.

These dataclasses are that configuration with the invariants attached. They are
grouped by component, so an ablation is a change to one block:

    OpinionConfig       the panel's rating distribution
    DeferralConfig      the panel-disagreement head and its schedule
    UncertaintyConfig   MC-dropout predictive uncertainty
    ConformalConfig     the split-conformal coverage guarantee

`ExperimentConfig.from_namespace` builds one from the flat YAML the project
already has, so the configs stay the documented, locked files they are and this
becomes a validated VIEW of them rather than a second source of truth.

Nothing here knows what the images are or which cohort is loaded. What a config
NEEDS from a dataset is `needed_capabilities()`, which the domain checks against
the cohort it is about to bind - that is how "do not train an opinion head on a
cohort whose panel never disagrees" becomes an error at the config cell.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


OPINION_LOSSES = ("kl", "emd")
CLS_LOSSES = ("bce", "focal")
SELECTION_RULES = ("auc", "auc_band", "auc_band_signal")


class InvalidConfig(ValueError):
    """A configuration that cannot describe a run, raised at construction."""


def _positive(name: str, value: float) -> None:
    if value < 0:
        raise InvalidConfig(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class ModelConfig:
    """The network, and the one property of it the DATA decides.

    `n_ratings` is K, the width of the opinion distribution the annotation
    protocol produces - 5 for a 1..5 severity scale, 2 for present/absent. It
    belongs to the cohort, not the model, which is why it is a field and not a
    constant: a head built at the wrong K trains against a target of another
    length.
    """

    backbone: str = "resnet50"
    pretrained: bool = True
    n_ratings: int = 5
    feat_stage: str = "layer3"
    # Override the loader's input size. None (default) = the backbone's registered
    # size, which is what every run of the paper used. An int feeds the model a
    # larger frame, which moves the feature grid with it: resnet50 at 224 puts
    # layer3 on 14x14, at 448 on 28x28.
    img_size: int | None = None

    def __post_init__(self):
        if self.n_ratings < 2:
            raise InvalidConfig(
                f"model.n_ratings must be at least 2, got {self.n_ratings}")


@dataclass(frozen=True)
class OpinionConfig:
    """Learning the panel's rating histogram instead of one hard label.

    "kl" treats the K categories as unordered; "emd" treats them as the ordinal
    scale they are, so putting mass on rating 4 when the panel said 5 is less
    wrong than putting it on rating 1. The two disagree, which is why the study
    runs both.
    """

    enabled: bool = True
    loss: str = "emd"
    weight: float = 1.0
    # The study's circularity control: "all" trains every item on a RANDOM other
    # item's whole panel (one permutation per batch, applied to every panel
    # target). "disp" permutes ONLY the dispersion targets and keeps each image's
    # own rating histogram, isolating the per-field information. "none" is every
    # other arm.
    panel_shuffle: str = "none"

    def __post_init__(self):
        if self.loss not in OPINION_LOSSES:
            raise InvalidConfig(
                f"opinion.loss must be one of {list(OPINION_LOSSES)}, "
                f"got {self.loss!r}")
        _positive("opinion.weight", self.weight)
        if self.panel_shuffle not in ("none", "all", "disp"):
            raise InvalidConfig(
                f"opinion.panel_shuffle must be none|all|disp, "
                f"got {self.panel_shuffle!r}")


REGRESS_TARGETS = ("disagreement", "d_single", "d_profile", "d_diag", "opinion_fields")


@dataclass(frozen=True)
class UncertaintyConfig:
    """MC-dropout predictive uncertainty.

    `samples` = 0 means a single deterministic pass, and the epistemic term is
    then exactly 0 - honest rather than missing, since one deterministic model
    has no spread to measure.
    """

    mc_dropout: bool = True
    samples: int = 20

    def __post_init__(self):
        if self.samples < 0:
            raise InvalidConfig(
                f"uncertainty.samples must be >= 0, got {self.samples}")


@dataclass(frozen=True)
class ConformalConfig:
    """Split-conformal (LAC) prediction sets at miscoverage `alpha`."""

    enabled: bool = True
    alpha: float = 0.1

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise InvalidConfig(
                f"conformal.alpha must be strictly between 0 and 1 (it is a "
                f"miscoverage rate; target coverage is 1 - alpha), "
                f"got {self.alpha}")


@dataclass(frozen=True)
class DeferralConfig:
    """The decide-or-defer rule and the head that feeds it.

    `warmup_frac` exists because deferral supervision only means something
    relative to a classifier worth deferring FROM: at full strength from step 0
    the two objectives fight for the trunk when the classification signal is
    weakest.
    """

    enabled: bool = True
    # The full weight of the arm's panel term (dispersion regression, per-field
    # histograms, or both), the same for EVERY arm (0.3 in the paper).
    weight: float = 0.3
    warmup_frac: float = 0.3
    target_coverage: float = 0.8
    threshold: float = 0.5
    # REGRESS the panel dispersion (MSE) instead of thresholding it into the
    # 3-way head. Default False = the historical arms, bit for bit.
    regress_dispersion: bool = False
    # WHICH dispersion the regression head learns: the cohort's own `d`
    # ("disagreement"; report-level on `lidc_report` / `chaksu_report`), the
    # diagnosis field's `d_single`, or `d_profile`, one value per report field
    # (the head then has one output per field), `d_diag`, the diagnostic
    # field's column of `d_profile` (unrounded, unlike `d_single` on Chákṣu), or
    # `opinion_fields`: one rating histogram PER FIELD, learned with the opinion
    # loss (the head then has P x K outputs; the per-field soft-label baseline).
    regress_target: str = "disagreement"
    # How the target is scaled for the LOSS only. "rankgauss" regresses its
    # normal scores against this fold's TRAINING targets (`triage.rank_gauss`):
    # monotone, so the evaluated order is unchanged, while a heavy right tail
    # no longer dominates the squared error. "none" is every other arm.
    regress_transform: str = "none"
    # With the per-field target (`d_profile`), ALSO learn each field's rating
    # histogram on extra outputs of the same head (P x K after the P
    # dispersions), under the opinion loss and the same ramped weight. An
    # auxiliary target only: the arm still routes on, and is scored by, the P
    # dispersions. False = every other arm.
    aux_fields: bool = False

    def __post_init__(self):
        if self.aux_fields and self.regress_target != "d_profile":
            raise InvalidConfig("deferral.aux_fields needs regress_target: d_profile")
        if self.regress_target not in REGRESS_TARGETS:
            raise InvalidConfig(
                f"deferral.regress_target must be one of {REGRESS_TARGETS}, "
                f"got {self.regress_target!r}")
        if self.regress_transform not in ("none", "rankgauss"):
            raise InvalidConfig(
                f"deferral.regress_transform must be none|rankgauss, "
                f"got {self.regress_transform!r}")
        _positive("deferral.weight", self.weight)
        if not 0.0 <= self.warmup_frac <= 1.0:
            raise InvalidConfig(
                f"deferral.warmup_frac is a fraction of training and must be "
                f"in [0, 1], got {self.warmup_frac}")
        if not 0.0 < self.target_coverage <= 1.0:
            raise InvalidConfig(
                f"deferral.target_coverage must be in (0, 1], "
                f"got {self.target_coverage}")


@dataclass(frozen=True)
class TrainingConfig:
    """The training PROTOCOL: how long, when to stop, how the loss is weighted.

    Split from the contribution blocks because it is the half that must NOT
    differ between arms - two arms compared under different patience are not a
    comparison.
    """

    epochs: int = 100
    warmup: int = 1                # linear LR warm-up epochs
    patience: int = 15             # early stop after this many epochs with no
                                   # val-AUC improvement
    ckpt_every: int = 1            # epochs between resume checkpoints
    # Which epoch a fold keeps.
    #
    # "auc" = the best val-AUC epoch. Arm-neutral, but it latches onto a single
    # lucky epoch, which can come before an arm's own objective reached full weight.
    #
    # "auc_band" = the LAST epoch whose SMOOTHED val AUC is within `sel_tol` of
    # the best. Taking the last member of the band gives each arm the most exposure
    # to its own objective that its classifier can carry. Only val AUC is ever
    # read, so no outcome metric and no per-arm objective can reach the choice.
    # This is the rule the study uses.
    selection: str = "auc_band"
    # "auc_band_signal" = inside the same val-AUC band, the epoch whose smoothed
    # VALIDATION `sel_signal` (a per-epoch value the study's val monitor logs) is
    # highest; patience also waits while that signal is still rising inside the
    # band. For a study whose arms all ship the signal being scored (here: each
    # arm's own disagreement signal) it gives every arm - the prior art included -
    # its own best validation epoch, with the classifier held inside the parity
    # band. Test is never read, and no arm is selected on a quantity it does not
    # itself produce.
    sel_signal: str = "val_rho_fields"
    # A SECOND rule tracked in the same run (its weights in `band.pt`), so a study
    # can report its result under both rules from one training run. None = off.
    sel_also: str | None = None
    # Smoothing window for the val-AUC series, used by patience AND by the band.
    # 1 = no smoothing, and then a single spike anchors the band onto itself.
    sel_window: int = 5
    # Width of the val-AUC band. This is the PARITY tolerance - epochs inside it
    # count as classifiers of equal quality - so it stays small enough that the
    # band cannot buy an arm a materially worse classifier.
    sel_tol: float = 0.01
    amp: bool = True               # mixed precision; inert off CUDA
    loss: str = "bce"              # class-weighted BCE, or focal
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    def __post_init__(self):
        if self.epochs < 1:
            raise InvalidConfig(f"training.epochs must be >= 1, got {self.epochs}")
        if self.loss not in CLS_LOSSES:
            raise InvalidConfig(
                f"training.loss must be one of {list(CLS_LOSSES)}, "
                f"got {self.loss!r}")
        if self.selection not in SELECTION_RULES:
            raise InvalidConfig(
                f"training.selection must be one of {list(SELECTION_RULES)}, "
                f"got {self.selection!r}")
        if self.sel_also is not None and self.sel_also not in SELECTION_RULES:
            raise InvalidConfig(
                f"training.sel_also must be one of {list(SELECTION_RULES)} or "
                f"null, got {self.sel_also!r}")
        if self.sel_window < 1:
            raise InvalidConfig(
                f"training.sel_window must be >= 1, got {self.sel_window}")
        if self.sel_tol <= 0:
            raise InvalidConfig(
                f"training.sel_tol is the width of the val-AUC parity band and "
                f"must be > 0, got {self.sel_tol}")
        if self.ckpt_every < 1:
            raise InvalidConfig(
                f"training.ckpt_every must be >= 1, got {self.ckpt_every}")
        for name in ("warmup", "patience"):
            _positive(f"training.{name}", getattr(self, name))


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW, with the differential learning rate the backbones are built for.

    `lr_trunk` is the rate for a PRETRAINED trunk, which wants a smaller step
    than the parts learned from scratch; `lr_head` covers everything else. A
    model with no `param_groups` has no such split and trains entirely at
    `lr_head`. The flat config names these `lr_vit`/`lr_head` after the backbone
    it was written for; the typed names say what they mean instead.
    """

    lr_trunk: float = 1e-4
    lr_head: float = 3e-4
    weight_decay: float = 0.05

    def __post_init__(self):
        for name in ("lr_trunk", "lr_head", "weight_decay"):
            _positive(f"optimizer.{name}", getattr(self, name))


@dataclass(frozen=True)
class DataLoaderConfig:
    """How batches are drawn - NOT where they come from.

    The cohort, its storage layout and the frozen split stay with the domain.
    What is protocol, and so has to be identical across arms, is the batch size
    the gradient is averaged over and the shift the inputs are drawn under.
    """

    batch_size: int = 32
    num_workers: int = 0
    # Decentring: the fraction of width an input may be shifted by. 0.0 is off,
    # and is the historical behaviour bit for bit.
    shift_frac: float = 0.0
    shift_seed: int = 0

    def __post_init__(self):
        if self.batch_size < 1:
            raise InvalidConfig(
                f"data.batch_size must be >= 1, got {self.batch_size}")
        _positive("data.num_workers", self.num_workers)
        if not 0.0 <= self.shift_frac < 1.0:
            raise InvalidConfig(
                f"data.shift_frac is a fraction of width and must be in "
                f"[0, 1), got {self.shift_frac}")


@dataclass(frozen=True)
class ExperimentConfig:
    """One run: the model, the contributions switched on, and the protocol.

    Frozen, so an arm of an ablation is `replace(base, opinion=...)` and can
    never be a mutation of the config another arm is still using.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    opinion: OpinionConfig = field(default_factory=OpinionConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    deferral: DeferralConfig = field(default_factory=DeferralConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    seed: int = 42

    @property
    def epochs(self) -> int:
        """The run length, which lives in `training`.

        Kept as a property because the deferral ramp is defined as a fraction of
        TOTAL training, so the objective has to read it; routing that through
        the one field in `training` is what stops a run whose schedule was
        computed against a different length than the loop actually ran.
        """
        return self.training.epochs

    # -- what this run needs the DATA to supply ----------------------------- #
    def needed_capabilities(self) -> tuple[str, ...]:
        """Capability names the cohort must support for this run to mean anything.

        The domain checks these against the cohort it binds. Keeping the list
        here rather than in the notebook is what stops "which studies can run
        on which cohort" from being answered in more than one place.

        The deferral head needs the PANEL, not the opinion head. Its target is
        built from how divided the panel was (`triage.build_deferral_targets`
        reads the batch's disagreement), never from the opinion head's output, so
        deferral without the opinion head is a real arm: learning WHEN to defer
        without learning WHAT the panel said. This used to be refused in
        `__post_init__`, which conflated the head with the data it is trained on;
        the data half of that rule is what is kept, here.
        """
        needed = ["classification"]
        if self.opinion.enabled or self.deferral.enabled:
            needed.append("opinion_distribution")
        return tuple(needed)

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict:
        """Plain JSON/YAML-safe mapping - written beside a checkpoint so the
        weights and the settings that produced them travel together."""
        return asdict(self)

    @classmethod
    def from_dict(cls, block: dict) -> ExperimentConfig:
        """Rebuild from `to_dict`. Unknown keys are an error, not a shrug: they
        are the symptom of a config written by a different version."""
        blocks = {"model": ModelConfig,
                  "opinion": OpinionConfig,
                  "uncertainty": UncertaintyConfig, "conformal": ConformalConfig,
                  "deferral": DeferralConfig, "training": TrainingConfig,
                  "optimizer": OptimizerConfig, "data": DataLoaderConfig}
        kwargs = {name: kind(**block[name])
                  for name, kind in blocks.items() if name in block}
        if "seed" in block:
            kwargs["seed"] = block["seed"]
        # `epochs` was a top-level scalar before the training protocol was typed.
        # A result.json written then is still a record of a real run, so it is
        # read into its new home rather than rejected as an unknown key.
        if "epochs" in block:
            kwargs["training"] = replace(
                kwargs.get("training", TrainingConfig()),
                epochs=int(block["epochs"]))
        unknown = set(block) - set(blocks) - {"seed", "epochs"}
        if unknown:
            raise InvalidConfig(f"unknown config keys {sorted(unknown)}")
        return cls(**kwargs)

    # -- the bridge from the project's flat YAML ---------------------------- #
    @classmethod
    def from_namespace(cls, cfg) -> ExperimentConfig:
        """Build from the flat `backend.config.load(...)` namespace.

        The YAML files stay the documented, locked protocol they already are;
        this is a validated VIEW of them, so there is no second place a value
        can be written down and no migration to get wrong. Every default here is
        the one the flat config documents.
        """
        def get(name, default):
            # The YAML uses `null` to mean "unset, take the default".
            value = getattr(cfg, name, default)
            return default if value is None else value

        opinion_loss = get("opinion_loss", "emd")
        return cls(
            model=ModelConfig(
                backbone=get("model", "resnet50"),
                pretrained=bool(get("pretrained", True)),
                n_ratings=int(get("n_ratings", 5)),
                feat_stage=get("feat_stage", "layer3"),
                img_size=get("img_size", None)),
            # `opinion_loss: none` is how the flat config switches the panel terms off.
            opinion=OpinionConfig(
                enabled=opinion_loss != "none",
                loss=(opinion_loss if opinion_loss != "none" else "emd"),
                weight=float(get("lam_op", 1.0))),
            uncertainty=UncertaintyConfig(
                mc_dropout=int(get("mc_passes", 0)) > 0,
                samples=int(get("mc_passes", 0))),
            conformal=ConformalConfig(alpha=float(get("alpha", 0.1))),
            deferral=DeferralConfig(
                enabled=opinion_loss != "none",
                weight=float(get("lam_def", 0.3)),
                warmup_frac=float(get("defer_warmup", 0.3)),
                target_coverage=float(get("target_coverage", 0.8))),
            training=TrainingConfig(
                epochs=int(get("epochs", 100)),
                warmup=int(get("warmup", 1)),
                patience=int(get("patience", 15)),
                ckpt_every=int(get("ckpt_every", 1)),
                selection=get("selection", "auc_band"),
                sel_signal=get("sel_signal", "val_rho_fields"),
                sel_also=get("sel_also", None),
                sel_window=int(get("sel_window", 5)),
                sel_tol=float(get("sel_tol", 0.01)),
                # The flat config states this negatively, as `no_amp`.
                amp=not bool(get("no_amp", False)),
                loss=get("loss", "bce"),
                focal_alpha=float(get("focal_alpha", 0.25)),
                focal_gamma=float(get("focal_gamma", 2.0))),
            optimizer=OptimizerConfig(
                lr_trunk=float(get("lr_vit", 1e-4)),
                lr_head=float(get("lr_head", 3e-4)),
                weight_decay=float(get("weight_decay", 0.05))),
            data=DataLoaderConfig(
                batch_size=int(get("batch", 32)),
                num_workers=int(get("workers", 0)),
                shift_frac=float(get("shift_frac", 0.0)),
                shift_seed=int(get("shift_seed", 0))),
            seed=int(get("seed", 42)))

    @classmethod
    def of(cls, cfg) -> ExperimentConfig:
        """This config, from whichever layer the caller happens to hold.

        The project has two configuration layers in flight - the flat YAML
        namespace the notebook loads, and this typed one - and every consumer
        would otherwise have to know which it was handed. One resolver here is
        what lets the runner, the optimiser and the façade all be written
        against the typed config without breaking the notebook.
        """
        return cfg if isinstance(cfg, cls) else cls.from_namespace(cfg)
