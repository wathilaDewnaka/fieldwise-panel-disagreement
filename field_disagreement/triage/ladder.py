"""The arms: seven prior-art deferral methods, the claim, and its control.

    A_cls    routes on classifier confidence (softmax response)       prior art
    B_kl     opinion head, KL to the panel's rating histogram;        prior art
             routes on its spread (Raghu et al. 2019, UVC; Peterson et al. 2019)
    C_emd    as B_kl with the ordinal EMD loss                        prior art
    D_reg    dispersion head regressing the panel's disagreement on   prior art
             the diagnosis (Raghu et al. 2019, Direct Uncertainty Prediction)
    R_rr     classifier trained on ONE randomly drawn reader's call   prior art
             per item per step; routes on its confidence
    U_edl    evidential classifier (Sensoy et al. 2018): a Beta over  prior art
             the call, routes on its vacuity 2 / (alpha_0 + alpha_1)
    P_emd    one EMD rating histogram PER REPORT FIELD on C_emd's     prior art
             head; routes on the mean of the fields' predicted spreads
    H_fld    one disagreement output PER REPORT FIELD, each regressing ours (the claim)
             that field's panel disagreement (rank-normalised),
             PLUS each field's rating histogram (P_emd's
             target) on the same head; routes on the mean of its
             per-field disagreement outputs
    H_fldp   H_fld with every panel target (opinion histogram,        control
             per-field disagreements, field histograms) taken from one
             random other item

`P_emd` -> `H_fld` adds exactly the per-field disagreement target, so the claim
nests on the strongest per-field prior art.

Every panel term has ONE weight, `deferral.weight` (base.yaml `lam_def`, 0.3),
whichever arm it belongs to.

ABLATION (ABLATION_ARMS), trained only when a run names it in `screen_arms` and
reported beside the study, never inside its tables:

    H_disp   H_fld without the per-field histograms: the per-field
             disagreement outputs alone

Every arm is derived from ONE experiment config and differs from it only in the
fields its `Arm` names, so backbone, schedule, selection, folds and seed are
identical across arms by construction. The contract is the study protocol (paper, Methods and Appendix).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["Arm", "ARMS", "ABLATION_ARMS", "ABLATIONS", "TRAINED_ARMS", "ALL_ARMS",
           "CLAIM", "CONTROL", "PRIOR_ART", "DISPLAY_NAMES", "ships", "ladder_config",
           "rung_overrides"]


@dataclass(frozen=True)
class Arm:
    opinion: str = "none"            # opinion-head objective: none | kl | emd
    regress: bool = False            # a dispersion head regressing a panel d (MSE)
    target: str = "disagreement"     # its target: d | d_single | d_profile | opinion_fields
    transform: str = "none"          # none | rankgauss (the loss target only)
    shuffle: str = "none"            # none | all: another item's panel
    cls_target: str = "consensus"    # consensus | random_reader (one reader per step)
    head: str = "logit"              # logit | evidential (a Beta over the call)
    aux_fields: bool = False         # + per-field rating histograms on the dispersion head


ARMS = {
    "A_cls":  Arm(),
    "B_kl":   Arm(opinion="kl"),
    "C_emd":  Arm(opinion="emd"),
    "D_reg":  Arm(opinion="emd", regress=True, target="d_single"),
    "R_rr":   Arm(cls_target="random_reader"),
    "U_edl":  Arm(head="evidential"),
    "P_emd":  Arm(opinion="emd", regress=True, target="opinion_fields"),
    "H_fld":  Arm(opinion="emd", regress=True, target="d_profile", transform="rankgauss",
                  aux_fields=True),
    "H_fldp": Arm(opinion="emd", regress=True, target="d_profile", transform="rankgauss",
                  aux_fields=True, shuffle="all"),
}
ABLATION_ARMS = {
    "H_disp": Arm(opinion="emd", regress=True, target="d_profile", transform="rankgauss"),
}
ABLATIONS = tuple(ABLATION_ARMS)
CLAIM = "H_fld"
CONTROL = "H_fldp"
PRIOR_ART = ("A_cls", "B_kl", "C_emd", "D_reg", "R_rr", "U_edl", "P_emd")
ALL_ARMS = tuple(ARMS)

# The names the paper and the notebook's tables and figures use for each arm.
DISPLAY_NAMES = {
    "A_cls": "Confidence", "B_kl": "Soft-KL", "C_emd": "Soft-EMD", "D_reg": "DUP",
    "R_rr": "Random-reader", "U_edl": "Evidential", "P_emd": "Field-soft",
    "H_fld": "Field-disp (ours)", "H_fldp": "Field-disp shuffled",
    "A_cls_mc": "MC dropout",
    "H_disp": "Field-disp, no histograms",
}

# Every arm a run may train: the study's nine and the weight-matched ablations.
TRAINED_ARMS = {**ARMS, **ABLATION_ARMS}


def ships(arm: str) -> str:
    """The one panel-risk signal an arm routes on: its dispersion head, else the
    spread of its opinion histogram, else the classifier's confidence. The
    evidential arm routes on its vacuity, which the evaluation writes to the
    `epistemic` column."""
    a = TRAINED_ARMS[arm]
    if a.head == "evidential":
        return "epistemic"
    return "panel_disp" if a.regress else ("panel_spread" if a.opinion != "none"
                                           else "confidence")


def ladder_config(experiment, arm: str):
    """`experiment` with its opinion and deferral blocks set for `arm`."""
    a = TRAINED_ARMS[arm]
    on = a.opinion != "none"
    return replace(
        experiment,
        opinion=replace(experiment.opinion, enabled=on,
                        loss=a.opinion if on else experiment.opinion.loss,
                        panel_shuffle=a.shuffle),
        deferral=replace(experiment.deferral, enabled=a.regress,
                         regress_dispersion=a.regress, regress_target=a.target,
                         regress_transform=a.transform, aux_fields=a.aux_fields))


def rung_overrides(experiment, arm: str, rung: str) -> dict:
    """The {"block.field": value} changes that turn `arm` into `rung` on
    `experiment`: the opinion and deferral fields where their two
    `ladder_config`s differ. A study that swaps one study arm for another inside a
    larger design applies exactly these, so the swapped arm is the ladder's rung
    and not a hand-written approximation of it."""
    a, b = ladder_config(experiment, arm), ladder_config(experiment, rung)
    out = {}
    for block in ("opinion", "deferral"):
        x, y = getattr(a, block), getattr(b, block)
        for f in x.__dataclass_fields__:
            if getattr(x, f) != getattr(y, f):
                out[f"{block}.{f}"] = getattr(y, f)
    # Switches turned off go first, so no intermediate config pairs a field with
    # a setting it is only valid beside (aux_fields needs the d_profile target).
    return dict(sorted(out.items(), key=lambda kv: kv[1] is not False))
