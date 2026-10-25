"""Opinion-distribution learning, panel-aware deferral, predictive
uncertainty, and the split-conformal coverage guarantee.

The four pieces are meant to be read as one decision path::

    opinion head        q_hat over the K ratings        -> aleatoric (panel spread)
    MC dropout          T stochastic passes             -> epistemic (model spread)
    conformal LAC       C(x) with 1-alpha coverage      -> ambiguous / atypical
    deferral policy     all of the above + confidence   -> decide or defer

`opinion_kl_loss` / `opinion_emd_loss` / `deferral_ce_loss` are the training-time
objectives; `DeferralPolicy`, `uncertainty` and `conformal` are what runs at
inference, and none of them reads an annotation.
"""
from __future__ import annotations

from .opinion import (OPINION_LOSSES, opinion_kl_loss, opinion_emd_loss,
                      opinion_loss, opinion_entropy,
                      opinion_spread)
from .deferral import (rank_gauss, build_deferral_targets,
                       deferral_ce_loss,
                       defer_lambda, confidence_gap, SOTA_MISS, SERVED_MISS, NEGATIVE_VETO, OBVIOUS_NEGATIVE,
                       EXTREME_VETO,
                       safety_cut, veto_cut, vetoable, served_refer, ServedRule,
                       conformal_ambiguity,
                       build_signals, prediction_signals, fit_deferral_policy,
                       candidate_rules, DeferralPolicy)
from .uncertainty import (mc_dropout, mc_dropout_predict, has_dropout,
                          entropy_decomposition, binary_decomposition,
                          predictive_variance, summarise)
from . import conformal, uncertainty, risk_control
from .risk_control import Certificate, calibrate, dual_calibrate, apply_certificate

from . import ladder

__all__ = [
    # opinion-distribution learning
    "OPINION_LOSSES",
    "opinion_kl_loss",
    "opinion_emd_loss",
    "opinion_loss",
    "opinion_entropy",
    "opinion_spread",
    # deferral: supervised head + learned policy
    "rank_gauss",
    "build_deferral_targets",
    "deferral_ce_loss",
    "defer_lambda",
    "confidence_gap",
    "SOTA_MISS",
    "SERVED_MISS",
    "NEGATIVE_VETO",
    "OBVIOUS_NEGATIVE",
    "EXTREME_VETO",
    "vetoable",
    "safety_cut",
    "veto_cut",
    "served_refer",
    "ServedRule",
    "conformal_ambiguity",
    "build_signals",
    "prediction_signals",
    "fit_deferral_policy",
    "candidate_rules",
    "DeferralPolicy",
    # predictive uncertainty
    "mc_dropout",
    "mc_dropout_predict",
    "has_dropout",
    "entropy_decomposition",
    "binary_decomposition",
    "predictive_variance",
    "summarise",
    # sub-modules
    "conformal",
    "uncertainty",
    # risk control over the auto-decided set
    "risk_control",
    "Certificate",
    "calibrate",
    "dual_calibrate",
    "apply_certificate",
    # the study arm ladder
    "ladder",
]
