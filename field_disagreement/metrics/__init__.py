"""Evaluation metrics (pure numpy, no torch/sklearn): classification AUC (rank-based,
tie-safe), calibration, selective prediction, opinion-distribution distances and the
fold-stratified, cluster-resampled bootstrap.
"""
from __future__ import annotations

from .classification import roc_auc, subgroup_metrics
from .calibration import (brier, nll, ece, fit_temperature, apply_temperature)
from .selective import (risk_coverage, risk_at_coverage, sensitivity_at_coverage,
                        coverage_at_sensitivity, error_concentration,
                        rank_error_auc, selective_metrics, compare_selectors,
                        contested_leakage,
                        SERVED_OUTCOMES, served_outcomes,
                        expected_reader_accuracy, team_accuracy)
from .distribution import (emd_distance, kl_divergence, distribution_entropy,
                           entropy_error, opinion_metrics, spread_error,
                           predicted_spread, dispersion_reliability,
                           dispersion_calibration_error,
                           fit_opinion_baselines, predict_opinion_baseline)
from .bootstrap import (bootstrap_auc_ci, disagreement_separation, rank_agreement,
                        aggregate_folds, paired_fold_test, paired_auc_bootstrap)
from .rank import spearman
from .paired import paired_t_interval, paired_comparison

__all__ = [
    "paired_t_interval",
    "paired_comparison",
    "roc_auc",
    "subgroup_metrics",
    "brier",
    "nll",
    "ece",
    "fit_temperature",
    "apply_temperature",
    "risk_coverage",
    "risk_at_coverage",
    "sensitivity_at_coverage",
    "coverage_at_sensitivity",
    "error_concentration",
    "rank_error_auc",
    "selective_metrics",
    "compare_selectors",
    "contested_leakage",
    "SERVED_OUTCOMES",
    "served_outcomes",
    "expected_reader_accuracy",
    "team_accuracy",
    "emd_distance",
    "kl_divergence",
    "distribution_entropy",
    "entropy_error",
    "opinion_metrics",
    "spread_error",
    "dispersion_reliability",
    "dispersion_calibration_error",
    "predicted_spread",
    "fit_opinion_baselines",
    "predict_opinion_baseline",
    "bootstrap_auc_ci",
    "disagreement_separation",
    "rank_agreement",
    "aggregate_folds",
    "paired_fold_test",
    "paired_auc_bootstrap",
    "spearman",
]
