"""The study says the same thing in the framework, the config, the notebook
and the scorer: the arms and what separates them, the proposed arm and its
control, the one run config, and the declared constants.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from field_disagreement.triage import ladder as L
from backend.notebooks.shared import study_arms as SARM
from backend.notebooks.shared import study_results as SRES

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "backend" / "notebooks" / "train_evaluate.ipynb"
CONFIGS = ROOT / "configs" / "notebooks"
# The nine arms of the study, in table order.
STUDY = ("A_cls", "B_kl", "C_emd", "D_reg", "R_rr", "U_edl", "P_emd", "H_fld", "H_fldp")


@pytest.fixture(scope="module")
def notebook() -> str:
    cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_the_study_is_exactly_the_declared_nine():
    assert tuple(L.ARMS) == L.ALL_ARMS == STUDY
    assert L.CLAIM == "H_fld" and L.CONTROL == "H_fldp"
    assert L.PRIOR_ART == STUDY[:7]
    # beyond the nine, only the declared weight-matched ablations can be trained
    assert L.TRAINED_ARMS == {**L.ARMS, **L.ABLATION_ARMS}


def test_the_scorer_reads_the_framework_ladder():
    assert SRES.ARMS == STUDY
    assert (SRES.CLAIM, SRES.CONTROL, SRES.PRIOR_ART) == (L.CLAIM, L.CONTROL, L.PRIOR_ART)
    assert SRES.RIVALS == L.PRIOR_ART + (L.CONTROL,)
    assert SRES.TRIAGE_RIVALS == L.PRIOR_ART + (SRES.MC_DROPOUT,)   # + MC dropout, no network


def _diff(a, b) -> dict:
    x, y = asdict(L.TRAINED_ARMS[a]), asdict(L.TRAINED_ARMS[b])
    return {k: (x[k], y[k]) for k in x if x[k] != y[k]}


@pytest.mark.parametrize("hi,lo,what", [
    ("C_emd", "B_kl", {"opinion": ("emd", "kl")}),
    ("D_reg", "C_emd", {"regress": (True, False), "target": ("d_single", "disagreement")}),
    ("P_emd", "C_emd", {"regress": (True, False), "target": ("opinion_fields", "disagreement")}),
    # the claim nests on P_emd: it adds the per-field disagreement target
    ("H_fld", "P_emd", {"target": ("d_profile", "opinion_fields"),
                        "transform": ("rankgauss", "none"),
                        "aux_fields": (True, False)}),
    ("H_fldp", "H_fld", {"shuffle": ("all", "none")}),
])
def test_each_rung_changes_only_what_it_names(hi, lo, what):
    assert _diff(hi, lo) == what


def test_the_regression_rungs_sit_on_the_c_emd_opinion_head():
    for a in ("D_reg", "P_emd", "H_fld", "H_fldp"):
        assert L.ARMS[a].opinion == L.ARMS["C_emd"].opinion == "emd"
        assert L.ARMS[a].regress is True
        assert L.ships(a) == "panel_disp"
    assert L.ships("A_cls") == "confidence" and L.ships("B_kl") == "panel_spread"


def test_ladder_config_switches_exactly_the_arms_fields():
    from field_disagreement import ExperimentConfig
    base = ExperimentConfig()
    for a, arm in L.ARMS.items():
        c = L.ladder_config(base, a)
        assert c.opinion.enabled == (arm.opinion != "none")
        assert c.opinion.panel_shuffle == arm.shuffle
        assert c.deferral.regress_dispersion is arm.regress
        assert c.deferral.regress_target == arm.target
        assert c.deferral.regress_transform == arm.transform
        assert (c.model, c.training, c.optimizer) == (
            base.model, base.training, base.optimizer)


def test_stopping_rule_is_the_same_for_every_arm():
    assert "min_epochs=int(cfg.defer_warmup * cfg.epochs)" in inspect.getsource(SARM.run_fold)
    # selection comes from the config: each arm's own validation signal inside
    # the val-AUC band, with the AUC-only band scored too as a sensitivity check
    assert "selection=experiment.training.selection" in inspect.getsource(SARM.run_fold)
    run = yaml.safe_load((CONFIGS / "study.yaml").read_text(encoding="utf-8"))
    assert (run["selection"], run["sel_also"], run["sel_signal"]) == (
        "auc_band_signal", "auc_band", "val_rho_fields")
    # the selection floor is epoch 10, not 30: at 30 every arm was forced past its
    # val-AUC peak
    base = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))
    assert base["defer_warmup"] * base["epochs"] == 10


def test_the_notebook_holds_no_study_logic(notebook):
    assert "SRES.report(" in notebook and "SARM.run_fold(" in notebook
    for gone in ("ARMS = {", "def evaluate_arm", "def build",
                 "STUDY_CONFIG", "STUDY_ARMS", "STUDY_FOLDS", "STUDY_SMOKE"):
        assert gone not in notebook, gone


def test_the_run_config_is_report_level_and_restates_no_cohort():
    a = yaml.safe_load((CONFIGS / "study.yaml").read_text(encoding="utf-8"))
    assert a["report_level"] is True
    for k in ("dataset", "require_dataset", "folds", "contested_d",
              "neutral_rating", "decentring", "shift_frac"):
        assert k not in a, f"{k!r} belongs to the cohort, not the study"


@pytest.mark.parametrize("cohort,report,d0", [("lidc", "lidc_report", 1.8174),
                                              ("chaksu", "chaksu_report", 2.3349)])
def test_the_study_binds_base_yamls_cohort_at_report_level(cohort, report, d0):
    import contextlib
    import io
    from backend.config import load
    from backend.domains import datasets as DS
    cfg = load("notebooks/study")
    cfg.dataset = cohort
    with contextlib.redirect_stdout(io.StringIO()):
        spec = DS.bind(cfg, require_data=False)
    assert spec.name == report and spec.report_level
    assert cfg.contested_d == d0
    # the run folder is named after the cohort, so `bind` cannot adopt another
    # cohort's cached results.
    assert cfg.run_dir.startswith(f"study_{report}_dc20")


def test_certificate_levels_are_declared_in_config():
    base = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))
    for key in ("cert_alpha", "cert_betas", "cert_delta"):
        assert key in base, key
    for gone in ("lam_rank", "rank_margin", "panel_items", "shuffle_d"):
        assert gone not in base, f"withdrawn knob {gone!r} is back in base.yaml"


def test_the_certificate_does_not_calibrate_on_the_selection_split():
    from backend.notebooks.shared import panel
    assert "split_for_calibration(test" in inspect.getsource(panel.panel_battery)
