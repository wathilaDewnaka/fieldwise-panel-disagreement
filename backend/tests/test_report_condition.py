"""The study's report-level cohorts (`lidc_report`, `chaksu_report`).

Pins the arithmetic of the two conditions, that the loader hands the
single-attribute d to `D_reg`, and that each rung regresses the target it
names.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from field_disagreement.config import ExperimentConfig, InvalidConfig
from backend.config import paths

P = paths()


def _files(key, n=25):
    root = Path(getattr(P, key, "") or "")
    fs = sorted(root.glob("*.npz"))[:n]
    if not fs:
        pytest.skip(f"{key} is not built on this machine")
    return fs


@pytest.mark.parametrize("key,parent,attrs", [
    ("lidc_report_processed", "lidc_multi_processed", 7),
    ("chaksu_report_processed", "chaksu_vcdr_processed", 10)])
def test_the_condition_is_its_parent_plus_the_report(key, parent, attrs):
    if not Path(getattr(P, parent, "") or "").is_dir():
        pytest.skip(f"{parent} is not built on this machine")
    for f in _files(key):
        z, zp = np.load(f, allow_pickle=True), np.load(Path(getattr(P, parent)) / f.name,
                                                       allow_pickle=True)
        assert z["d_profile"].shape == (attrs,)
        assert np.isclose(float(z["disagreement"]), float(z["d_profile"].mean()), atol=1e-5)
        assert np.array_equal(z["image"], zp["image"]) and int(z["label"]) == int(zp["label"])
        assert np.allclose(z["opinion_dist"], zp["opinion_dist"])
        # the single-attribute d is kept exactly
        single = (zp["d_per_attribute"][-1] if "d_per_attribute" in zp.files
                  else zp["disagreement"])
        assert np.isclose(float(z["d_single"]), float(single), atol=1e-6)


def test_the_loader_passes_the_single_attribute_d_and_the_profile_through():
    from backend.domains.adapter import PatchDataset
    f = _files("lidc_report_processed", 1)[0]
    item = PatchDataset(f.parent, [f.name], train=False)[0]
    z = np.load(f, allow_pickle=True)
    assert item["d_single"].ndim == 0
    assert np.isclose(float(item["d_single"]), float(z["d_single"]), atol=1e-6)
    # the per-field target: one disagreement per report field, its mean the report-level d
    assert item["d_profile"].shape == (7,)
    assert np.allclose(item["d_profile"].numpy(), z["d_profile"], atol=1e-6)


def test_other_cohorts_have_no_report_fields():
    from backend.domains.adapter import PatchDataset
    f = _files("processed", 1)[0]
    item = PatchDataset(f.parent, [f.name], train=False)[0]
    assert "d_single" not in item


def test_the_regression_target_is_validated():
    cfg = ExperimentConfig()
    with pytest.raises(InvalidConfig, match="regress_target"):
        replace(cfg.deferral, regress_target="nonsense")
    replace(cfg.deferral, regress_target="d_profile")      # the per-field target


def _objective(target):
    from field_disagreement.objectives import Objective
    cfg = ExperimentConfig()
    cfg = replace(cfg,
                  deferral=replace(cfg.deferral, enabled=True, regress_dispersion=True,
                                   regress_target=target))
    return Objective(cfg, cls_loss=torch.nn.BCEWithLogitsLoss(), tau_d=1.0)


@pytest.mark.parametrize("target", ["disagreement", "d_single"])
def test_each_target_is_the_one_regressed(target, monkeypatch):
    n = 12
    g = torch.Generator().manual_seed(0)
    b = {"label": torch.tensor([0, 1] * (n // 2)),
         "opinion_dist": torch.softmax(torch.randn(n, 5, generator=g), 1),
         "disagreement": torch.rand(n, generator=g) * 3,
         "d_single": torch.rand(n, generator=g)}
    out = {"logit": torch.zeros(n, requires_grad=True),
           "opinion": torch.zeros(n, 5, requires_grad=True),
           "dispersion": torch.zeros(n, requires_grad=True)}
    seen = {}
    real = torch.nn.functional.mse_loss

    def spy(pred, tgt, *a, **k):
        seen["pred"], seen["tgt"] = pred, tgt
        return real(pred, tgt, *a, **k)
    monkeypatch.setattr(torch.nn.functional, "mse_loss", spy)
    _objective(target)(None, b, out, epoch=100, device="cpu",
                       rng=np.random.default_rng(0))
    assert torch.equal(seen["tgt"], b[target])
    assert seen["pred"] is out["dispersion"]
