"""the rank-normalised regression target (`deferral.regress_transform`)."""
import numpy as np
import pytest
import torch

from field_disagreement import triage
from field_disagreement.config import DeferralConfig, InvalidConfig


def test_rank_gauss_is_monotone_and_keeps_the_order():
    ref = torch.sort(torch.rand(500) ** 4 * 8)[0]          # heavy right tail
    x = torch.rand(200) ** 4 * 8
    z = triage.rank_gauss(x, ref)
    o = torch.argsort(x)
    assert torch.all(z[o][1:] >= z[o][:-1])                 # never reorders
    assert torch.isfinite(z).all()


def test_rank_gauss_ties_share_one_score_and_the_median_is_zero():
    ref = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    z = triage.rank_gauss(torch.tensor([0.0, 0.0, 1.0]), ref)
    assert z[0] == z[1]                                     # mid-rank for ties
    assert abs(float(z[2])) < 1e-6                          # 1.0 is the median


def test_rank_gauss_is_bounded_outside_the_reference():
    ref = torch.arange(10, dtype=torch.float32)
    z = triage.rank_gauss(torch.tensor([-100.0, 100.0]), ref)
    assert torch.isfinite(z).all() and z[0] < 0 < z[1]


def test_transform_defaults_to_none_and_rejects_unknown():
    assert DeferralConfig().regress_transform == "none"
    with pytest.raises(InvalidConfig):
        DeferralConfig(regress_transform="log")
    DeferralConfig(regress_transform="rankgauss", regress_target="d_single")


def test_objective_refuses_rankgauss_without_a_training_reference():
    from dataclasses import replace
    from field_disagreement.config import ExperimentConfig
    from field_disagreement.objectives import Objective
    base = ExperimentConfig()
    cfg = replace(base, deferral=replace(base.deferral, enabled=True,
                                         regress_dispersion=True,
                                         regress_transform="rankgauss"))
    obj = Objective(cfg, cls_loss=torch.nn.BCEWithLogitsLoss())
    assert obj.r_ref is None
    ref = np.sort(np.random.default_rng(0).gamma(1.0, size=100)).astype(np.float32)
    obj2 = Objective(cfg, cls_loss=torch.nn.BCEWithLogitsLoss(), r_ref=ref)
    assert obj2.r_ref is ref


def test_a_per_field_dispersion_head_has_one_output_per_field():
    from field_disagreement.registry import build_model
    x = torch.randn(3, 1, 64, 64)
    m = build_model("cnn", in_ch=1, dispersion_head=7).eval()
    assert m(x, return_all=True)["dispersion"].shape == (3, 7)
    m1 = build_model("cnn", in_ch=1, dispersion_head=True).eval()
    assert m1(x, return_all=True)["dispersion"].shape == (3,)   # the scalar arms: unchanged


def test_a_per_field_target_is_rank_normalised_field_by_field():
    """Each field is ranked against its OWN training column, so a field on a
    larger scale cannot dominate the loss."""
    from dataclasses import replace
    from field_disagreement.config import ExperimentConfig
    from field_disagreement.objectives import Objective
    rng = np.random.default_rng(0)
    base = ExperimentConfig()
    cfg = replace(base, opinion=replace(base.opinion, enabled=False),
                  deferral=replace(base.deferral, enabled=True, regress_dispersion=True,
                                   regress_target="d_profile",
                                   regress_transform="rankgauss"))
    tr = np.stack([rng.gamma(1.0, size=200), 100 * rng.gamma(1.0, size=200)], 1)
    obj = Objective(cfg, cls_loss=torch.nn.BCEWithLogitsLoss(),
                    r_ref=np.sort(tr, axis=0).astype(np.float32))
    prof = torch.tensor(tr[:4], dtype=torch.float32)
    want = torch.stack([triage.rank_gauss(prof[:, j], torch.as_tensor(obj.r_ref[:, j].copy()))
                        for j in range(2)], 1)
    assert want[:, 1].abs().max() < 4                       # scale-free, like field 0
    out = {"logit": torch.zeros(4), "dispersion": want.clone().requires_grad_()}
    batch = {"label": torch.tensor([0, 1, 0, 1]), "disagreement": prof.mean(1),
             "d_profile": prof}
    loss, _ = obj(None, batch, out, cfg.epochs, "cpu")      # past the defer warm-up
    assert obj.lambdas(cfg.epochs)["defer"] > 0
    bce = torch.nn.BCEWithLogitsLoss()(torch.zeros(4), batch["label"].float())
    assert torch.isclose(loss, bce, atol=1e-5)               # perfect per-field fit: MSE 0
