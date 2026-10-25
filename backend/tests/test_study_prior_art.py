"""The study's random-reader and evidential prior-art arms: they must do what their names say."""
import numpy as np
import torch
import torch.nn as nn

from field_disagreement.triage import ladder as L
from backend.domains.adapter import CALLS_PAD, PatchDataset
from backend.notebooks.shared import study_prior_art as X
from backend.notebooks.shared import study_results as R


# ---------------------------------------------------------------- the ladder
def test_only_these_two_arms_change_the_classifier():
    for a, arm in L.ARMS.items():
        assert (arm.cls_target == "random_reader") == (a == "R_rr"), a
        assert (arm.head == "evidential") == (a == "U_edl"), a


def test_extension_arms_route_on_their_own_signal():
    assert L.ships("R_rr") == "confidence"
    assert L.ships("U_edl") == "epistemic"
    assert {"R_rr", "U_edl"} <= set(L.PRIOR_ART) <= set(R.RIVALS)
    assert R.ARMS == L.ALL_ARMS


def test_extension_arms_switch_off_every_panel_term():
    from field_disagreement import ExperimentConfig
    base = ExperimentConfig()
    for arm in ("R_rr", "U_edl"):
        c = L.ladder_config(base, arm)
        assert not c.opinion.enabled and not c.deferral.enabled


# ------------------------------------------------------------ reader targets
def _npz(tmp_path, name, **arrays):
    np.savez(tmp_path / name, **arrays)
    return np.load(tmp_path / name)


def test_reader_targets_follow_the_cohort_rule(tmp_path):
    lidc = PatchDataset._reader_targets(_npz(tmp_path, "a.npz", malignancy=np.array([1, 3, 4])))
    assert lidc.shape == (CALLS_PAD,)
    assert lidc[:3].tolist() == [0.0, 0.5, 1.0] and (lidc[3:] == -1).all()
    votes = PatchDataset._reader_targets(_npz(tmp_path, "b.npz", votes=np.array([1, 0, 1, 1, 0])))
    assert votes[:5].tolist() == [1, 0, 1, 1, 0] and (votes[5:] == -1).all()


def test_random_reader_draws_only_real_readers_and_is_reproducible():
    t = torch.tensor([[0.0, 1.0, -1, -1], [0.5, 0.5, 0.5, -1]])
    a = X.sample_reader_targets(t, np.random.default_rng(7))
    b = X.sample_reader_targets(t, np.random.default_rng(7))
    assert torch.equal(a, b)
    assert a[0].item() in (0.0, 1.0) and a[1].item() == 0.5
    draws = {X.sample_reader_targets(t, np.random.default_rng(s))[0].item() for s in range(40)}
    assert draws == {0.0, 1.0}


def test_soft_target_bce_weights_positive_calls_only():
    f = X.soft_target_bce(3.0)
    logit = torch.zeros(3)
    pos = f(logit, torch.tensor([1.0, 1.0, 1.0]))
    neu = f(logit, torch.tensor([0.5, 0.5, 0.5]))
    assert torch.isclose(pos, torch.tensor(np.log(2.0), dtype=pos.dtype))
    assert neu > 0 and torch.isfinite(neu)


# --------------------------------------------------------------- evidential
class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def param_groups(self, a, b):
        return [{"params": list(self.parameters()), "lr": b}]

    def forward(self, x, return_all=False):
        return {"logit": x[:, 0], "dispersion": self.lin(x)}


def test_evidential_net_reads_its_evidence():
    net = X.EvidentialNet(_Tiny())
    out = net(torch.randn(5, 4), return_all=True)
    assert out["logit"].shape == (5,) and out["evidence"].shape == (5, 2)
    v = out["dispersion"]
    assert ((v > 0) & (v <= 1)).all()
    assert net.param_groups(1e-4, 3e-4)[0]["lr"] == 3e-4      # delegated to the net
    assert torch.equal(net(torch.ones(2, 4)), net(torch.ones(2, 4), return_all=True)["logit"])


def test_edl_loss_rewards_correct_evidence():
    y = torch.tensor([1, 0])
    good = X.edl_loss(torch.tensor([[0.0, 20.0], [20.0, 0.0]]), y, epoch=20, pos_weight=2.0)
    bad = X.edl_loss(torch.tensor([[20.0, 0.0], [0.0, 20.0]]), y, epoch=20, pos_weight=2.0)
    assert torch.isfinite(good) and good < bad
    # the KL term is annealed: off at epoch 0
    e0 = X.edl_loss(torch.tensor([[20.0, 0.0]]), torch.tensor([1]), epoch=0, pos_weight=1.0)
    e9 = X.edl_loss(torch.tensor([[20.0, 0.0]]), torch.tensor([1]), epoch=9, pos_weight=1.0)
    assert e9 > e0
