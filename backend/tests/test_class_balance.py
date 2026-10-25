"""Class imbalance is corrected once, from the training split only.

`class_balance: weighted_loss` (every reported run) up-weights positives in the
loss; `balanced_sampler` draws training batches with equal class odds and leaves
the loss unweighted. Neither may read validation or test."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from backend.domains import adapter as D
from backend.notebooks.shared.training import balanced_sampling, pos_weight_for


def _df(n_neg=60, n_pos=10):
    files = [f"f{i}.npz" for i in range(n_neg + n_pos)]
    return pd.DataFrame({"file": files, "label": [0] * n_neg + [1] * n_pos,
                         "patient_id": [f"p{i}" for i in range(n_neg + n_pos)]})


def test_the_default_is_the_reported_protocol():
    assert not balanced_sampling(SimpleNamespace())
    assert not balanced_sampling(SimpleNamespace(class_balance="weighted_loss"))
    with pytest.raises(ValueError):
        balanced_sampling(SimpleNamespace(class_balance="smote"))


def test_weighted_loss_uses_the_training_ratio_and_the_sampler_switches_it_off():
    df = _df()
    train = df.file.tolist()
    assert pos_weight_for(df, train) == pytest.approx(6.0)
    assert pos_weight_for(df, train, SimpleNamespace(class_balance="weighted_loss")) == 6.0
    assert pos_weight_for(df, train, SimpleNamespace(class_balance="balanced_sampler")) == 1.0


def test_the_sampler_draws_both_classes_equally_from_train_files_only():
    df = _df()
    train = df.file.tolist()[:50] + df.file.tolist()[60:66]      # 50 neg, 6 pos
    s = D.balanced_sampler(df, train)
    assert len(s) == len(train)
    torch.manual_seed(0)
    drawn = [train[i] for _ in range(200) for i in s]
    lab = df.set_index("file").loc[drawn, "label"].to_numpy()
    assert abs(lab.mean() - 0.5) < 0.03                          # balanced odds
    assert set(drawn) <= set(train)                              # never outside train


def test_only_the_train_loader_is_resampled(monkeypatch):
    df = _df()
    split = {"train": df.file.tolist()[:40], "val": df.file.tolist()[40:50],
             "test": df.file.tolist()[50:], "score3": [], "ambiguous": []}
    monkeypatch.setattr(D.S, "make_fold", lambda *a, **k: split)
    monkeypatch.setattr(D.S, "assert_no_leakage", lambda *a, **k: None)
    L = D.make_loaders("unused", df, {}, 0, balanced=True)
    assert isinstance(L["train"].sampler, torch.utils.data.WeightedRandomSampler)
    for k in ("val", "test", "score3"):
        assert isinstance(L[k].sampler, torch.utils.data.SequentialSampler)
    off = D.make_loaders("unused", df, {}, 0, balanced=False)
    assert isinstance(off["train"].sampler, torch.utils.data.RandomSampler)
