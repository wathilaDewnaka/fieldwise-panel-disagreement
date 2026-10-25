"""Attention pooling and the shared classifier / opinion / panel heads."""
from __future__ import annotations
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """1x1 conv -> softmax over the feature grid: attention-pooling weights.

    Returns a_grid (N, 1, h, w), which sums to 1 over the grid. The feature map is
    pooled with these weights into the vector every head reads (a learned-weight
    generalisation of global average pooling; at initialisation the near-uniform
    softmax reproduces it).
    """
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        a = self.conv(feat)                                  # (N, 1, h, w)
        n, _, h, w = a.shape
        return a.view(n, -1).softmax(dim=1).view(n, 1, h, w)   # spatial softmax (sum 1)


class _HeadMixin:
    """Shared opinion/deferral head construction + output assembly."""

    def _build_aux_heads(self, dim, p_drop, opinion_head, defer_head,
                         n_ratings=5, dispersion_head=False):
        """`n_ratings` is K, the width of the opinion distribution the panel
        expresses — it MUST match the `opinion_dist` the dataset supplies, or the
        head is trained against a target of a different length.

        It is a parameter and not a constant because K is a property of the
        annotation protocol, not of the model: a 1..5 severity scale gives K=5, a
        four-level grade scale K=4, and a panel that only says present/absent
        K=2. The default of 5 is the historical value, so a caller that does not
        pass it builds exactly the network it did before."""
        if n_ratings < 2:
            raise ValueError(f"n_ratings must be at least 2, got {n_ratings}")
        self.attn = SpatialAttention(self._feat_ch)
        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(dim, 1)
        self.opinion = nn.Linear(dim, n_ratings) if opinion_head else None
        self.defer = nn.Linear(dim, 3) if defer_head else None
        # The dispersion regressor: E[d | x], the panel's disagreement itself
        # rather than a threshold on it. `dispersion_head` is True for one
        # scalar, or an int P for one output per report field (d_profile).
        self.dispersion = (nn.Linear(dim, int(dispersion_head))
                           if dispersion_head else None)

    @staticmethod
    def _attn_pool(feat: torch.Tensor, a_grid: torch.Tensor) -> torch.Tensor:
        """Attention-weighted spatial pooling. a_grid sums to 1 over the grid, so
        this is a learned-weight generalisation of GAP (same scale; at init the
        near-uniform softmax reproduces GAP, a safe start)."""
        return (feat * a_grid).sum(dim=(2, 3))

    def _assemble(self, feat, z):
        logit = self.classifier(self.dropout(z)).squeeze(1)  # (N,)
        out = {"logit": logit, "feat": feat}
        if self.opinion is not None:
            out["opinion"] = self.opinion(z)                 # (N, K) logits
        if self.defer is not None:
            out["defer"] = self.defer(z)                     # (N, 3) logits
        if self.dispersion is not None:
            disp = self.dispersion(z)                        # (N, P)
            out["dispersion"] = disp.squeeze(1) if disp.shape[1] == 1 else disp
        return out
