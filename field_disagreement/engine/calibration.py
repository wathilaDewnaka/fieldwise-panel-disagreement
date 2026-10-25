"""A fitted calibrator as an OBJECT, not a number in a config file.

`field_disagreement.triage.conformal` already has the LAC arithmetic. What it does not have
is somewhere for the fitted threshold to live. A `q_hat` copied by hand into a
config file as a bare float would carry a coverage guarantee with no record of
the risk level it was fitted at, how many calibration points stood behind it, or
which class order the probabilities were in. A wrong paste is undetectable and
still looks like a guarantee.

`Calibrator` closes that: it is fitted from a held-out split, it travels as a
dict, and it refuses to imply a guarantee it does not have. An unfitted
calibrator (`q_hat is None`) collapses `C(x)` to the argmax singleton and reports
`alpha=None`, which is the honest degenerate case rather than a fabricated
interval.

Nothing here knows what the classes are: `class_names` is supplied by the caller
and defaults to neutral placeholders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..triage import conformal as _conformal

DEFAULT_CLASS_NAMES: tuple[str, ...] = ("negative", "positive")


@dataclass
class Calibrator:
    """Split-conformal (LAC) threshold plus the provenance that makes it readable.

    `q_hat` is the only thing the prediction set needs; everything else exists so
    that a calibrator loaded from disk can be checked against the run that
    produced it instead of trusted blindly.
    """

    q_hat: float | None = None
    alpha: float = 0.1
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
    n_calibration: int = 0
    empirical_coverage: float | None = None
    meta: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Truthy only once actually fitted, so `if calibrator:` reads correctly."""
        return self.q_hat is not None

    # -- fitting ------------------------------------------------------------ #
    @classmethod
    def fit(cls, probs, labels, alpha: float = 0.1,
            class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
            **meta) -> "Calibrator":
        """Calibrate on a HELD-OUT split — never the split the model trained on.

        `probs` is either the (N,) positive-class probability or a full (N, K)
        distribution; `labels` are integer class indices. Exchangeability between
        this split and deployment is what the guarantee rests on, so a calibration
        split drawn from the same preprocessing as the served inputs is not a
        detail, it is the assumption.

        The empirical coverage is recorded ON THE CALIBRATION SPLIT, which is
        optimistic by construction — it is stored as a sanity check that the fit
        ran, not as evidence the guarantee holds. Report coverage on test.
        """
        p = np.asarray(probs, dtype=float)
        dist = _conformal.binary_probs(p) if p.ndim == 1 else p
        y = np.asarray(labels, dtype=int).ravel()
        if len(dist) != len(y):
            raise ValueError(f"{len(dist)} probabilities against {len(y)} labels")
        if dist.shape[1] != len(class_names):
            raise ValueError(
                f"{dist.shape[1]} probability columns against "
                f"{len(class_names)} class names {list(class_names)}")

        q_hat = _conformal.lac_calibrate(dist, y, alpha=alpha)
        return cls(q_hat=q_hat, alpha=float(alpha),
                   class_names=tuple(class_names), n_calibration=int(len(y)),
                   empirical_coverage=_conformal.empirical_coverage(dist, y, q_hat),
                   meta=dict(meta))

    # -- inference ---------------------------------------------------------- #
    def prediction_set(self, prob) -> list[str]:
        """C(x) = { y : p_y >= 1 - q_hat } for ONE case.

        Accepts the scalar positive-class probability or a full (K,) distribution.

        The EMPTY set is returned as empty. It is a real and informative outcome —
        at this risk level no label is plausible, i.e. the case is atypical of
        everything the threshold was calibrated on — and injecting the argmax to
        avoid an awkward result would hide exactly the cases most worth escalating.
        `decide_or_defer` turns it into a forced deferral.
        """
        p = np.asarray(prob, dtype=float).ravel()
        dist = _conformal.binary_probs(p)[0] if p.size == 1 else p
        if self.q_hat is None:
            return [self.class_names[int(dist.argmax())]]
        keep = dist >= (1.0 - self.q_hat)
        return [c for c, k in zip(self.class_names, keep) if k]

    def label_for(self, prob) -> str:
        """The argmax label, independent of the set — what the model would say if
        forced to choose. Reported alongside C(x), never instead of it."""
        p = np.asarray(prob, dtype=float).ravel()
        dist = _conformal.binary_probs(p)[0] if p.size == 1 else p
        return self.class_names[int(dist.argmax())]

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict:
        """Plain JSON/YAML-safe mapping — this is the on-disk form."""
        return {"q_hat": self.q_hat, "alpha": self.alpha,
                "class_names": list(self.class_names),
                "n_calibration": self.n_calibration,
                "empirical_coverage": self.empirical_coverage,
                "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, block: dict | None) -> "Calibrator":
        """Rebuild from `to_dict` output. `None` or `{}` gives an UNFITTED
        calibrator rather than an error, so a decision can be built before a run
        has produced a calibration."""
        if not block:
            return cls()
        names = block.get("class_names") or DEFAULT_CLASS_NAMES
        q = block.get("q_hat", block.get("conformal_qhat"))
        return cls(q_hat=None if q is None else float(q),
                   alpha=float(block.get("alpha", 0.1)),
                   class_names=tuple(names),
                   n_calibration=int(block.get("n_calibration", 0)),
                   empirical_coverage=(None if block.get("empirical_coverage") is None
                                       else float(block["empirical_coverage"])),
                   meta=dict(block.get("meta") or {}))
