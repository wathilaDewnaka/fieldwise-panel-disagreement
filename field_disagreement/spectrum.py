"""Panel reductions: a panel of masks or ratings -> the quantities the study uses.

  * spatial agreement (`agreement_surface`, `agreement_from_count`): the per-pixel
    share of readers who marked a pixel, and their union. Preprocessing stores it,
    and training uses it only to clamp the translation augmentation so the readers'
    marks stay in the frame.
  * item-level opinion (`opinion_histogram`, `rating_spread`, ...): the panel's
    rating histogram and its dispersion, which the arms are trained on.

Each domain decides how an annotator's mark becomes a mask and what an annotator's
opinion is; how a panel is reduced is shared here, so every cohort reduces it the
same way (the same threshold for "marked", the same tie rule).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# spatial agreement: a panel of masks -> A(x, y)
# --------------------------------------------------------------------------- #
def agreement_from_count(count: np.ndarray, n_annotators: int
                         ) -> tuple[np.ndarray, np.ndarray]:
    """(frac, binary) from a per-pixel annotator count.

    `frac = count / R` is what makes an item read by 3 annotators comparable with
    one read by 4 — a raw count of 3 is unanimous on the former and 3-of-4 on the
    latter. R is always the panel that read THIS item, never a cohort constant.

    Separate from `agreement_surface` for a domain that accumulates into a larger
    canvas and crops, and so already holds the count."""
    if n_annotators <= 0:
        raise ValueError(f"n_annotators must be positive, got {n_annotators}")
    count = np.asarray(count)
    return count.astype(np.float32) / n_annotators, (count >= 1).astype(np.uint8)


def agreement_surface(masks, threshold: float = 0.5
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A(x, y) for one item: (count, frac, binary) from its annotators' masks.

    `threshold` is what counts as marked, and is a real parameter: masks arrive
    as booleans, as 8-bit 0/255, and as float or uint16 from medical-image
    readers, where `> 0` and `> 0.5` are not the same question.

    Masks must share a shape — a domain that resizes does so per annotator and
    only then counts, because resizing the COUNT map interpolates between
    annotator counts and invents agreement levels nobody voted for."""
    masks = list(masks)
    if not masks:
        raise ValueError(
            "no annotator masks: an item must have at least one reader")
    shapes = {np.shape(m) for m in masks}
    if len(shapes) != 1:
        raise ValueError(f"annotator masks disagree on shape: {sorted(shapes)}")

    count = np.sum([(np.asarray(m) > threshold) for m in masks],
                   axis=0, dtype=np.uint8)
    frac, binary = agreement_from_count(count, len(masks))
    return count, frac, binary


def mean_pairwise_dice(masks, threshold: float = 0.5) -> float:
    """Mean Dice over every unordered pair of annotators; 1.0 for a lone one, and
    1.0 for two empty masks — they agree that there is nothing there."""
    masks = [np.asarray(m) > threshold for m in masks]
    areas = [int(m.sum()) for m in masks]
    scores = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            denom = areas[i] + areas[j]
            inter = int((masks[i] & masks[j]).sum())
            scores.append(1.0 if denom == 0 else 2.0 * inter / denom)
    return float(np.mean(scores)) if scores else 1.0


# --------------------------------------------------------------------------- #
# item-level opinion: a panel of ratings -> q, d, H
# --------------------------------------------------------------------------- #
def opinion_histogram(values, levels) -> np.ndarray:
    """Normalised histogram over `levels` — the (K,) `q`.

    `levels` is declared, not inferred: a level nobody picked for this item must
    still occupy its bin, or K varies per item and the head trains against
    targets of changing length. An off-scale value raises rather than being
    dropped, which is how a panel of 4 quietly becomes a panel of 3."""
    levels = list(levels)
    v = np.asarray(values).ravel()
    if not v.size:
        raise ValueError("no annotator values to build an opinion distribution from")
    counts = np.zeros(len(levels), dtype=np.float32)
    for x in v:
        try:
            counts[levels.index(type(levels[0])(x))] += 1.0
        except ValueError:
            raise ValueError(
                f"value {x!r} is not one of the declared levels {levels}") from None
    return counts / counts.sum()


def rating_entropy(dist) -> float:
    """Shannon entropy of an opinion histogram, in NATS.

    Zero bins are skipped rather than clipped, so a unanimous panel scores
    exactly 0. Not `metrics.distribution_entropy`, which scores a batch and
    divides by log K to compare predictions across different K."""
    q = np.asarray(dist, dtype=np.float64).ravel()
    nz = q[q > 0]
    return float(-(nz * np.log(nz)).sum())


def rating_spread(values) -> float:
    """s.d. of the panel's ratings — the spread for a domain that HAS a rating
    scale. Unbounded above."""
    return float(np.std(np.asarray(values, dtype=np.float64)))


def median_label(values, ambiguous, positive: int = 1, negative: int = 0,
                 ambiguous_code: int = -1) -> int:
    """Binary class from the panel's median, keeping an explicit AMBIGUOUS code.

    A median landing exactly on the undecided level is the case the panel
    declined to call; collapsing it into either class invents a decision nobody
    made. It is excluded from every supervised binary loss and evaluated
    separately."""
    m = float(np.median(np.asarray(values, dtype=np.float64)))
    if m > ambiguous:
        return int(positive)
    if m < ambiguous:
        return int(negative)
    return int(ambiguous_code)
