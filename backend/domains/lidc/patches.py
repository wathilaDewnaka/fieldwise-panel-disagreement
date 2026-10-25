"""The array arithmetic behind a processed nodule — pure numpy, no pylidc.

Split out of `prepare.py` so the parts that decide what a training example
*means* (the agreement surface, the normalisation, the rating targets, the
score-3 rule) can be unit-tested on synthetic input, on any machine, without the
DICOM tree or the pylidc/SQLAlchemy stack. `prepare.py` keeps the I/O: query
scans, read volumes, call these, write .npz.
"""
from __future__ import annotations

import numpy as np

from field_disagreement import spectrum


def window_hu(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to the lung window and scale to [0, 1]."""
    x = np.clip(x.astype(np.float32), lo, hi)
    return (x - lo) / (hi - lo)


def pad_and_crop(img: np.ndarray, ci: int, cj: int, size: int) -> np.ndarray:
    """Crop a (size, size) patch centred at (ci, cj). Zero-pad if out of bounds."""
    H, W = img.shape
    half = size // 2
    i0, i1 = ci - half, ci - half + size
    j0, j1 = cj - half, cj - half + size

    pi0 = max(0, -i0); pj0 = max(0, -j0)
    pi1 = max(0, i1 - H); pj1 = max(0, j1 - W)
    i0c, i1c = max(0, i0), min(H, i1)
    j0c, j1c = max(0, j0), min(W, j1)

    patch = img[i0c:i1c, j0c:j1c]
    if pi0 or pi1 or pj0 or pj1:
        patch = np.pad(patch, ((pi0, pi1), (pj0, pj1)))
    return patch


def agreement_surface(reader_masks: list[np.ndarray], bbox, full_shape: tuple[int, int],
                      centre: tuple[int, int], patch_size: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A(x, y) for one nodule, as the three maps that are saved.

    `reader_masks` are the per-reader 2D masks at the centre axial slice, each
    laid out in `bbox` (the (rows, cols) slice pair the masks are expressed in).
    Returns:

      count   (P, P) uint8    how many readers marked each pixel, 0..R
      frac    (P, P) float32  count / R, in [0, 1]
      binary  (P, P) uint8    union of readers — 1 where ANY reader marked

    `frac` is the normalisation that makes a 3-reader nodule comparable with a
    4-reader one: a raw count of 3 is unanimous on the former and 3-of-4 on the
    latter, so the raw count must never be compared across nodules. R is taken
    from the number of masks supplied, i.e. the readers who actually annotated
    THIS nodule, never a constant 4."""
    if not reader_masks:
        raise ValueError("no reader masks: a nodule must have at least one annotation")

    count_full = np.zeros(full_shape, dtype=np.uint8)
    for m in reader_masks:
        count_full[bbox] += m.astype(np.uint8)

    # Accumulate-then-crop rather than `spectrum.agreement_surface`: the masks
    # arrive laid out in `bbox` on a much larger canvas, not as same-shaped
    # arrays. The NORMALISATION is still the framework's.
    count = pad_and_crop(count_full, centre[0], centre[1], patch_size)
    frac, binary = spectrum.agreement_from_count(count, len(reader_masks))
    return count, frac, binary


# The K=5 bins of opinion_dist. Stated as levels rather than implied by a
# bincount slice, so an off-scale score RAISES instead of being dropped from the
# histogram while still counting toward `disagreement` and the label.
RATING_LEVELS = (1, 2, 3, 4, 5)

# The clinically undecided level. A panel whose median lands exactly here is the
# "score-3" case.
AMBIGUOUS_SCORE = 3


def label_from_scores(scores: np.ndarray) -> int:
    """Standard LIDC rule on the median of the per-reader malignancy scores:
    > 3 malignant (1), < 3 benign (0), == 3 ambiguous (-1).

    A median of exactly 3 is the "score-3" case: genuinely undecided by the
    panel, never a binary training target, evaluated separately."""
    return spectrum.median_label(scores, AMBIGUOUS_SCORE)


def opinion_distribution(scores: np.ndarray) -> np.ndarray:
    """Normalised histogram over the malignancy categories 1..5 (sums to 1)."""
    return spectrum.opinion_histogram(scores, RATING_LEVELS)


def rating_targets(scores: np.ndarray) -> dict:
    """Every target derived from the per-reader ratings alone: the opinion
    distribution (soft label), the spread that supervises deferral, its
    entropy, and the binary label."""
    scores = np.asarray(scores, dtype=np.int64)
    odist = opinion_distribution(scores)
    return {
        "opinion_dist": odist,
        "disagreement": spectrum.rating_spread(scores),
        "rating_entropy": spectrum.rating_entropy(odist),
        "label": label_from_scores(scores),
    }
