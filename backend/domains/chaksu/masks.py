"""The array arithmetic behind one processed Chákṣu sample — numpy only, no
knowledge of the archive's layout (that is `prepare.py`'s job).

Counterpart of `lidc/patches.py`, split out for the same reason: unit-testable on
synthetic arrays, on a machine with no download. The panel reductions come from
`field_disagreement.spectrum`. Two things are specific to Chákṣu and live here:

  * `disc_window`, the optic-disc-centred square crop (about three disc radii),
    the standard fundus pipeline. Cropping also re-centres the target: the
    uncropped cohort is naturally decentred (optic-disc centroid a mean 0.185 of
    the frame from the centre).
  * `vote_targets`, the panel's opinion over a DECISION rather than over a mask.
    Chákṣu's five experts each say normal / glaucoma-suspect per image.

`disagreement` here is the s.d. of five BINARY votes. That is bounded by 0.5 and
takes only three values, {0, 0.400, 0.490} — see the warning in `prepare.py`.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from field_disagreement import spectrum

agreement_surface = spectrum.agreement_surface
mean_pairwise_dice = spectrum.mean_pairwise_dice

__all__ = ["agreement_surface", "mean_pairwise_dice", "to_grid",
           "MASK_THRESHOLD", "DECISION_LEVELS", "disc_window", "crop",
           "FOV_FRACTION", "locate_disc", "auto_window",
           "vote_targets"]

# 0/255 uint8 masks, so `> 0` and `> 0.5` select identically. Passed explicitly
# anyway.
MASK_THRESHOLD = 0.0

_FILTERS = {"bilinear": Image.BILINEAR, "nearest": Image.NEAREST}


def pad_to_square(arr: np.ndarray) -> np.ndarray:
    """Centre-pad an (H, W) or (H, W, C) array to a square with zeros."""
    h, w = arr.shape[:2]
    if h == w:
        return arr
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return np.pad(arr, [(top, side - h - top), (left, side - w - left)]
                  + [(0, 0)] * (arr.ndim - 2))


def to_grid(arr: np.ndarray, size: int, order: str = "bilinear",
            square: bool = True) -> np.ndarray:
    """Centre-pad to a square, then resize to (size, size).

    Padding first keeps the pixel grid isotropic: resizing a non-square frame
    straight to a square stretches one axis, so a mask boundary would mean
    different things in x and y.

    `order` is 'bilinear' for the image or 'nearest' for a label-like array (an
    annotator's mask). PIL applies the filter in the SOURCE resolution, so the
    8x downsample a 2048px frame needs is properly averaged rather than
    point-sampled."""
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        raise TypeError(f"this domain is 8-bit throughout; got {arr.dtype}")
    if order not in _FILTERS:
        raise ValueError(f"unknown resize order {order!r}; expected "
                         f"one of {sorted(_FILTERS)}")
    if square:
        arr = pad_to_square(arr)
    with Image.fromarray(arr) as im:
        return np.asarray(im.resize((size, size), _FILTERS[order]))


DECISION_LEVELS = (0, 1)        # the K=2 bins of opinion_dist: normal, glaucoma


def disc_window(masks: list[np.ndarray], radii: float = 3.0
                ) -> tuple[int, int, int]:
    """A square crop box (top, left, side) centred on the optic disc.

    `radii` is the width of the window in disc RADII, taken from the union of the
    experts' disc masks — 3.0 means "three radii across", i.e. one and a half on
    each side of the disc centre.

    Both the centre and the radius are AREA measures — the union's centroid, and
    the radius of the circle of the same area — rather than the centre and half
    the extent of its bounding box. Some masks in this archive carry a stray
    speck hundreds of pixels from the disc (three of the five on Bosch Image106),
    which leaves the area almost untouched but triples the bbox: that image's
    window came out 1102 px instead of ~370, shrinking the cup to 1% of the crop
    and undoing the whole reason for cropping.

    The box is clamped to stay inside the frame and never exceeds the shorter
    side, so the crop is always square and always real pixels: no padding, and
    therefore no border artefact."""
    if not masks:
        raise ValueError("no disc masks: nothing to centre the crop on")
    union = np.zeros(masks[0].shape, dtype=bool)
    for m in masks:
        union |= np.asarray(m) > 0

    ys, xs = np.nonzero(union)
    if len(ys) == 0:
        raise ValueError("the disc union is empty: nothing to centre the crop on")

    h, w = union.shape
    cy, cx = ys.mean(), xs.mean()
    radius = float(np.sqrt(len(ys) / np.pi))

    side = int(round(radii * radius))
    side = max(1, min(side, h, w))
    top = min(max(int(round(cy - side / 2.0)), 0), h - side)
    left = min(max(int(round(cx - side / 2.0)), 0), w - side)
    return top, left, side


FOV_FRACTION = 0.36     # auto window side, as a fraction of the field-of-view diameter


def locate_disc(rgb: np.ndarray) -> tuple[float, float, float]:
    """(cy, cx, fov_diameter) in native pixels, from the PHOTOGRAPH ALONE.

    The optic disc is the brightest extended structure inside the field of view:
    the red+green luminance is smoothed at about a disc's scale and its maximum
    taken inside the eroded field of view (the bright rim of the aperture is
    excluded). No expert mask is read, so neither the centre nor the scale can
    carry the panel's delineation disagreement - which the mask-based
    `disc_window` does: its side is 3x the radius of the experts' UNION, and the
    union grows exactly when they disagree on the disc boundary."""
    from PIL import Image
    from scipy import ndimage as ndi

    h, w = rgb.shape[:2]
    s = 512.0 / max(h, w)
    small = np.asarray(Image.fromarray(np.asarray(rgb, np.uint8)).resize(
        (max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BILINEAR), float)
    r = small[..., 0]
    fov = ndi.binary_opening(r > 0.15 * np.percentile(r, 99), iterations=3)
    lab, n = ndi.label(fov)
    if n:
        fov = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    diam = 2.0 * np.sqrt(max(int(fov.sum()), 1) / np.pi)
    inner = ndi.binary_erosion(fov, iterations=max(1, int(0.06 * diam)))
    # red channel at about a disc's scale, minus a wide background so a broadly
    # bright half of the field does not win; then the centroid of the bright blob
    # around the maximum (the maximum alone sits on the cup, off the disc centre).
    # Chosen on localisation accuracy over the cohort (whole disc in the window
    # 97.7%, median error 0.10 disc radii); no item's own masks enter its crop.
    lum = ndi.gaussian_filter(r, sigma=0.05 * diam) - ndi.gaussian_filter(r, sigma=0.25 * diam)
    lum[~inner] = -np.inf
    cy, cx = np.unravel_index(int(np.argmax(lum)), lum.shape)
    peak, floor = lum[cy, cx], float(np.median(lum[inner]))
    blob, _ = ndi.label(lum >= peak - 0.3 * (peak - floor))
    ys, xs = np.nonzero(blob == blob[cy, cx])
    return ys.mean() / s, xs.mean() / s, diam / s


def auto_window(rgb: np.ndarray, fov_fraction: float = FOV_FRACTION
                ) -> tuple[int, int, int]:
    """A square crop box (top, left, side) from `locate_disc`: centred on the
    detected disc, side a FIXED fraction of the field-of-view diameter. The same
    clamping as `disc_window` keeps it inside the frame."""
    h, w = rgb.shape[:2]
    cy, cx, diam = locate_disc(rgb)
    side = max(1, min(int(round(fov_fraction * diam)), h, w))
    top = min(max(int(round(cy - side / 2.0)), 0), h - side)
    left = min(max(int(round(cx - side / 2.0)), 0), w - side)
    return top, left, side


def crop(arr: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    """Apply a `disc_window` box to an (H, W) or (H, W, C) array."""
    top, left, side = box
    return np.asarray(arr)[top:top + side, left:left + side]


def vote_targets(votes) -> dict:
    """Every item-level scalar the DECISION panel supplies.

      votes           per expert: 1 = glaucoma suspect, 0 = normal
      opinion_dist    normalised histogram over DECISION_LEVELS (K = 2)
      label           the panel's majority; a tie reads as normal
      disagreement    s.d. of the votes — BOUNDED BY 0.5, and on a five-expert
                      panel it takes only the values {0, 0.400, 0.490}
      rating_entropy  Shannon entropy of opinion_dist (nats)

    This distribution is not degenerate: 41.5% of the cohort is non-unanimous
    and 14.6% is a genuine 2-3 split, so the K=2 opinion head has something to
    learn, and the label varies (14% positive).

    `votes` may be a mapping (expert -> vote) or a plain sequence; a mapping is
    read in its own key order, so `prepare.py` sorts the panel once and both the
    stored `votes` array and the `experts` string follow that order."""
    values = list(votes.values()) if isinstance(votes, dict) else list(votes)
    if not values:
        raise ValueError("no expert decisions to derive item-level targets from")
    v = np.asarray(values, dtype=np.int64)
    if not np.isin(v, DECISION_LEVELS).all():
        raise ValueError(f"votes must be 0/1, got {sorted(set(v.tolist()))}")

    odist = spectrum.opinion_histogram(v, DECISION_LEVELS)
    return {
        "votes": v.astype(np.int8),
        "opinion_dist": odist,
        # Strict `>`: on the 3 images with a four-expert panel a 2-2 tie is
        # NORMAL, the conservative reading of "the panel did not call it". NOT
        # `spectrum.median_label`, which records a tie as AMBIGUOUS rather than
        # resolving it.
        "label": int(2 * int(v.sum()) > v.size),
        "disagreement": spectrum.rating_spread(v),
        "rating_entropy": spectrum.rating_entropy(odist),
    }
