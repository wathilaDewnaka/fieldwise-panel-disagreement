"""LIDC-IDRI preprocessing contract tests.

Everything here runs on synthetic arrays, so it passes on a machine that has
never seen the LIDC-IDRI DICOM tree. The tests that need the real dataset live
in `test_dataset.py`; Chákṣu has `test_preprocessing_chaksu.py`.

Covered:
  * agreement-map construction (reader count, union mask, level sets)
  * normalisation when fewer than four radiologists annotated a nodule
  * mask / image dimensions, including out-of-bounds padding
  * rating-distribution generation (opinion_dist, disagreement, entropy)
  * score-3 classification, and that score-3 is never a training target
  * patient leakage across the frozen folds
  * one spatial transform shared by image, mask and agreement surface
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from field_disagreement import splits as S

from backend.config import ROOT, load
from backend.domains.lidc.patches import (agreement_surface, label_from_scores,
                                          opinion_distribution, pad_and_crop,
                                          rating_targets, window_hu)

PATCH = load("preprocessing/preprocess_lidc").patch_size
FULL = (128, 128)                 # stand-in "volume slice" the masks live in
CENTRE = (64, 64)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def disc(radius: float, shape=FULL, centre=CENTRE) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius ** 2).astype(np.uint8)


def nested_readers(n: int) -> list[np.ndarray]:
    """`n` readers who agree on the core and disagree at the rim: reader k
    outlines a disc of radius 10 - k. Every pixel's reader count is then known
    in closed form, which is what makes the agreement surface checkable."""
    return [disc(10 - k) for k in range(n)]


def full_bbox():
    return (slice(0, FULL[0]), slice(0, FULL[1]))


# --------------------------------------------------------------------------- #
# agreement-map construction
# --------------------------------------------------------------------------- #
def test_agreement_map_is_the_per_pixel_reader_count():
    readers = nested_readers(4)
    count, frac, binary = agreement_surface(readers, full_bbox(), FULL, CENTRE, PATCH)

    # the centre pixel is inside every reader's outline -> all 4 agree
    c = PATCH // 2
    assert count[c, c] == 4
    # a pixel only the widest reader marked (radius 10, not 9) -> exactly 1
    assert count[c, c + 10] == 1
    assert count[c, c + 11] == 0
    # counts are the sum of the individual masks, everywhere
    expected = sum(pad_and_crop(r, *CENTRE, PATCH).astype(int) for r in readers)
    assert np.array_equal(count.astype(int), expected)


def test_binary_mask_is_the_union_of_readers():
    readers = nested_readers(3)
    count, _, binary = agreement_surface(readers, full_bbox(), FULL, CENTRE, PATCH)
    union = np.clip(sum(pad_and_crop(r, *CENTRE, PATCH).astype(int) for r in readers), 0, 1)
    assert np.array_equal(binary, union)
    # union == "at least one reader", i.e. the 1-level set of the count map
    assert np.array_equal(binary.astype(bool), count >= 1)


def test_agreement_level_sets_are_nested():
    """A stricter consensus must be recoverable by thresholding the count map —
    the count map is the panel's agreement surface."""
    count, _, _ = agreement_surface(nested_readers(4), full_bbox(), FULL, CENTRE, PATCH)
    areas = [(count >= k).sum() for k in range(1, 5)]
    assert areas == sorted(areas, reverse=True)
    assert areas[-1] > 0


def test_agreement_surface_rejects_a_nodule_with_no_readers():
    with pytest.raises(ValueError):
        agreement_surface([], full_bbox(), FULL, CENTRE, PATCH)


# --------------------------------------------------------------------------- #
# normalisation with fewer than four radiologists
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_readers", [1, 2, 3, 4])
def test_agreement_frac_normalises_by_the_readers_of_THIS_nodule(n_readers):
    """The checklist item: a 3-reader nodule must not be scaled as if 4 people
    looked at it. Unanimous means 1.0 whether the panel was 3 or 4."""
    count, frac, _ = agreement_surface(nested_readers(n_readers),
                                       full_bbox(), FULL, CENTRE, PATCH)
    c = PATCH // 2
    assert count[c, c] == n_readers          # every reader marked the core
    assert frac[c, c] == pytest.approx(1.0)  # ... so agreement there is total
    assert np.allclose(frac, count.astype(np.float32) / n_readers)


def test_agreement_frac_is_bounded_and_comparable_across_panel_sizes():
    _, frac3, _ = agreement_surface(nested_readers(3), full_bbox(), FULL, CENTRE, PATCH)
    _, frac4, _ = agreement_surface(nested_readers(4), full_bbox(), FULL, CENTRE, PATCH)
    for f in (frac3, frac4):
        assert f.dtype == np.float32
        assert f.min() >= 0.0 and f.max() <= 1.0
    # both panels are unanimous on the core, so the normalised surfaces agree
    # there even though the raw counts (3 vs 4) do not
    core = disc(6.0)[:PATCH, :PATCH]  # shape-compatible sub-region indicator
    c = PATCH // 2
    assert frac3[c, c] == frac4[c, c] == pytest.approx(1.0)
    assert core.shape == (PATCH, PATCH)


def test_raw_count_alone_would_be_ambiguous():
    """Guards the reason `agreement_frac` exists: the same raw count means
    different things on a 3- and a 4-reader nodule."""
    count3, frac3, _ = agreement_surface(nested_readers(3), full_bbox(), FULL, CENTRE, PATCH)
    count4, frac4, _ = agreement_surface(nested_readers(4), full_bbox(), FULL, CENTRE, PATCH)
    c = PATCH // 2
    # a pixel 8 from centre sits inside the r=10, 9 and 8 outlines on both panels
    assert count3[c, c] == 3 and count4[c, c + 8] == 3      # identical raw counts
    assert frac3[c, c] == pytest.approx(1.00)               # unanimous
    assert frac4[c, c + 8] == pytest.approx(0.75)           # 3 of 4 — not unanimous


# --------------------------------------------------------------------------- #
# dimensions
# --------------------------------------------------------------------------- #
def test_maps_have_the_configured_patch_size():
    count, frac, binary = agreement_surface(nested_readers(4), full_bbox(),
                                            FULL, CENTRE, PATCH)
    for m in (count, frac, binary):
        assert m.shape == (PATCH, PATCH)
    assert count.dtype == np.uint8 and binary.dtype == np.uint8


def test_patch_is_padded_not_shrunk_at_the_volume_edge():
    """A nodule near the border still yields a full-size patch, zero-padded."""
    edge = (2, 2)
    count, frac, binary = agreement_surface([disc(6, centre=edge)], full_bbox(),
                                            FULL, edge, PATCH)
    assert count.shape == binary.shape == frac.shape == (PATCH, PATCH)
    assert count[0, 0] == 0                       # padded region is empty
    assert count[PATCH // 2, PATCH // 2] == 1     # nodule centre landed centred


def test_image_stack_has_2n_plus_1_channels_matching_the_maps():
    cfg = load("preprocessing/preprocess_lidc")
    vol = np.random.default_rng(0).integers(-1200, 600, size=(*FULL, 9)).astype(np.int16)
    stack = np.stack([pad_and_crop(vol[:, :, z], *CENTRE, cfg.patch_size)
                      for z in range(4 - cfg.neighbours, 5 + cfg.neighbours)], axis=0)
    image = window_hu(stack, cfg.hu_min, cfg.hu_max)
    assert image.shape == (2 * cfg.neighbours + 1, cfg.patch_size, cfg.patch_size)
    assert image.dtype == np.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    # the maps share the last two axes with the image -> one coordinate frame
    count, _, _ = agreement_surface(nested_readers(3), full_bbox(), FULL, CENTRE, PATCH)
    assert image.shape[-2:] == count.shape


# --------------------------------------------------------------------------- #
# rating distribution
# --------------------------------------------------------------------------- #
def test_opinion_distribution_is_a_normalised_histogram_over_1_to_5():
    d = opinion_distribution(np.array([1, 2, 2, 5]))
    assert d.shape == (5,)
    assert d.sum() == pytest.approx(1.0)
    assert np.allclose(d, [0.25, 0.5, 0.0, 0.0, 0.25])


def test_rating_targets_capture_spread_and_entropy():
    unanimous = rating_targets(np.array([4, 4, 4, 4]))
    assert unanimous["disagreement"] == pytest.approx(0.0)
    assert unanimous["rating_entropy"] == pytest.approx(0.0)
    assert unanimous["label"] == 1

    split = rating_targets(np.array([1, 2, 4, 5]))
    assert split["disagreement"] > unanimous["disagreement"]
    assert split["rating_entropy"] == pytest.approx(np.log(4))   # four equal bins
    assert split["opinion_dist"].sum() == pytest.approx(1.0)


def test_disagreement_is_the_population_std_of_the_ratings():
    scores = np.array([1, 3, 4, 5])
    assert rating_targets(scores)["disagreement"] == pytest.approx(float(np.std(scores)))


# --------------------------------------------------------------------------- #
# score-3 classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scores, expected", [
    ([5, 5, 4, 4], 1),          # median 4.5 -> malignant
    ([1, 1, 2, 2], 0),          # median 1.5 -> benign
    ([3, 3, 3, 3], -1),         # unanimous 3 -> ambiguous
    ([1, 3, 3, 5], -1),         # median exactly 3 -> ambiguous
    ([1, 2, 4, 5], -1),         # median 3.0 from a split panel -> ambiguous
    ([2, 3, 4], -1),            # odd panel, median 3
    ([3, 4, 5], 1),             # median 4
    ([1, 2, 3], 0),             # median 2
])
def test_score3_rule(scores, expected):
    assert label_from_scores(np.array(scores)) == expected


def test_ambiguous_is_a_third_class_not_a_silent_benign():
    assert label_from_scores(np.array([3, 3, 3])) not in (0, 1)
    assert label_from_scores(np.array([3, 3, 3])) == S.AMBIGUOUS


# --------------------------------------------------------------------------- #
# patient leakage
# --------------------------------------------------------------------------- #
def toy_manifest(n_patients=40, seed=0) -> "object":
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        for k in range(rng.integers(1, 4)):
            rows.append({"file": f"P{p:03d}_n{k}.npz", "patient_id": f"P{p:03d}",
                         "label": int(rng.choice([0, 1, -1])), "n_readers": 4})
    return pd.DataFrame(rows)


def test_every_patient_lands_in_exactly_one_fold():
    df = toy_manifest()
    assign = S.stratified_patient_kfold(df, n_splits=5, seed=42)
    assert set(assign) == set(df["patient_id"])
    assert all(0 <= v < 5 for v in assign.values())


def test_the_subject_column_may_carry_either_name():
    """`splits` was generalised from `patient_id` to `subject_id`. Both must
    resolve, because the manifest.csv every cohort has already cached on disk
    carries the historical name — and a frozen split that stops loading is five
    committed fold assignments invalidated at once."""
    df = toy_manifest()
    renamed = df.rename(columns={"patient_id": "subject_id"})

    assert S.subject_column(df) == "patient_id"
    assert S.subject_column(renamed) == "subject_id"
    # The fingerprint is what a frozen split is checked against, so it above all
    # must not depend on which name the column happens to have.
    assert S.manifest_fingerprint(df) == S.manifest_fingerprint(renamed)
    assert S.stratified_patient_kfold(df, 5, 42) == \
           S.stratified_patient_kfold(renamed, 5, 42)

    with pytest.raises(KeyError, match="no subject column"):
        S.subject_column(df.drop(columns=["patient_id"]))


def test_the_labels_are_named_for_the_arithmetic_not_for_one_cohort():
    """NEGATIVE / POSITIVE, not BENIGN / MALIGNANT: the fold arithmetic is the
    same whatever the positive class denotes, and `test_core_purity` now holds
    the whole package to that."""
    assert (S.NEGATIVE, S.POSITIVE, S.AMBIGUOUS) == (0, 1, -1)
    assert S.BINARY_LABELS == (S.NEGATIVE, S.POSITIVE)
    assert not hasattr(S, "MALIGNANT")


def test_no_patient_appears_in_two_splits_of_any_fold():
    df = toy_manifest()
    assign = S.stratified_patient_kfold(df, n_splits=5, seed=42)
    for k in range(5):
        sp = S.make_fold(df, assign, k, 5)
        S.assert_no_leakage(df, sp)                 # raises on a leak
        pats = {name: {f.split("_n")[0] for f in files} for name, files in sp.items()}
        assert pats["train"] & pats["test"] == set()
        assert pats["train"] & pats["val"] == set()


def test_score3_is_never_a_training_target():
    df = toy_manifest()
    assign = S.stratified_patient_kfold(df, n_splits=5, seed=42)
    amb = set(df.loc[df.label == S.AMBIGUOUS, "file"])
    for k in range(5):
        sp = S.make_fold(df, assign, k, 5)
        assert not (set(sp["train"]) & amb)
        assert not (set(sp["val"]) & amb)
        assert not (set(sp["test"]) & amb)
        assert set(sp["score3"]) <= amb


def test_leakage_check_actually_catches_a_leak():
    """A guard that never fires is not a guard."""
    df = toy_manifest()
    assign = S.stratified_patient_kfold(df, n_splits=5, seed=42)
    sp = S.make_fold(df, assign, 0, 5)
    sp["train"] = sp["train"] + sp["test"][:1]      # smuggle a test patient into train
    with pytest.raises(AssertionError):
        S.assert_no_leakage(df, sp)


# --------------------------------------------------------------------------- #
# the frozen split (file-level: no dataset required)
# --------------------------------------------------------------------------- #
def frozen_path():
    return ROOT / load("notebooks/base").folds


def test_frozen_split_exists_and_is_well_formed():
    blob = json.loads(frozen_path().read_text(encoding="utf-8"))
    cfg = load("notebooks/base")
    assert blob["seed"] == cfg.seed and blob["n_splits"] == cfg.n_splits
    folds = list(blob["assignment"].values())
    assert len(folds) == blob["n_patients"] > 0
    assert set(folds) == set(range(cfg.n_splits))       # every fold is populated
    assert blob["manifest_fingerprint"].startswith("sha256:")


def test_frozen_split_is_balanced_across_folds():
    blob = json.loads(frozen_path().read_text(encoding="utf-8"))
    sizes = np.bincount(list(blob["assignment"].values()), minlength=blob["n_splits"])
    assert sizes.max() - sizes.min() <= 0.05 * sizes.mean() + 1


# --------------------------------------------------------------------------- #
# one transform, applied to image + mask + agreement together
# --------------------------------------------------------------------------- #
def write_nodule(dirpath, name, *, patient_id="P000", scores=(4, 4, 5, 5),
                 n_readers=4, patch=None):
    """A synthetic .npz in the on-disk format, with a deliberately ASYMMETRIC
    lesion so a rotation or flip that reached one array but not another cannot
    go unnoticed. The image is built FROM the mask, so 'same transform' becomes
    an exact array equality rather than a visual judgement."""
    patch = patch or PATCH
    mask = np.zeros((patch, patch), np.uint8)
    mask[20:34, 24:30] = 1                    # tall, off-centre rectangle
    mask[20:24, 24:26] = 0                    # notch: kills the last symmetry
    count = mask.astype(np.uint8) * n_readers
    count[28:34, 24:30] = 1                   # a low-agreement tail
    count = (count * mask).astype(np.uint8)
    frac = (count.astype(np.float32) / n_readers)

    image = np.stack([mask.astype(np.float32)] * 3, axis=0)   # image == mask, exactly
    scores = np.asarray(scores, dtype=np.int64)
    t = rating_targets(scores)
    np.savez_compressed(
        dirpath / name, image=image, agreement_map=count, agreement_frac=frac,
        binary_mask=mask, malignancy=scores.astype(np.int8),
        opinion_dist=t["opinion_dist"], disagreement=np.float32(t["disagreement"]),
        rating_entropy=np.float32(t["rating_entropy"]), n_readers=np.int8(n_readers),
        label=np.int8(t["label"]), patient_id=patient_id,
        series_uid="1.2.3", nodule_idx=0)
    return dirpath / name


@pytest.fixture
def synthetic_dir(tmp_path):
    write_nodule(tmp_path, "P000_n00.npz")
    return tmp_path


def test_augmentation_moves_image_mask_and_agreement_together(synthetic_dir):
    from backend.domains.adapter import PatchDataset
    ds = PatchDataset(synthetic_dir, ["P000_n00.npz"], train=True, img_size=PATCH)

    seen = set()
    for trial in range(60):
        np.random.seed(trial)
        s = ds[0]
        img, mask = s["image"].numpy(), s["binary_mask"].numpy()
        amap, frac = s["agreement_map"].numpy(), s["agreement_frac"].numpy()

        # the image was constructed equal to the mask; one shared transform keeps it so
        assert np.array_equal(img[0], mask), f"image and mask diverged on trial {trial}"
        assert np.array_equal(img[1], img[2])
        # the agreement surface travelled with them: same support, same normalisation
        assert np.array_equal(amap > 0, mask > 0)
        assert np.allclose(frac, amap / s["n_readers"].item())
        seen.add(mask.tobytes())

    assert len(seen) > 1, "augmentation never actually transformed anything"


def test_every_orientation_keeps_the_maps_aligned(synthetic_dir):
    """Exercise all 16 (rot, flip, flip) combinations explicitly, not by luck."""
    from backend.domains.adapter import PatchDataset
    ds = PatchDataset(synthetic_dir, ["P000_n00.npz"], train=False, img_size=PATCH)
    s = ds[0]
    img, mask, frac = s["image"].numpy(), s["binary_mask"].numpy(), s["agreement_frac"].numpy()

    for k in range(4):
        for f0 in (False, True):
            for f1 in (False, True):
                tf = (k, f0, f1)
                a = PatchDataset._apply_tf(img, tf, (1, 2))
                b = PatchDataset._apply_tf(mask, tf, (0, 1))
                c = PatchDataset._apply_tf(frac, tf, (0, 1))
                assert np.array_equal(a[0], b), f"image/mask mismatch at {tf}"
                assert np.array_equal(c > 0, b > 0), f"agreement/mask mismatch at {tf}"


def test_decentring_shifts_every_map_by_the_same_offset(synthetic_dir):
    """Decentring is applied to four arrays independently; if the offsets ever
    drift apart the maps silently describe the wrong pixels."""
    from backend.domains.adapter import PatchDataset
    plain = PatchDataset(synthetic_dir, ["P000_n00.npz"], train=False, img_size=PATCH)[0]
    shifted = PatchDataset(synthetic_dir, ["P000_n00.npz"], train=False,
                            img_size=PATCH, shift_frac=0.2, shift_seed=7)[0]

    m0, m1 = plain["binary_mask"].numpy(), shifted["binary_mask"].numpy()
    assert m1.sum() == m0.sum(), "decentring clipped the mask"
    off = np.array(np.nonzero(m1)).min(axis=1) - np.array(np.nonzero(m0)).min(axis=1)

    for key in ("agreement_map", "agreement_frac"):
        a0, a1 = plain[key].numpy(), shifted[key].numpy()
        assert (a1 > 0).sum() == (a0 > 0).sum()
        got = np.array(np.nonzero(a1 > 0)).min(axis=1) - np.array(np.nonzero(a0 > 0)).min(axis=1)
        assert np.array_equal(got, off), f"{key} shifted by {got}, mask by {off}"

    i0, i1 = plain["image"].numpy()[0], shifted["image"].numpy()[0]
    got = np.array(np.nonzero(i1)).min(axis=1) - np.array(np.nonzero(i0)).min(axis=1)
    assert np.array_equal(got, off), f"image shifted by {got}, mask by {off}"


def test_decentring_reflects_the_image_but_zero_fills_the_maps():
    """The band decentring vacates must be REAL tissue on the image and EMPTY on the
    maps, and either way must not touch in-mask pixels.

    Zero-filling the image would paint a synthetic black band over ~20% of every
    frame (max 36.5% on a diagonal offset), which encodes which way the lesion
    moved. Reflecting the MAPS instead would be worse: a mirrored lesion is a
    lesion the annotators never drew.
    """
    from backend.domains.adapter import PatchDataset

    rng = np.random.default_rng(0)
    img = (rng.random((3, 64, 64)) * 0.8 + 0.1).astype(np.float32)   # no exact zeros
    mask = np.zeros((64, 64), np.float32)
    mask[26:38, 26:38] = 1.0

    for dy, dx in [(13, 0), (0, -13), (9, -11), (-13, 13)]:
        im_reflect = PatchDataset._shift(img, dy, dx, fill="reflect")
        im_zero = PatchDataset._shift(img, dy, dx)
        mk = PatchDataset._shift(mask, dy, dx)

        assert (im_zero == 0).mean() > 0.15, "fixture should exercise a real dead band"
        assert (im_reflect == 0).mean() == 0.0, (
            f"reflect fill left {(im_reflect == 0).mean():.1%} dead pixels at ({dy},{dx})")
        assert im_reflect.min() >= img.min() and im_reflect.max() <= img.max(), \
            "reflect must not invent out-of-range intensities"

        # maps stay zero-filled: mask area is conserved, never grown by a mirror image
        assert mk.sum() == mask.sum(), "decentring changed the mask area"

        # the clamp keeps the whole bbox inside, so in-mask pixels are never filled at
        # all.
        sel = mk > 0
        assert np.allclose(im_reflect[:, sel], im_zero[:, sel]), \
            "fill mode changed in-mask pixels; it must only affect the vacated band"

    assert np.array_equal(PatchDataset._shift(img, 0, 0, fill="reflect"), img)
    with pytest.raises(ValueError, match="fill must be"):
        PatchDataset._shift(img, 3, 3, fill="wrap")


def test_resize_keeps_the_maps_on_the_image_grid(synthetic_dir):
    """At 224 (the ImageNet stem) the maps must be resized with the image, or
    they no longer share its coordinate frame."""
    from backend.domains.adapter import PatchDataset
    s = PatchDataset(synthetic_dir, ["P000_n00.npz"], train=False, img_size=224)[0]
    assert s["image"].shape == (3, 224, 224)
    for key in ("agreement_map", "agreement_frac", "binary_mask"):
        assert s[key].shape == (224, 224), key
    assert set(np.unique(s["binary_mask"].numpy())) <= {0.0, 1.0}   # nearest, not smeared
    assert float(s["agreement_frac"].max()) <= 1.0
