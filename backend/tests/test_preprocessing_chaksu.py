"""Chákṣu preprocessing contract tests.

Synthetic arrays throughout, plus an end-to-end pass over a fake download — two
zips built in a tmpdir with the archive's real layout — so it passes with no data,
like the LIDC tests.

The invariants guarded here are the ones that fail QUIETLY on this cohort: a
`disagreement` bounded by 0.5 that base.yaml's 1.0 would turn into an empty
subgroup, decision CSVs whose header and values are misspelled four ways, Bosch
CSVs that list the test images in the train split's file, and seven duplicate
images that only bite once train and test are pooled for CV.
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from backend.config import load
from backend.domains.chaksu.masks import (agreement_surface, crop, disc_window,
                                          vote_targets)
from backend.domains.chaksu.prepare import (ChaksuSource, read_decisions, run,
                                            split_report, subject_of)

H, W = 40, 60                    # a non-square native frame, like the real ones
DISC = (10, 22, 30, 42)          # top, bottom, left, right — centre (16, 36), r 6


def box(top, bottom, left, right, h=H, w=W):
    m = np.zeros((h, w), dtype=np.uint8)
    m[top:bottom, left:right] = 1
    return m


# --------------------------------------------------------------------------- #
# the panel's opinion — decisions, not masks
# --------------------------------------------------------------------------- #
def test_disagreement_is_the_vote_sd_and_cannot_exceed_a_half():
    """The trap. LIDC's `disagreement` is an unbounded rating s.d.; five binary
    votes give a number in [0, 0.5] that takes only three values, so the 1.0 in
    base.yaml selects the empty set and a quantile is brittle. Any threshold on
    it has to be set explicitly for this cohort."""
    seen = {round(vote_targets([0] * (5 - k) + [1] * k)["disagreement"], 3)
            for k in range(6)}
    assert max(seen) <= 0.5
    assert sorted(seen) == [0.0, 0.4, 0.49]


def test_opinion_dist_is_over_the_decision_and_carries_real_spread():
    """Not constant: it is what the
    five experts actually said, and 41.5% of the cohort is non-unanimous."""
    t = vote_targets({"Expert 1": 0, "Expert 2": 1, "Expert 3": 1,
                      "Expert 4": 0, "Expert 5": 0})
    assert t["opinion_dist"].shape == (2,)
    assert t["opinion_dist"] == pytest.approx([0.6, 0.4])
    assert t["votes"].tolist() == [0, 1, 1, 0, 0]
    assert t["rating_entropy"] > 0

    unanimous = vote_targets([0] * 5)
    assert unanimous["opinion_dist"].tolist() == [1.0, 0.0]
    assert unanimous["rating_entropy"] == pytest.approx(0.0)
    assert unanimous["disagreement"] == pytest.approx(0.0)


def test_the_label_is_the_majority_and_a_tie_reads_as_normal():
    assert vote_targets([1, 1, 1, 0, 0])["label"] == 1
    assert vote_targets([1, 1, 0, 0, 0])["label"] == 0
    assert vote_targets([1, 1, 0, 0])["label"] == 0      # the 4-expert panels
    assert vote_targets([1, 1, 1, 0])["label"] == 1


def test_an_empty_or_non_binary_panel_is_an_error():
    with pytest.raises(ValueError, match="no expert decisions"):
        vote_targets([])
    with pytest.raises(ValueError, match="0/1"):
        vote_targets([0, 1, 3])


# --------------------------------------------------------------------------- #
# the optic-disc crop
# --------------------------------------------------------------------------- #
def test_the_window_is_square_and_centred_on_the_disc():
    top, left, side = disc_window([box(*DISC)], radii=3.0)
    assert side == 20                          # 3 x sqrt(144/pi), the 12x12 disc
    # the disc spans rows 10-21 and columns 30-41, so its centroid is (15.5,
    # 35.5); the box is integer-aligned, so it lands within half a pixel of that
    assert abs(top + side / 2 - 15.5) <= 0.5
    assert abs(left + side / 2 - 35.5) <= 0.5

    cropped = crop(box(*DISC), (top, left, side))
    assert cropped.shape == (side, side)
    assert cropped.sum() == box(*DISC).sum()             # the disc is all inside


def test_a_stray_speck_does_not_blow_the_window_up():
    """Real defect, not a hypothetical: three of the five disc masks on Bosch
    Image106 carry a speck hundreds of pixels away. A bbox-derived radius tripled
    that image's window and shrank the cup to 1% of the crop; the area-derived
    one barely moves."""
    speckled = box(*DISC)
    speckled[38:40, 56:58] = 1

    clean = disc_window([box(*DISC)])
    _, _, side = disc_window([speckled])
    assert side <= clean[2] * 1.1


def test_the_window_takes_the_union_of_the_panel_not_one_expert():
    small, large = box(14, 18, 34, 38), box(*DISC)
    assert disc_window([small, large]) == disc_window([large])


def test_the_window_is_clamped_inside_the_frame_so_the_crop_is_never_padded():
    """A padded border is a constant a network can latch onto, and it would
    differ between devices. Clamping trades perfect centring for real pixels."""
    corner = box(0, 12, 0, 12)
    top, left, side = disc_window([corner], radii=3.0)
    assert (top, left) == (0, 0)

    huge = box(0, H, 0, W)                               # bigger than the frame
    _, _, side = disc_window([huge], radii=3.0)
    assert side == min(H, W)


def test_an_empty_disc_union_says_so_instead_of_cropping_the_corner():
    with pytest.raises(ValueError, match="empty"):
        disc_window([np.zeros((H, W), np.uint8)])
    with pytest.raises(ValueError, match="no disc masks"):
        disc_window([])


def test_the_agreement_surface_grades_the_panel_inside_the_crop():
    window = disc_window([box(*DISC)])
    masks = [crop(box(12, 20, 32, 40 + k), window) for k in (0, -2, -4)]
    count, frac, binary = agreement_surface(masks)

    assert count.dtype == np.uint8 and count.max() == 3
    assert sorted(np.unique(count).tolist()) == [0, 1, 2, 3]
    assert frac.max() == pytest.approx(1.0)
    assert (binary == (count >= 1)).all()


# --------------------------------------------------------------------------- #
# subjects
# --------------------------------------------------------------------------- #
def test_only_bosch_names_a_patient_and_two_eyes_of_one_stay_together():
    assert subject_of("Bosch", "P7_Image2") == subject_of("Bosch", "P7_Image1")
    assert subject_of("Bosch", "P7_Image2") == "Bosch_P7"
    # the other 1285 images carry no patient id at all: each is its own subject
    assert subject_of("Remidio", "IMG_2617") == "Remidio_IMG_2617"
    assert subject_of("Forus", "13") == "Forus_13"


# --------------------------------------------------------------------------- #
# a fake download: two zips laid out like the archive
# --------------------------------------------------------------------------- #
EXPERTS = [f"Expert {i}" for i in range(1, 6)]

# The header and value spellings that actually ship, one per expert.
HEADERS = ["Glaucoma Decision", "Glaucoam Decision", "Glaucoma Decision ",
           "Glaucoma decision", "Glaucoma Decision"]
GLAUCOMA = ["GLAUCOMA SUSPECT", "GLAUCOMA  SUSUPECT", "GLAUCOMA SUSPECT",
            "GLAUCOMA SUSPECT", "GLAUCOMA SUSPECT"]

# device -> split -> {stem: (image seed, votes)}. Bosch `Image9` is a byte-copy
# of `Image8` (seed shared), and the two P7 images are one patient's two eyes.
COHORT = {
    "Forus": {
        "Train": {"1": (1, [0, 0, 0, 0, 0]), "2": (2, [1, 1, 0, 0, 1])},
        "Test": {"3": (3, [0, 1, 1, 0, 0])},
    },
    "Bosch": {
        "Train": {"P7_Image1": (4, [1, 1, 1, 1, 1]),
                  "P7_Image2": (5, [0, 0, 0, 0, 1]),
                  "Image8": (6, [0, 0, 0, 0, 0]),
                  "Image9": (6, [0, 0, 0, 0, 0])},      # the duplicate pair
        "Test": {"Image20": (7, [1, 1, 0, 0, 0])},
    },
}
# One expert's Train/Forus row names an image that is in no split — the release's
# three orphan rows — INSTEAD of the image it should name, so "2" has a
# four-expert panel there.
ORPHAN = ("Forus", "Train", "Expert 4", "2", "22GLSUSP")


def _png(arr, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def _masks(expert_idx: int):
    """This expert's (cup, disc). The disc is shared — the crop must not depend
    on who drew it — and the cup grows per expert, which makes a graded band."""
    k = expert_idx
    return (box(14 - k, 20 + k, 32 - k, 40 + k), box(*DISC))


def _decision_csv(rows, header: str, positive: str) -> str:
    out = [f"Images,Disc Area,Cup Area,ACDR,VCDR,{header}"]
    for stem, vote in rows:
        # the archive's own format: "<stem>.<ext>-<stem>-1.<ext>"
        out.append(f"{stem}.png-{stem}-1.png,900,300,0.33,0.59,"
                   f"{positive if vote else 'NORMAL'}")
    return "\n".join(out) + "\n"


@pytest.fixture()
def download(tmp_path):
    from backend.domains.chaksu.prepare import DECISIONS, IMAGES, MASKS

    for split in ("Train", "Test"):
        with zipfile.ZipFile(tmp_path / f"{split}.zip", "w") as z:
            z.writestr(f"{split}/{IMAGES}/.DS_Store", "junk")
            z.writestr(f"__MACOSX/{split}/._{IMAGES}", "junk")

            for device, per_split in COHORT.items():
                for stem, (seed, votes) in per_split[split].items():
                    rng = np.random.default_rng(seed)
                    z.writestr(f"{split}/{IMAGES}/{device}/{stem}.JPG",
                               _png(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)))
                    for i, expert in enumerate(EXPERTS):
                        cup, disc = _masks(i)
                        for part, mask in (("Cup", cup), ("Disc", disc)):
                            # Test/Forus ships PNG masks beside TIFF ones, in the
                            # same folder: they must resolve by STEM, not by
                            # extension.
                            ext, fmt = (("png", "PNG") if split == "Test"
                                        and device == "Forus" else ("tif", "TIFF"))
                            z.writestr(
                                f"{split}/{MASKS}/{expert}/{device}/{part}/{stem}.{ext}",
                                _png(mask * 255, fmt))

                for i, expert in enumerate(EXPERTS):
                    # The Bosch CSVs are duplicated SUPERSETS: both splits' files
                    # list all Bosch images, so a manifest built from the CSVs
                    # would put the test labels in the train table.
                    listed = (per_split["Train"] | per_split["Test"]
                              if device == "Bosch" else per_split[split])
                    rows = [(stem, votes[i]) for stem, (_, votes) in listed.items()]
                    if (device, split, expert) == ORPHAN[:3]:
                        rows = [r for r in rows if r[0] != ORPHAN[3]]
                        rows.append((ORPHAN[4], 1))
                    z.writestr(f"{split}/{DECISIONS}/{expert}/{device}.csv",
                               _decision_csv(rows, HEADERS[i], GLAUCOMA[i]))
    return tmp_path


@pytest.fixture()
def cfg():
    cfg = load("preprocessing/preprocess_chaksu")
    cfg.patch_size = 8
    # these tests check the .npz contract against a known mask geometry; the
    # synthetic images have no optic disc for the photograph-only locator to find
    cfg.crop_mode = "panel"
    return cfg


# --------------------------------------------------------------------------- #
# reading the archive
# --------------------------------------------------------------------------- #
def test_the_decision_csv_is_matched_loosely_on_both_sides(download):
    """Four header spellings and two value spellings ship. An exact match on
    `Glaucoma Decision` silently loses whole expert-device tables."""
    with zipfile.ZipFile(download / "Train.zip") as z:
        for expert in EXPERTS:
            votes = read_decisions(z, "Train", expert, "Forus")
            assert "1" in votes, expert                  # every header spelling
            assert set(votes) <= {"1", "2", ORPHAN[4]}
        # 'GLAUCOMA  SUSUPECT' is the same call as 'GLAUCOMA SUSPECT'
        assert read_decisions(z, "Train", "Expert 2", "Forus")["2"] == 1
        assert read_decisions(z, "Train", "Expert 1", "Forus")["1"] == 0
        # a device this expert never read is absent, not an exception
        assert read_decisions(z, "Train", "Expert 1", "Remidio") == {}


def test_the_panel_is_the_experts_with_both_a_mask_and_a_call(download, cfg):
    src = ChaksuSource(download, cfg)
    items = {it.image_id: it for it in src.items}

    assert items["Forus_1"].n_readers == 5
    assert len(items["Forus_1"].cup) == len(items["Forus_1"].disc) == 5
    # the orphan CSV row means expert 4 has no call on this image
    assert items["Forus_2"].n_readers == 4
    assert "Expert 4" not in items["Forus_2"].votes
    assert "Expert 4" not in items["Forus_2"].cup
    assert src.n_orphan_rows == 1


def test_the_duplicated_bosch_csv_does_not_pull_test_images_into_train(download,
                                                                       cfg):
    """Both splits' Bosch CSVs list all Bosch images. Items come from the image
    FILES, which are correctly disjoint, so the superset rows only ever show up
    as orphans."""
    src = ChaksuSource(download, cfg)
    by_split = {it.image_id: it.split for it in src.items}

    assert by_split["Bosch_Image20"] == "Test"
    assert by_split["Bosch_P7_Image1"] == "Train"
    assert sum(v == "Train" for v in by_split.values()) == 6


def test_an_incomplete_panel_is_dropped_and_counted(download, cfg):
    cfg.min_experts = 5
    src = ChaksuSource(download, cfg)
    assert "Forus_2" not in {it.image_id for it in src.items}
    assert src.n_incomplete == 1


def test_a_missing_download_says_which_config_entry_to_fix(tmp_path, cfg):
    with pytest.raises(FileNotFoundError, match="chaksu_raw"):
        ChaksuSource(tmp_path / "nope", cfg)


def test_a_missing_split_zip_is_named(tmp_path, cfg, download):
    (download / "Test.zip").unlink()
    with pytest.raises(FileNotFoundError, match="Test.zip"):
        ChaksuSource(download, cfg)


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_end_to_end_writes_the_shared_npz_contract(download, cfg, tmp_path):
    out = tmp_path / "out"
    df = run(ChaksuSource(download, cfg), out, cfg)

    d = np.load(out / "Forus_2.npz", allow_pickle=True)
    P = cfg.patch_size
    assert d["image"].shape == (3, P, P) and d["image"].dtype == np.float32
    assert 0.0 <= d["image"].min() and d["image"].max() <= 1.0
    assert d["agreement_map"].shape == (P, P) and d["agreement_map"].dtype == np.uint8
    assert d["agreement_frac"].max() == pytest.approx(1.0)
    assert (d["binary_mask"] == (d["agreement_map"] >= 1)).all()
    assert int(d["n_readers"]) == 4 and int(d["label"]) == 1
    assert d["opinion_dist"].shape == (2,) and d["opinion_dist"].sum() == pytest.approx(1)
    assert 0.0 < float(d["disagreement"]) <= 0.5
    assert 0.0 < float(d["mean_dice"]) <= 1.0
    assert str(d["patient_id"]) == "Forus_2" and str(d["series_uid"]) == "Forus_2"
    assert str(d["device"]) == "Forus" and str(d["split"]) == "Train"
    assert str(d["structure"]) == "cup"
    assert d["crop_box"].tolist() == list(disc_window([box(*DISC)], cfg.crop_radii))
    assert d["native_shape"].tolist() == [H, W]

    assert set(df.file) == {f"{i}.npz" for i in
                            ("Forus_1", "Forus_2", "Forus_3", "Bosch_Image8",
                             "Bosch_Image20", "Bosch_P7_Image1", "Bosch_P7_Image2")}
    assert (out / "index.csv").is_file()


def test_the_duplicate_image_is_dropped_before_folds_are_frozen(download, cfg,
                                                                tmp_path):
    """Seven byte-identical pairs ship. They never cross the official train/test
    line, but this repo pools both splits for 5-fold CV — which would put a copy
    of a training image in the test fold."""
    df = run(ChaksuSource(download, cfg), tmp_path / "out", cfg)
    assert "Bosch_Image8.npz" in set(df.file)
    assert "Bosch_Image9.npz" not in set(df.file)        # the later id goes

    cfg.dedupe = False
    kept = run(ChaksuSource(download, cfg), tmp_path / "dup", cfg)
    assert "Bosch_Image9.npz" in set(kept.file)


def test_the_agreement_structure_switches_between_cup_and_disc(download, cfg,
                                                               tmp_path):
    """The disc is what the crop is centred on either way; the CUP is what the
    panel actually disagrees about."""
    cup = run(ChaksuSource(download, cfg), tmp_path / "cup", cfg)

    cfg.agreement_structure = "disc"
    disc = run(ChaksuSource(download, cfg), tmp_path / "disc", cfg)

    # every expert drew the same disc here, so every marked pixel is unanimous;
    # the cup's band is graded
    assert (disc.unanimous_frac == disc.mask_frac).all()
    assert (cup.unanimous_frac < cup.mask_frac).all()
    assert cup.crop_side.tolist() == disc.crop_side.tolist()


def test_the_provenance_records_what_was_dropped_and_why(download, cfg, tmp_path):
    out = tmp_path / "out"
    run(ChaksuSource(download, cfg), out, cfg)
    prov = json.loads((out / "preprocess_provenance.json").read_text())

    assert prov["n_items"] == 7 and prov["n_images_available"] == 8
    assert prov["n_dropped"] == {"duplicate": 1}
    # the Bosch superset rows are NOT orphans — they name real images in the
    # other split. Only the row naming an image in neither split counts.
    assert prov["n_decisions_without_image"] == 1
    assert prov["config"]["patch_size"] == cfg.patch_size
    assert prov["config"]["crop_radii"] == cfg.crop_radii


def test_the_release_split_is_recorded_and_can_be_restricted(download, cfg,
                                                             tmp_path):
    """Both the release's split and the project's own pooled CV must stay
    possible, so the origin travels with every sample."""
    full = run(ChaksuSource(download, cfg), tmp_path / "all", cfg)
    assert set(full.split) == {"Train", "Test"}
    assert set(split_report(full).device) == {"Forus", "Bosch"}

    cfg.splits = ["Test"]
    only = run(ChaksuSource(download, cfg), tmp_path / "test", cfg)
    assert sorted(only.image_id) == ["Bosch_Image20", "Forus_3"]


def test_end_to_end_manifest_is_readable_by_the_shared_split_code(download, cfg,
                                                                  tmp_path):
    from field_disagreement import splits as S

    out = tmp_path / "out"
    run(ChaksuSource(download, cfg), out, cfg)

    man = S.build_manifest(out, rebuild=True)
    assert set(man.columns) >= {"file", "patient_id", "label", "n_readers"}
    # the patient with two eyes appears once, holding both of their images
    assert sorted(man.patient_id).count("Bosch_P7") == 2


def test_a_mask_that_does_not_line_up_with_its_image_is_skipped_loudly(download,
                                                                       cfg,
                                                                       tmp_path):
    """Silently resizing a mismatched mask would slide the agreement surface off
    the structure it grades."""
    from backend.domains.chaksu.prepare import MASKS

    with zipfile.ZipFile(download / "Train.zip", "a") as z:
        z.writestr(f"Train/{MASKS}/Expert 1/Forus/Cup/1.tif",
                   _png(np.zeros((H + 4, W), np.uint8), "TIFF"))

    df = run(ChaksuSource(download, cfg), tmp_path / "out", cfg)
    assert "Forus_1.npz" not in set(df.file)


def test_the_auto_window_finds_a_bright_disc_without_any_mask():
    """The annotation-free crop: a bright disc on a dark fundus is found from the
    pixels alone, and the window is a fixed share of the field of view."""
    from backend.domains.chaksu.masks import FOV_FRACTION, auto_window, locate_disc
    h, w = 600, 800
    yy, xx = np.mgrid[:h, :w]
    img = np.zeros((h, w, 3), np.uint8)
    fov = (yy - 300) ** 2 + (xx - 400) ** 2 < 280 ** 2
    img[fov] = (120, 60, 30)
    disc = (yy - 280) ** 2 + (xx - 520) ** 2 < 35 ** 2
    img[disc] = (250, 220, 180)
    cy, cx, diam = locate_disc(img)
    assert abs(cy - 280) < 12 and abs(cx - 520) < 12
    assert abs(diam - 560) < 30
    top, left, side = auto_window(img)
    assert side == round(FOV_FRACTION * diam) or abs(side - FOV_FRACTION * diam) <= 1
    assert top <= 280 - 35 and top + side >= 280 + 35
    assert left <= 520 - 35 and left + side >= 520 + 35
