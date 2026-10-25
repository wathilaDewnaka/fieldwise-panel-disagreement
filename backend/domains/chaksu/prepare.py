"""Chákṣu preprocessing — the retinal fundus / glaucoma domain.

    python -m backend.domains.chaksu.prepare                  # the whole cohort
    python -m backend.domains.chaksu.prepare --max-items 8    # a quick smoke run
    python -m backend.domains.chaksu.prepare --splits Test

Writes ONE .npz per fundus image, in the schema `backend/domains/lidc/prepare.py`
writes, so `field_disagreement.splits`, the torch adapter and the notebook read
this cohort with no changes:

  image          (3, P, P)  float32   the OD-centred crop, /255 -> [0, 1]
  agreement_map  (P, P)     uint8     A(x,y): experts who marked each pixel, 0..R
  agreement_frac (P, P)     float32   A(x,y) / n_readers, in [0, 1]
  binary_mask    (P, P)     uint8     1 where ANY expert marked (Omega)
  votes          (R,)       int8      per expert: 1 glaucoma suspect, 0 normal
  opinion_dist   (2,)       float32   histogram over (normal, glaucoma)
  disagreement   float32              s.d. of the votes — BOUNDED BY 0.5
  mean_dice      float32              mean pairwise Dice of the panel's masks
  rating_entropy float32              Shannon entropy of opinion_dist (nats)
  n_readers      int8                 experts on this image's panel (4 or 5)
  label          int8                 the panel's majority decision
  patient_id     str                  the grouping unit (see `subject_of`)
  series_uid     str                  "<device>_<stem>", the image id
  nodule_idx     int                  0 — one optic disc per image
  + image_id, experts, device, split, structure, crop_box, native_shape

Why this cohort is worth the trouble
------------------------------------
Five experts read all 1345 images and record a decision and a set of
measurements for each, so every image carries a genuine per-reader opinion
distribution (LIDC reads 34.6% of its nodules with only three readers).

Three things that would fail silently, and what is done about them
------------------------------------------------------------------
* `disagreement` is the s.d. of five BINARY votes, so it is bounded by 0.5 and
  takes only the values {0, 0.400, 0.490}. LIDC's `contested_d: 1.0` selects
  the EMPTY SET here (see `backend/domains/datasets.py`). A quantile does not
  rescue it either, because a three-valued variable
  has no useful quantiles
  (q=0.70 and q=0.80 both give 15.5%, q=0.90 gives 0.2%). Set `contested_d`
  EXPLICITLY: 0.35 selects the 41.5% non-unanimous items, 0.45 the 14.6% genuine
  2-3 splits. `run` prints this.
* The archive is a trap for `unzip`. The masks are uncompressed TIFF, so the
  11 GB download expands to 230 GB. Nothing here extracts: both split zips are
  streamed member by member and only the resized .npz samples are written.
* Seven byte-identical duplicate images ship in the Bosch subset. The official
  train/test boundary is clean, but this repo pools both for 5-fold CV, which
  would scatter a duplicate pair across folds. `dedupe` drops the later one.

Two more archive quirks are handled where they are read: the decision CSVs carry
four spellings of the header (`Glaucoam Decision`, a trailing space, ...) and two
of the value (`GLAUCOMA  SUSUPECT`), so both are matched loosely; and the Bosch
CSVs are duplicated supersets holding all 145 Bosch images in BOTH splits, so
decisions are joined via the image FILES — building an item list from the CSVs
would put test labels in the train table.

Geometry: the crop is not optional
----------------------------------
On the full frame the optic cup covers 0.73% of the image. Every sample is
therefore cropped about the optic disc (the standard fundus pipeline), located
from the photograph alone (`crop_mode: auto`). The crop
RE-CENTRES the target: uncropped, this cohort is naturally decentred (OD centroid
a mean 0.185 of the frame from the centre), and cropping recreates LIDC's centred
geometry, so runs on this cohort use the same decentring (`shift_frac`) as LIDC.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from backend.config import load, paths
from backend.domains.chaksu.masks import (MASK_THRESHOLD, agreement_surface,
                                          FOV_FRACTION, auto_window, crop,
                                          disc_window, mean_pairwise_dice,
                                          to_grid, vote_targets)

IMAGES = "1.0_Original_Fundus_Images"
MASKS = "3.0_Doctors_Annotations_Binary_OD_OC"      # the binary OD/OC masks
DECISIONS = "6.0_Glaucoma_Decision"
PARTS = {"cup": "Cup", "disc": "Disc"}
EXPERTS = tuple(f"Expert {i}" for i in range(1, 6))
PATIENT = re.compile(r"^(P\d+)_")        # Bosch names two eyes of one patient


class Item(NamedTuple):
    """One fundus image with its panel — the unit that becomes a sample."""

    image_id: str
    split: str                  # "Train" / "Test", the release's own split
    device: str                 # Remidio | Forus | Bosch
    image: str                  # member name inside <split>.zip
    cup: dict[str, str]         # expert -> member, ordered by expert
    disc: dict[str, str]
    votes: dict[str, int]       # expert -> 1 glaucoma suspect / 0 normal
    subject: str

    @property
    def n_readers(self) -> int:
        return len(self.votes)


def subject_of(device: str, stem: str) -> str:
    """The unit a split must not cross.

    Only the Bosch subset names patients, as a `P##_` prefix on images of the
    same person's two eyes; the other 1285 images carry no patient id at all, so
    they are their own subject. The `P##` namespaces of Train and Bosch-Test are
    NOT known to be different people, so the split is deliberately left out of
    the id: merging two patients only costs a little fold balance, whereas
    separating one across folds is leakage."""
    match = PATIENT.match(stem)
    return f"{device}_{match.group(1)}" if match else f"{device}_{stem}"


def read_decisions(zf: zipfile.ZipFile, split: str, expert: str,
                   device: str) -> dict[str, int]:
    """One expert's glaucoma calls for one device, keyed by image stem.

    The CSVs need loose matching on both sides. Headers appear as `Glaucoma
    Decision`, `Glaucoam Decision`, `Glaucoma Decision ` and `Glaucoma decision`;
    values as `GLAUCOMA SUSPECT` and `GLAUCOMA  SUSUPECT`. The image column reads
    `<stem>.<ext>-<stem>-1.<ext>`, whose first dash-separated field is the file
    the row is about.

    Note what is NOT done here: the Bosch CSVs list all 145 Bosch images in BOTH
    splits, so this returns rows for images that live in the other zip. They are
    dropped by the caller, which iterates over the image FILES."""
    member = f"{split}/{DECISIONS}/{expert}/{device}.csv"
    try:
        text = zf.read(member).decode("utf-8-sig", "replace")
    except KeyError:
        return {}

    reader = csv.DictReader(io.StringIO(text))
    fields = [f for f in (reader.fieldnames or []) if f]
    image_col = next((f for f in fields if "image" in f.strip().lower()), None)
    dec_col = next((f for f in fields if "decision" in f.strip().lower()), None)
    if image_col is None or dec_col is None:
        raise ValueError(f"{member} has no image/decision column, only {fields}")

    out: dict[str, int] = {}
    for row in reader:
        value = (row.get(dec_col) or "").strip().upper()
        name = (row.get(image_col) or "").split("-")[0].strip()
        if not value or not name:
            continue
        out[Path(name).stem] = int(value.startswith("GLAUCOMA"))
    return out


class ChaksuSource:
    """Every image in the download, with its five masks and five decisions.

    Indexes `Train.zip` and `Test.zip` in place — the archive is never extracted
    (uncompressed TIFF masks: 11 GB of zip becomes 230 GB of disk). The zips stay
    open for `read` to stream from.

    Everything joins on the image STEM, never on the filename: a Remidio image is
    `IMG_2617.JPG` and its masks are `IMG_2617.tif`, and Test/Forus ships masks
    as a mixture of .png and .tif in the same folder."""

    def __init__(self, root, cfg):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Chaksu raw data not found at {self.root}. "
                f"Fix `chaksu_raw` in configs/paths.yaml — it is the folder "
                f"holding Train.zip and Test.zip.")

        self.zips: dict[str, zipfile.ZipFile] = {}
        self.items: list[Item] = []
        self.n_incomplete = 0
        self._images: set[tuple[str, str]] = set()      # (device, stem), all splits
        self._decisions: set[tuple[str, str]] = set()
        for split in cfg.splits:
            self._index(split, cfg)

        # Rows naming an image that exists in NEITHER split — three of them ship,
        # in Train/Forus. Judged across both splits on purpose: the Bosch CSVs
        # are duplicated supersets, so half their rows are "orphans" of their own
        # split and only look wrong.
        self.n_orphan_rows = len(self._decisions - self._images)
        if not self.items:
            raise FileNotFoundError(
                f"no annotated images found under {self.root} — expected "
                f"{list(cfg.splits)} zips holding {IMAGES}/ and {MASKS}/")
        self.items.sort(key=lambda it: it.image_id)

        clashes = [k for k, n in Counter(it.image_id for it in self.items).items()
                   if n > 1]
        if clashes:
            raise ValueError(f"image ids are not unique across splits: {clashes[:5]}")

    def _index(self, split: str, cfg) -> None:
        path = self.root / f"{split}.zip"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing from the Chaksu download")
        zf = self.zips[split] = zipfile.ZipFile(path)

        images: dict[tuple[str, str], str] = {}
        masks: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
        for name in zf.namelist():
            # The archive was zipped on a Mac: every folder carries a resource
            # fork under __MACOSX/ and a .DS_Store, neither of which is data.
            if name.startswith("__MACOSX") or name.endswith(("/", ".DS_Store")):
                continue
            part = name.split("/")
            if len(part) == 4 and part[1] == IMAGES:
                images[(part[2], PurePosixPath(name).stem)] = name
            elif len(part) == 6 and part[1] == MASKS:
                masks[(part[2], part[3], part[4])][PurePosixPath(name).stem] = name

        self._images |= set(images)
        devices = sorted({device for device, _ in images})
        for device in devices:
            stems = {s for d, s in images if d == device}
            votes = {e: read_decisions(zf, split, e, device) for e in EXPERTS}
            self._decisions |= {(device, s) for v in votes.values() for s in v}

            for stem in sorted(stems):
                # The panel is the experts who supplied BOTH masks AND a call, so
                # the agreement surface and the opinion distribution come from the
                # same people. Three Train/Forus images lose one expert this way.
                panel = {e: (masks[(e, device, PARTS["cup"])].get(stem),
                             masks[(e, device, PARTS["disc"])].get(stem),
                             votes[e].get(stem))
                         for e in EXPERTS}
                panel = {e: v for e, v in panel.items() if None not in v}
                if len(panel) < cfg.min_experts:
                    self.n_incomplete += 1
                    continue
                self.items.append(Item(
                    image_id=f"{device}_{stem}",
                    split=split,
                    device=device,
                    image=images[(device, stem)],
                    cup={e: v[0] for e, v in panel.items()},
                    disc={e: v[1] for e, v in panel.items()},
                    votes={e: v[2] for e, v in panel.items()},
                    subject=subject_of(device, stem),
                ))

    def _read(self, split: str, member: str, mode: str) -> np.ndarray:
        with Image.open(io.BytesIO(self.zips[split].read(member))) as im:
            return np.asarray(im.convert(mode), dtype=np.uint8)

    def read(self, item: Item) -> tuple[np.ndarray, list[np.ndarray],
                                        list[np.ndarray]]:
        """(RGB image, [cup masks], [disc masks]) at native size, 0/1 masks."""
        image = self._read(item.split, item.image, "RGB")
        out = []
        for members in (item.cup, item.disc):
            masks = []
            for expert, member in members.items():
                mask = (self._read(item.split, member, "L") > 127).astype(np.uint8)
                if mask.shape != image.shape[:2]:
                    raise ValueError(f"{expert}'s mask is {mask.shape} but the "
                                     f"image is {image.shape[:2]}")
                masks.append(mask)
            out.append(masks)
        return image, out[0], out[1]


def process_item(source: ChaksuSource, item: Item, cfg, out_dir: Path,
                 rows: list[dict], seen: dict[str, str]) -> str | None:
    """Prepare one image. Appends a row if kept; else names why it was not."""
    image, cup, disc = source.read(item)

    if cfg.dedupe:
        # Seven byte-identical Bosch images ship twice. Hashing the DECODED
        # pixels costs nothing extra (they are in hand) and also catches a
        # re-encoded copy that a hash of the file bytes would miss.
        digest = hashlib.md5(image.tobytes()).hexdigest()
        if digest in seen:
            return "duplicate"
        seen[digest] = item.image_id

    # The opinion targets are the experts' decisions, not their masks.
    targets = vote_targets(item.votes)

    # The agreement surface is the CUP by default (the experts agree on the disc
    # at Dice 0.95, on the cup at 0.82). It is used only by the translation clamp.
    panel = {"cup": cup, "disc": disc}[cfg.agreement_structure]
    mean_dice = mean_pairwise_dice(panel, MASK_THRESHOLD)   # native resolution, as drawn

    # `auto` (the default) locates the disc from the photograph;
    # `panel` is the historical window from the experts' union mask, whose SCALE
    # encodes their disc-boundary disagreement, a label leak.
    if getattr(cfg, "crop_mode", "auto") == "panel":
        box = disc_window(disc, cfg.crop_radii)
    else:
        box = auto_window(image, getattr(cfg, "fov_fraction", FOV_FRACTION))
    x = to_grid(crop(image, box), cfg.patch_size, "bilinear", square=False)
    # Each expert's mask is resized INDIVIDUALLY with nearest and counted only
    # afterwards; resizing the count map would interpolate between expert counts
    # and invent agreement levels nobody voted for.
    small = [to_grid(crop(m, box), cfg.patch_size, "nearest", square=False)
             for m in panel]
    count, frac, binary = agreement_surface(small, MASK_THRESHOLD)

    mask_frac = float(binary.mean())
    if mask_frac <= cfg.min_mask_frac:      # strict: 0.0 drops an empty union,
        return "empty_mask"

    fname = f"{item.image_id}.npz"
    np.savez_compressed(
        out_dir / fname,
        image=np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32) / 255.0,
        agreement_map=count,
        agreement_frac=frac.astype(np.float32),
        binary_mask=binary,
        votes=targets["votes"],
        opinion_dist=targets["opinion_dist"].astype(np.float32),
        disagreement=np.float32(targets["disagreement"]),
        mean_dice=np.float32(mean_dice),
        rating_entropy=np.float32(targets["rating_entropy"]),
        n_readers=np.int8(item.n_readers),
        label=np.int8(targets["label"]),
        patient_id=item.subject,
        series_uid=item.image_id,
        nodule_idx=0,
        image_id=item.image_id,
        experts="|".join(item.votes),
        device=item.device,
        split=item.split,
        structure=cfg.agreement_structure,
        crop_box=np.array(box, dtype=np.int32),
        crop_mode=np.asarray(getattr(cfg, "crop_mode", "auto")),
        native_shape=np.array(image.shape[:2], dtype=np.int32),
    )
    rows.append({
        "file": fname,
        "patient_id": item.subject,
        "series_uid": item.image_id,
        "image_id": item.image_id,
        "nodule_idx": 0,
        "split": item.split,
        "device": item.device,
        "n_readers": item.n_readers,
        "experts": "|".join(item.votes),
        "votes": "|".join(map(str, targets["votes"].tolist())),
        "opinion_dist": "|".join(f"{p:.3f}" for p in targets["opinion_dist"]),
        "disagreement": round(targets["disagreement"], 4),
        "mean_dice": round(mean_dice, 4),
        "label": targets["label"],
        "mask_frac": round(mask_frac, 4),
        "unanimous_frac": round(float((count == item.n_readers).mean()), 4),
        "crop_side": int(box[2]),
        "native_h": int(image.shape[0]),
        "native_w": int(image.shape[1]),
    })
    return None


def split_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per split and device: what came out, and whether the targets vary."""
    if df.empty:
        return df
    g = df.groupby(["split", "device"])
    return pd.DataFrame({
        "items": g.size(),
        "subjects": g.patient_id.nunique(),
        "readers": g.n_readers.apply(lambda s: f"{s.min()}-{s.max()}"),
        "glaucoma": g.label.apply(lambda s: int((s == 1).sum())),
        "normal": g.label.apply(lambda s: int((s == 0).sum())),
        "split_panels": g.disagreement.apply(lambda s: int((s > 0).sum())),
        "median_dice": g.mean_dice.median().round(3),
        "median_mask": g.mask_frac.median().round(3),
    }).reset_index()


def run(source: ChaksuSource, out_dir: Path, cfg) -> pd.DataFrame:
    """Process every eligible item into `out_dir`. Returns the index."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = source.items[:cfg.max_items] if cfg.max_items else source.items
    rows: list[dict] = []
    seen: dict[str, str] = {}
    dropped: Counter[str] = Counter()
    for item in tqdm(items, desc="images"):
        try:
            reason = process_item(source, item, cfg, out_dir, rows, seen)
        except Exception as e:                      # a bad member, not a bad run
            tqdm.write(f"[skip] {item.image_id}: {e}")
            reason = "error"
        if reason:
            dropped[reason] += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "index.csv", index=False)

    (out_dir / "preprocess_provenance.json").write_text(json.dumps({
        "created": date.today().isoformat(),
        "dataset": "chaksu",
        "n_items": len(df),
        "n_subjects": int(df.patient_id.nunique()) if len(df) else 0,
        "n_images_available": len(source.items),
        "n_dropped_incomplete_panel": source.n_incomplete,
        "n_decisions_without_image": source.n_orphan_rows,
        "n_dropped": dict(dropped),
        "per_split": (split_report(df).to_dict(orient="records") if len(df)
                      else []),
        "config": vars(cfg),
    }, indent=2), encoding="utf-8")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preprocess the Chaksu fundus cohort into training samples.")
    ap.add_argument("--max-items", type=int, default=None,
                    help="truncate for a smoke run (overrides the config)")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="release splits to read (default: Train and Test)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: `chaksu_processed` in paths.yaml)")
    args = ap.parse_args()

    cfg = load("preprocessing/preprocess_chaksu")
    if args.max_items is not None:
        cfg.max_items = args.max_items
    if args.splits:
        cfg.splits = args.splits

    P = paths()
    out_dir = args.out or getattr(P, "chaksu_processed", None)
    raw = getattr(P, "chaksu_raw", None)
    if out_dir is None or raw is None:
        raise SystemExit("configs/paths.yaml needs `chaksu_raw` and "
                         "`chaksu_processed` (or pass --out).")

    df = run(ChaksuSource(raw, cfg), Path(out_dir), cfg)

    n_subjects = df.patient_id.nunique() if len(df) else 0
    print(f"\nDone. {len(df)} images from {n_subjects} subjects saved to {out_dir}")
    if not len(df):
        return

    print()
    print(split_report(df).to_string(index=False))
    print(f"\nlabel: {int((df.label == 1).sum())} glaucoma / "
          f"{int((df.label == 0).sum())} normal "
          f"({df.label.mean():.1%} positive) — at this prevalence a 5-fold CV "
          f"gives ~{int(0.2 * df.label.sum())} positives per test fold, so "
          f"expect wider CIs than on LIDC.")
    print(f"opinion: {(df.disagreement > 0).mean():.1%} of panels are "
          f"non-unanimous.")
    print(f"\ndisagreement is the s.d. of {df.n_readers.max()} BINARY votes: "
          f"bounded by 0.5 and taking only "
          f"{sorted(df.disagreement.unique().round(3).tolist())}. "
          f"LIDC's contested_d = 1.0 selects the EMPTY SET here, and no "
          f"quantile works on a three-valued scale — set contested_d "
          f"explicitly (0.35 = the non-unanimous items, 0.45 = the "
          f"genuine 2-3 splits).")
    print(f"\ngeometry: every sample is cropped to {cfg.crop_radii} disc radii "
          f"about the optic disc, which re-centres the target. Turn `shift_frac` "
          f"back on for this cohort and re-calibrate it here — LIDC's 0.20 is "
          f"for a 64 px patch.")


if __name__ == "__main__":
    main()
