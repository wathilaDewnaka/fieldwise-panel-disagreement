"""LNDb as an EXTERNAL TEST cohort, in the `lidc_report` format.

LNDb (Pedrosa et al., 2019) is a lung-CT database from a Portuguese hospital in
which up to three radiologists, drawn from a pool of five, read each CT
independently - no consensus review - and rated every nodule >= 3 mm on LIDC's
own characteristics. That makes it the external test of the LIDC-trained arms:
same report fields, a different hospital, scanner fleet and reading panel. Nothing
here is trained on.

For every nodule rated by >= MIN_READERS radiologists this writes one .npz with
the keys `PatchDataset` reads for `lidc_report`, built the same way:

  image          (3, 64, 64)  centre axial slice +/- 1, native in-plane grid,
                              lung window [-1000, 400] HU -> [0, 1]
  agreement_map / agreement_frac / binary_mask   from each reader's own mask at
                              the centre slice (`lidc.patches.agreement_surface`)
  malignancy     (R,)         per-reader malignancy 1..5
  opinion_dist / rating_entropy / label          `lidc.patches.rating_targets`
                              (median > 3 malignant, < 3 benign, == 3 ambiguous)
  d_per_attribute (7,)        s.d. of the readers' ratings per field
  d_profile      (7,)         d_per_attribute in units of LIDC's per-field scale
  disagreement                mean(d_profile), the report-level target
  d_single                    malignancy's s.d.

The per-field scales are LIDC's (read from lidc_report's provenance), not refitted
here: an external test applies the training cohort's definition unchanged. The
per-field primary is invariant to them anyway.

Two facts of the release this module depends on, both checked at load:
  * `chars_trainNodules.csv` is row-aligned with `trainNodules.csv`; its `RadID`
    is the GLOBAL radiologist (1..5), while `trainNodules.csv`'s is the reader slot
    on that CT (1..3), which is what the masks and `trainNodules_gt.csv` use.
  * a rating of 0 means "not rated"; such a reader is left out of the panel.

The CTs stay inside data0-5.rar: each needed CT and its readers' masks are pulled
out with 7-Zip into a scratch folder, cropped, and deleted, so the 24 GB release
never sits on disk unpacked.

LNDb is CC BY-NC-ND 4.0: the processed files may not be redistributed.

    python -m backend.domains.lndb.prepare
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import paths
from backend.domains.lidc.patches import (agreement_surface, pad_and_crop,
                                          rating_targets, window_hu)
from backend.domains.report_condition import LIDC_ATTRS

PATCH_SIZE = 64
NEIGHBOURS = 1
HU_MIN, HU_MAX = -1000, 400
# LIDC keeps nodules read by >= 3 radiologists; LNDb assigns at most three per CT
# and most multi-read nodules have two, so two is the smallest panel with a spread.
MIN_READERS = 2
# LNDb column of each LIDC field, in LIDC_ATTRS order
COLUMNS = {"subtlety": "subtlety", "sphericity": "sphericity", "margin": "margin",
           "lobulation": "lobulation", "spiculation": "spiculation",
           "texture": "Text", "malignancy": "malignancy"}


# --------------------------------------------------------------------------- #
# the panels
# --------------------------------------------------------------------------- #
def panels(csv_dir: Path, min_readers: int = MIN_READERS) -> pd.DataFrame:
    """One row per consolidated nodule with >= `min_readers` rated readers:
    the reader slots, their own finding ids, global reader ids and (R, 7) ratings."""
    gt = pd.read_csv(csv_dir / "trainNodules_gt.csv")
    tn = pd.read_csv(csv_dir / "trainNodules.csv")
    ch = pd.read_csv(csv_dir / "chars_trainNodules.csv")
    if not (len(ch) == len(tn) and (ch.LNDbID.values == tn.LNDbID.values).all()
            and (ch.FindingID.values == tn.FindingID.values).all()):
        raise ValueError("chars_trainNodules.csv is no longer row-aligned with "
                         "trainNodules.csv; the reader join below would be wrong")
    tn = tn.join(ch.drop(columns=["LNDbID", "FindingID"]).rename(columns={"RadID": "reader"}))
    by = tn.set_index(["LNDbID", "RadID", "FindingID"])
    rows = []
    for _, r in gt[gt.Nodule == 1].iterrows():
        slots = [int(x) for x in str(r.RadID).split(",")]
        finds = [int(float(x)) for x in str(r.RadFindingID).split(",")]
        keep = []
        for s, f in zip(slots, finds):
            v = by.loc[(r.LNDbID, s, f)]
            sc = np.array([v[COLUMNS[a]] for a in LIDC_ATTRS], float)
            if v.Nodule == 1 and (sc >= 1).all():
                keep.append((s, f, int(v.reader), sc))
        if len(keep) >= min_readers:
            rows.append({"lndb": int(r.LNDbID), "finding": int(r.FindingID),
                         "slots": [k[0] for k in keep], "rad_findings": [k[1] for k in keep],
                         "readers": [k[2] for k in keep],
                         "ratings": np.stack([k[3] for k in keep]).astype(int),
                         "xyz": np.array([r.x, r.y, r.z], float)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# MetaImage, read without SimpleITK
# --------------------------------------------------------------------------- #
_DTYPES = {"MET_SHORT": np.int16, "MET_USHORT": np.uint16, "MET_UCHAR": np.uint8,
           "MET_CHAR": np.int8, "MET_INT": np.int32, "MET_FLOAT": np.float32}


def read_mhd(path: Path):
    """(Z, Y, X) array, spacing (x, y, z), offset (x, y, z), 3x3 direction."""
    head = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            head[k.strip()] = v.strip()
    if head.get("CompressedData", "False") == "True":
        raise NotImplementedError(f"{path}: compressed MetaImage")
    dims = [int(x) for x in head["DimSize"].split()]
    dt = np.dtype(_DTYPES[head["ElementType"]])
    if head.get("BinaryDataByteOrderMSB", "False") == "True":
        dt = dt.newbyteorder(">")
    arr = np.fromfile(Path(path).with_name(head["ElementDataFile"]), dtype=dt)
    arr = arr.reshape(dims[::-1])
    spacing = np.array(head["ElementSpacing"].split(), float)
    offset = np.array(head["Offset"].split(), float)
    # MetaIO lists the direction cosines column by column: transpose to get the
    # matrix whose COLUMNS are the voxel axes (what SimpleITK's GetDirection gives)
    direction = np.array(head.get("TransformMatrix", "1 0 0 0 1 0 0 0 1").split(),
                         float).reshape(3, 3).T
    return arr, spacing, offset, direction


def world_to_voxel(xyz, spacing, offset, direction) -> np.ndarray:
    """World (x, y, z) mm -> voxel (x, y, z) indices, as LNDb's utils.py does."""
    m = direction * spacing[None, :]
    return np.rint(np.linalg.solve(m, np.asarray(xyz, float) - offset)).astype(int)


# --------------------------------------------------------------------------- #
# one nodule
# --------------------------------------------------------------------------- #
def nodule_npz(vol: np.ndarray, masks: dict[int, np.ndarray], row, scale: np.ndarray,
               ijk_hint: np.ndarray) -> dict | None:
    """The lidc_report keys for one nodule, or None when no reader's mask has a
    voxel (then there is no centre slice to crop, exactly LIDC's skip rule)."""
    reader_masks = [masks[s] == f for s, f in zip(row.slots, row.rad_findings)]
    union = np.zeros(vol.shape, bool)
    for m in reader_masks:
        union |= m
    if not union.any():
        return None
    zz, yy, xx = np.nonzero(union)
    ck = (zz.min() + zz.max() + 1) // 2
    ci = (yy.min() + yy.max() + 1) // 2
    cj = (xx.min() + xx.max() + 1) // 2
    Z, H, W = vol.shape
    count, frac, binary = agreement_surface(
        [m[ck] for m in reader_masks], bbox=(slice(0, H), slice(0, W)),
        full_shape=(H, W), centre=(ci, cj), patch_size=PATCH_SIZE)
    if not binary.any():
        return None
    slices = [pad_and_crop(vol[int(np.clip(ck + dz, 0, Z - 1))], ci, cj, PATCH_SIZE)
              for dz in range(-NEIGHBOURS, NEIGHBOURS + 1)]
    image = window_hu(np.stack(slices, axis=0), HU_MIN, HU_MAX)

    ratings = np.asarray(row.ratings, int)                  # (R, 7)
    mal = ratings[:, -1]
    t = rating_targets(mal)
    d_attr = ratings.astype(float).std(axis=0)              # population s.d., as LIDC
    prof = d_attr / scale
    return {
        "image": image.astype(np.float32),
        "agreement_map": count, "agreement_frac": frac.astype(np.float32),
        "binary_mask": binary,
        "malignancy": mal.astype(np.int8),
        "opinion_dist": t["opinion_dist"].astype(np.float32),
        "rating_entropy": np.float32(t["rating_entropy"]),
        "n_readers": np.int8(len(mal)), "label": np.int8(t["label"]),
        "reader_ids": np.asarray(row.readers, np.int8),
        "annotation_ids": np.asarray(row.rad_findings, np.int64),
        "patient_id": f"LNDb-{row.lndb:04d}", "series_uid": f"LNDb-{row.lndb:04d}",
        "nodule_idx": int(row.finding),
        "d_attributes": np.asarray(LIDC_ATTRS), "d_per_attribute": d_attr.astype(np.float32),
        "d_single": np.float32(d_attr[-1]),
        "d_profile": prof.astype(np.float32), "d_profile_attributes": np.asarray(LIDC_ATTRS),
        "disagreement": np.float32(prof.mean()),
        "centre_offset_mm": np.float32(0.0),
    }, {"centre_vox": (int(cj), int(ci), int(ck)), "hint_vox": tuple(int(v) for v in ijk_hint)}


# --------------------------------------------------------------------------- #
# archives
# --------------------------------------------------------------------------- #
def seven_zip() -> str:
    for c in (shutil.which("7z"), shutil.which("7za"),
              r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("7-Zip not found: install it or put 7z on PATH")


def archive_index(raw: Path, sz: str) -> dict[str, tuple[str, str]]:
    """basename -> (archive, member path) over data*.rar and masks.rar."""
    out = {}
    for arc in sorted(raw.glob("*.rar")):
        listing = subprocess.run([sz, "l", "-slt", str(arc)], capture_output=True,
                                 text=True, check=True).stdout
        for line in listing.splitlines():
            if line.startswith("Path = ") and not line.endswith(".rar"):
                member = line[7:]
                out[member.replace("\\", "/").split("/")[-1]] = (str(arc), member)
    return out


def extract(sz: str, archive: str, members: list[str], dest: Path) -> None:
    subprocess.run([sz, "e", archive, *members, f"-o{dest}", "-y"],
                   capture_output=True, check=True)


def lidc_scales() -> np.ndarray:
    prov = json.loads((Path(paths().lidc_report_processed) / "preprocess_provenance.json")
                      .read_text(encoding="utf-8"))
    s = prov["report_condition"]["per_attribute_scale"]
    return np.array([s[a] for a in LIDC_ATTRS], float)


def main(min_readers: int = MIN_READERS, limit: int | None = None) -> Path:
    p = paths()
    raw, out = Path(p.lndb_raw), Path(p.lndb_report_processed)
    csv_dir = raw / "csv"
    out.mkdir(parents=True, exist_ok=True)
    sz = seven_zip()
    idx = archive_index(raw, sz)
    pan = panels(csv_dir, min_readers)
    scale = lidc_scales()
    cts = sorted(pan.lndb.unique())[:limit]
    rows, skipped, far = [], [], []
    for n, lnd in enumerate(cts, 1):
        todo = pan[pan.lndb == lnd]
        name = f"LNDb-{lnd:04d}"
        with tempfile.TemporaryDirectory(dir=raw, prefix="_x_") as tmp:
            tmp = Path(tmp)
            want = defaultdict(list)
            for b in (f"{name}.mhd", f"{name}.raw"):
                want[idx[b][0]].append(idx[b][1])
            for s in sorted({s for sl in todo.slots for s in sl}):
                for b in (f"{name}_rad{s}.mhd", f"{name}_rad{s}.raw"):
                    want[idx[b][0]].append(idx[b][1])
            for arc, members in want.items():
                extract(sz, arc, members, tmp)
            vol, sp, off, dirn = read_mhd(tmp / f"{name}.mhd")
            masks = {s: read_mhd(tmp / f"{name}_rad{s}.mhd")[0]
                     for s in sorted({s for sl in todo.slots for s in sl})}
            for row in todo.itertuples():
                hint = world_to_voxel(row.xyz, sp, off, dirn)
                built = nodule_npz(vol, masks, row, scale, hint)
                if built is None:
                    skipped.append(f"{name} f{row.finding}")
                    continue
                z, geo = built
                # the mask centre must sit on the release's own centroid
                gap = np.abs(np.subtract(geo["centre_vox"], geo["hint_vox"]))
                if (gap[:2] > 10).any() or gap[2] > 10:
                    far.append(f"{name} f{row.finding} {gap.tolist()}")
                fname = f"{name}_n{row.finding:02d}.npz"
                np.savez_compressed(out / fname, **z)
                rows.append({"file": fname, "patient_id": name, "nodule_idx": row.finding,
                             "n_readers": len(row.readers),
                             "reader_ids": "|".join(map(str, row.readers)),
                             "malignancy_scores": "|".join(map(str, row.ratings[:, -1])),
                             "label": int(z["label"]),
                             "disagreement": float(z["disagreement"]),
                             "hu_min": float(vol.min()), "spacing_z": float(sp[2])})
        print(f"[{n}/{len(cts)}] {name}: {len(todo)} nodules", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "index.csv", index=False)
    df.to_csv(out / "manifest.csv", index=False)
    prov = {"source": "LNDb (Pedrosa et al. 2019), training release, CC BY-NC-ND 4.0",
            "role": "external test of the LIDC-trained arms; never trained on",
            "min_readers": min_readers, "patch_size": PATCH_SIZE, "neighbours": NEIGHBOURS,
            "hu_window": [HU_MIN, HU_MAX],
            "per_attribute_scale_from": "lidc_report (not refitted)",
            "per_attribute_scale": dict(zip(LIDC_ATTRS, map(float, scale))),
            "items": len(df), "cts": int(df.patient_id.nunique()) if len(df) else 0,
            "labels": df.label.value_counts().to_dict() if len(df) else {},
            "readers": df.n_readers.value_counts().to_dict() if len(df) else {},
            "skipped_no_mask": skipped, "centre_far_from_release_centroid": far}
    (out / "preprocess_provenance.json").write_text(json.dumps(prov, indent=2, default=int),
                                                    encoding="utf-8")
    print(f"LNDb: {len(df)} nodules from {prov['cts']} CTs -> {out}; "
          f"skipped {len(skipped)} without a mask; {len(far)} centred > 10 voxels "
          f"from the release centroid")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-readers", type=int, default=MIN_READERS)
    ap.add_argument("--limit", type=int, default=None, help="first N CTs only (a quick check)")
    a = ap.parse_args()
    main(a.min_readers, a.limit)
