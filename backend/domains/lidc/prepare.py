"""
LIDC-IDRI 2.5D preprocessing: adjacent axial slices plus a per-pixel reader
agreement surface A(x,y), shared by every study of the framework.

For every nodule reviewed by >= MIN_READERS radiologists this script saves:

  image          (C, P, P)  float32   2.5D = centre axial slice +/- NEIGHBOURS
                                       adjacent axial slices, windowed -> [0,1]
  agreement_map  (P, P)     uint8      A(x,y): per-pixel reader count 0..R
  agreement_frac (P, P)     float32    A(x,y) / n_readers, in [0,1] (comparable
                                       across 3- vs 4-reader nodules)
  binary_mask    (P, P)     uint8      1 where ANY radiologist marked nodule
  malignancy     (R,)       int8       per-reader malignancy scores (1..5)
  opinion_dist   (5,)       float32    normalised histogram over scores 1..5
                                       (soft-label / opinion-distribution target)
  disagreement   float32               std of the per-reader ratings (the spread;
                                       the panel-disagreement target)
  rating_entropy float32               Shannon entropy of opinion_dist (nats)
  n_readers      int8                  number of DISTINCT radiologists (<= 4)
  label          int8                  1 = malignant, 0 = benign, -1 = ambiguous
  reader_ids     (R,)      int8        which reading session produced each vote
  annotation_ids (R,)      int64       pylidc annotation ids, same order
  patient_id     str
  series_uid     str
  nodule_idx     int

  -> Each nodule becomes one .npz file.
  -> An index.csv is written for easy patient-level splitting in training.

Notes
-----
* The 2.5D channels are all the SAME (axial) plane at adjacent depths, so every
  channel and A(x,y) share one coordinate frame.
* agreement_map is the raw reader count; agreement_frac normalises it by the
  panel size (a count of 3 means "all" for a 3-reader nodule but "3 of 4" for a
  4-reader one). The binary nodule mask is the union (any reader); any stricter
  consensus is recoverable by thresholding agreement_map.
* opinion_dist and disagreement are derived only from the per-reader ratings, so
  they do not change the images -- they make the panel targets explicit.
* n_readers counts DISTINCT RADIOLOGISTS, not contours. pylidc's clustering can
  put two contours from one reader into one nodule, and counting those as two
  independent opinions inflates the denominator of agreement_frac and
  double-counts that reader in opinion_dist. Reader identity is recovered from
  the release XML and each cluster is audited before use — both in
  `backend/domains/lidc/readers.py` — and every decision is written to
  reports/annotation_quality.csv.

Labelling rule (standard LIDC convention, median over per-reader scores):
    median malignancy > 3  -> 1 (malignant)
    median malignancy < 3  -> 0 (benign)
    median malignancy == 3 -> -1 (ambiguous, excluded from binary training)
"""
from __future__ import annotations

import configparser
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# pylidc 0.2.3 (2020) calls two names that have since been removed, and both
# failures land mid-run rather than at import: `np.int` inside `Contour.to_matrix`
# and `Annotation.boolean_mask` (gone in NumPy 1.24), and
# `configparser.SafeConfigParser` inside `Scan.to_volume` (gone in Python 3.12).
# Every pylidc call below reaches one or the other, so there is no spelling to
# migrate to inside a dependency we do not own. Both were pure aliases, so
# restoring them undoes the removal and changes nothing else. Each is guarded, so
# an older interpreter keeps its own definition.
for _name, _builtin in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm

from backend.config import ROOT, load, paths
from backend.domains.lidc.patches import (agreement_surface, pad_and_crop,
                                          rating_targets, window_hu)
from backend.domains.lidc.readers import (EXCLUDE, ClusterAnnotation,
                                          ClusterAudit, ReaderIndex,
                                          audit_cluster, quality_report)

# Every knob lives in configs/preprocessing/preprocess_lidc.yaml; the output root in
# configs/paths.yaml. Nothing machine-specific appears in this file, so the same
# commit reproduces the same processed/<dataset>/ on any machine.
CFG = load("preprocessing/preprocess_lidc")

PATCH_SIZE     = CFG.patch_size
NEIGHBOURS     = CFG.neighbours
HU_MIN, HU_MAX = CFG.hu_min, CFG.hu_max
CONSENSUS_LVL  = CFG.consensus_level
MIN_READERS    = CFG.min_readers
MAX_PATIENTS   = CFG.max_patients

# Cluster-integrity thresholds — see clusters.py for what they discriminate.
SEPARATION_RATIO = CFG.separation_ratio
MERGE_MIN_SPLIT  = CFG.merge_min_split_readers

NUM_CHANNELS = 2 * NEIGHBOURS + 1


def audit_scan(scan: pl.Scan, readers: ReaderIndex, nodules=None) -> list:
    """Every cluster in a scan, with its integrity verdict attached.

    Split from the pixel work so the audit can be run — and its report
    regenerated — without reading a single DICOM. Clusters too small to be usable
    whatever their reader identity are dropped before the (non-trivial) centroid
    and diameter computations.

    `nodules` lets a caller that has already clustered pass the result in.
    `cluster_annotations()` is the expensive half of preprocessing, and the
    cluster_id in every audit indexes into that exact list — so sharing one call
    is both faster and the only way the ids are guaranteed to line up.
    """
    if nodules is None:
        nodules = scan.cluster_annotations()

    audits = []
    for cluster_id, anns in enumerate(nodules):
        if len(anns) < MIN_READERS:
            # Fewer CONTOURS than the minimum, so fewer distinct readers too: no
            # audit can rescue it. It is still RECORDED, so the quality report
            # accounts for every cluster pylidc produced rather than starting
            # from the ones that survived a filter applied before it. The
            # separation ratio is NaN because it is genuinely not computed here —
            # that is the expensive half, and the verdict does not depend on it.
            audits.append(ClusterAudit(
                patient_id=scan.patient_id,
                series_uid=scan.series_instance_uid,
                cluster_id=cluster_id,
                annotation_ids=tuple(int(a.id) for a in anns),
                reader_ids=tuple(int(r) for r in readers.reader_ids(anns)),
                n_unique_readers=len(set(readers.reader_ids(anns))),
                duplicate_reader_count=0,
                split_reader_count=0,
                max_separation_ratio=float("nan"),
                issue="too_few_contours",
                decision=EXCLUDE,
                resolution_method="excluded_below_min_contours",
                kept_annotation_ids=()))
            continue
        reader_ids = readers.reader_ids(anns)
        audits.append(audit_cluster(
            patient_id=scan.patient_id,
            series_uid=scan.series_instance_uid,
            cluster_id=cluster_id,
            annotations=[
                ClusterAnnotation(annotation_id=int(a.id), reader_id=int(r),
                                  centroid=tuple(float(c) for c in a.centroid),
                                  diameter=float(a.diameter),
                                  malignancy=int(a.malignancy))
                for a, r in zip(anns, reader_ids)],
            min_readers=MIN_READERS,
            separation_ratio=SEPARATION_RATIO,
            merge_min_split_readers=MERGE_MIN_SPLIT))
    return audits


def series_tag(scan: pl.Scan, series_by_patient: dict[str, list[str]]) -> str:
    """The token that makes an output filename unique when a patient was scanned
    more than once.

    THE BUG THIS FIXES. Files were named `{patient_id}_n{cluster_idx:02d}.npz`.
    `pl.query(pl.Scan)` yields one Scan per SERIES, and 7 LIDC-IDRI patients have
    two series each. Cluster index 3 of patient 0132's first series and cluster
    index 3 of its second series are DIFFERENT nodules with different panels, and
    both resolved to `LIDC-IDRI-0132_n03.npz` — so the second write silently
    overwrote the first, while BOTH rows were appended to index.csv. That is the
    whole of the 1386-vs-1377 gap: 9 collisions across those 7 patients (the
    other 5 of their clusters happened not to collide on index). The surviving
    file is the last one written, which is deterministic but arbitrary, and its
    9 index.csv rows describe the wrong series.

    The rule is deliberately conservative: a patient with ONE series keeps the
    bare name, so a rebuild leaves the other patients' filenames where they were. Only the 7 multi-series patients gain `_s{k}`,
    where k indexes the patient's series sorted by UID — sorted, so the token
    does not depend on the order pylidc happens to return scans in.

    `disambiguate_series` in preprocess_provenance.json records which rule built
    a given folder, so a cohort built before this fix identifies itself.
    """
    uids = series_by_patient.get(scan.patient_id, [])
    if len(uids) < 2:
        return ""
    return f"_s{uids.index(scan.series_instance_uid)}"


def process_scan(scan: pl.Scan, rows: list[dict], out_dir: Path,
                 readers: ReaderIndex, audits: list,
                 series_by_patient: dict[str, list[str]] | None = None) -> None:
    nodules = scan.cluster_annotations()      # list[list[Annotation]]
    if not nodules:
        return
    tag = series_tag(scan, series_by_patient or {})

    scan_audits = audit_scan(scan, readers, nodules)
    audits.extend(scan_audits)
    usable = {a.cluster_id: a for a in scan_audits if a.decision != EXCLUDE}
    for a in scan_audits:
        if a.decision == EXCLUDE:
            tqdm.write(f"[drop] {a.nodule_cluster_id}: {a.issue} "
                       f"({a.n_annotations} contours, {a.n_unique_readers} readers)")
    if not usable:
        return

    vol = scan.to_volume(verbose=False)       # (H, W, Z), HU
    H, W, Z = vol.shape

    for nod_idx, anns in enumerate(nodules):
        audit = usable.get(nod_idx)
        if audit is None:
            continue

        # ONE contour per radiologist, in reader order. Every count below is now
        # a count of independent opinions: the agreement surface's denominator,
        # the opinion histogram's sample size, and the spread that supervises
        # deferral all mean what their names say.
        keep = set(audit.kept_annotation_ids)
        anns = [a for a in anns if int(a.id) in keep]
        n_readers = len(anns)

        # per-reader 3D masks aligned in a shared bbox
        _, cbbox, per_reader_masks = consensus(
            anns, clevel=CONSENSUS_LVL, ret_masks=True
        )

        # nodule centre (in full-volume coords)
        ci = (cbbox[0].start + cbbox[0].stop) // 2
        cj = (cbbox[1].start + cbbox[1].stop) // 2
        ck = (cbbox[2].start + cbbox[2].stop) // 2

        # ----- agreement surface A(x,y) at the centre axial slice -----
        local_z = ck - cbbox[2].start
        agreement_patch, agreement_frac, binary_patch = agreement_surface(
            [m3d[..., local_z] for m3d in per_reader_masks],
            bbox=(cbbox[0], cbbox[1]), full_shape=(H, W),
            centre=(ci, cj), patch_size=PATCH_SIZE)

        # A nodule whose readers annotated disjoint z-ranges gets a tall union
        # bbox whose CENTRE slice nobody marked: the agreement surface comes out
        # empty, and the readers' masks cannot place the nodule in the patch.
        # Skip it loudly here. (One nodule in LIDC-IDRI: 0179_n08.)
        if not binary_patch.any():
            tqdm.write(f"[skip] {scan.patient_id} n{nod_idx:02d}: empty agreement "
                       f"surface at the centre slice ({n_readers} readers)")
            continue

        # ----- 2.5D image patch (adjacent axial slices) -----
        slices = []
        for dz in range(-NEIGHBOURS, NEIGHBOURS + 1):
            z = int(np.clip(ck + dz, 0, Z - 1))
            slices.append(pad_and_crop(vol[:, :, z], ci, cj, PATCH_SIZE))
        image = window_hu(np.stack(slices, axis=0), HU_MIN, HU_MAX)  # (C, P, P)

        # ----- rating targets (opinion distribution + disagreement) -----
        # One score per DISTINCT reader — `anns` was reduced to that above, so a
        # radiologist who drew twice no longer votes twice.
        reader_ids = readers.reader_ids(anns)
        scores = np.array([a.malignancy for a in anns], dtype=np.int64)
        t = rating_targets(scores)
        odist, disagreement = t["opinion_dist"], t["disagreement"]
        rating_entropy, label = t["rating_entropy"], t["label"]

        # ----- save -----
        fname = f"{scan.patient_id}{tag}_n{nod_idx:02d}.npz"
        np.savez_compressed(
            out_dir / fname,
            image=image.astype(np.float32),
            agreement_map=agreement_patch,
            agreement_frac=agreement_frac.astype(np.float32),
            binary_mask=binary_patch,
            malignancy=scores.astype(np.int8),
            opinion_dist=odist.astype(np.float32),
            disagreement=np.float32(disagreement),
            rating_entropy=np.float32(rating_entropy),
            n_readers=np.int8(n_readers),
            label=np.int8(label),
            # Provenance of every vote above, so a downstream check can re-derive
            # the panel rather than trusting the scalar count.
            reader_ids=np.asarray(reader_ids, dtype=np.int8),
            annotation_ids=np.asarray([int(a.id) for a in anns], dtype=np.int64),
            patient_id=scan.patient_id,
            series_uid=scan.series_instance_uid,
            nodule_idx=nod_idx,
        )
        rows.append({
            "file": fname,
            "patient_id": scan.patient_id,
            "series_tag": tag,
            "series_uid": scan.series_instance_uid,
            "nodule_idx": nod_idx,
            "n_readers": n_readers,
            "reader_ids": "|".join(map(str, reader_ids)),
            "n_contours": audit.n_annotations,
            "reader_resolution": audit.resolution_method,
            "malignancy_scores": "|".join(map(str, scores.tolist())),
            "median_malignancy": float(np.median(scores)),
            "opinion_dist": "|".join(f"{p:.3f}" for p in odist),
            "disagreement": round(disagreement, 4),
            "rating_entropy": round(rating_entropy, 4),
            "label": int(label),
        })


def xml_root() -> Path:
    """Where the release's annotation XML lives.

    `lidc_xml` in configs/paths.yaml when set; otherwise the `xml/` folder beside
    the DICOM tree, which is where the TCIA download puts it. Falling back rather
    than requiring the key keeps every existing paths.yaml working.
    """
    P = paths()
    explicit = getattr(P, "lidc_xml", None)
    return Path(explicit) if explicit else Path(P.lidc_raw).parent / "xml"


def prune_stale(out_dir: Path, rows: list[dict]) -> list[str]:
    """Delete .npz files this run did not write, and the manifest cache.

    Preprocessing writes one file per nodule and overwrites in place, so a nodule
    that a NEW run excludes — a merged cluster, a panel that falls below
    `min_readers` once its duplicate votes are removed — survives on disk from the
    previous run. That is not a cosmetic leftover: `field_disagreement.splits.build_manifest`
    globs `*.npz` and documents that it deliberately does NOT read index.csv, so a
    stale file walks straight back into training and the exclusion never happened.

    `manifest.csv` goes for the same reason: it is a cache keyed on nothing, and a
    stale one would describe the previous cohort.
    """
    written = {r["file"] for r in rows}
    stale = sorted(p for p in out_dir.glob("*.npz") if p.name not in written)
    for path in stale:
        path.unlink()

    cache = out_dir / "manifest.csv"
    if cache.exists():
        cache.unlink()

    return [p.name for p in stale]


def main(paper_cohort: bool = False) -> None:
    """Build the cohort. `paper_cohort=True` (`--paper-cohort`) uses the naming
    rule the paper's cohort was built with: no series token, so for the 7
    patients with two series a nodule whose cluster index repeats overwrites the
    earlier one (last write kept), and the folder holds the paper's 1,377 files.
    The default rule names those nodules uniquely and gives 1,386."""
    out_dir = paths().processed
    out_dir.mkdir(parents=True, exist_ok=True)

    # Built before any DICOM is touched: a scan whose reader identities cannot be
    # established must stop the run here, not after hours of pixel work.
    readers = ReaderIndex.build(xml_root())
    print(f"reader index: {len(readers)} scans from {xml_root()}")
    for odd in readers.anomalies():
        print(f"  [note] {odd.source.name}: {odd.n_sessions} reading sessions "
              f"(expected 4)")

    scans = pl.query(pl.Scan).all()
    if MAX_PATIENTS:
        scans = scans[:MAX_PATIENTS]

    # Which series belong to each patient, decided before any pixel work so the
    # output filename does not depend on the order scans come back in. 7 LIDC-IDRI
    # patients have two series; see `series_tag`.
    series_by_patient: dict[str, list[str]] = {}
    for scan in scans:
        series_by_patient.setdefault(scan.patient_id, []).append(
            scan.series_instance_uid)
    for uids in series_by_patient.values():
        uids.sort()
    multi = {p: u for p, u in series_by_patient.items() if len(u) > 1}
    if paper_cohort:
        print(f"--paper-cohort: the paper's naming rule, no series token; "
              f"{len(multi)} patient(s) with two series keep bare filenames")
        series_by_patient = {}
    elif multi:
        print(f"{len(multi)} patient(s) with more than one series; their nodules "
              f"get an `_s{{k}}` token so two series cannot share a filename: "
              f"{', '.join(sorted(multi))}")

    rows: list[dict] = []
    audits: list = []
    for scan in tqdm(scans, desc="scans"):
        try:
            process_scan(scan, rows, out_dir, readers, audits, series_by_patient)
        except Exception as e:
            tqdm.write(f"[skip] {scan.patient_id}: {e}")

    # A filename appearing twice means one nodule's .npz was overwritten by
    # another's while index.csv kept both rows — the cohort would be smaller than
    # its own index and 2 rows per collision would describe the wrong file. It
    # fails the run, except under --paper-cohort, where it is the documented
    # behaviour of the paper's build.
    counts = pd.Series([r["file"] for r in rows]).value_counts()
    clashes = counts[counts > 1]
    if len(clashes) and paper_cohort:
        print(f"--paper-cohort: {len(clashes)} filename(s) written twice; the later "
              f"write is kept, as in the paper's cohort")
    elif len(clashes):
        raise RuntimeError(
            f"{len(clashes)} output filename(s) written more than once, so "
            f"{int(clashes.sum() - len(clashes))} nodule(s) would be silently "
            f"lost: {', '.join(clashes.index[:10])}"
            + (" ..." if len(clashes) > 10 else ""))

    pd.DataFrame(rows).to_csv(out_dir / "index.csv", index=False)

    if MAX_PATIENTS:
        # A truncated run has no opinion about the scans it never looked at, and
        # pruning against its `rows` would delete the rest of the dataset.
        print(f"\n[note] max_patients={MAX_PATIENTS}: skipping the stale-file "
              f"sweep. This output is a smoke test, not a usable cohort.")
    else:
        removed = prune_stale(out_dir, rows)
        if removed:
            print(f"\nremoved {len(removed)} file(s) left by a previous run: "
                  f"{', '.join(removed)}")

    quality = quality_report(audits, ROOT / "reports" / "annotation_quality.csv")
    if len(quality):
        print("\nannotation integrity:")
        print(quality["decision"].value_counts().to_string())
        flagged = quality[quality["issue"] != "none"]
        if len(flagged):
            print(f"\n{len(flagged)} cluster(s) flagged:")
            print(flagged["issue"].value_counts().to_string())

    # Provenance: the exact settings these files were built from, written next to
    # them. `validate` compares it against configs/preprocessing/preprocess_lidc.yaml, so a config
    # edit that was never followed by a re-run is caught rather than assumed.
    (out_dir / "preprocess_provenance.json").write_text(json.dumps({
        "created": date.today().isoformat(),
        "n_nodules": len(rows),
        "n_scans": len(scans),
        # WHICH NAMING RULE built these files. Absent (or false) means the
        # paper's rule, under which a patient with two series lost one nodule per
        # colliding cluster index (1,377 files instead of 1,386). See `series_tag`.
        "disambiguate_series": not paper_cohort,
        "config": vars(CFG),
    }, indent=2), encoding="utf-8")

    print(f"\nDone. {len(rows)} nodules saved to {out_dir}")
    if rows:
        counts = pd.DataFrame(rows)["label"].value_counts()
        print(counts.rename({1: "malignant", 0: "benign", -1: "ambiguous"}))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LIDC-IDRI preprocessing: one .npz per nodule")
    ap.add_argument("--paper-cohort", action="store_true",
                    help="use the paper's naming rule and reproduce its 1,377-file cohort")
    main(paper_cohort=ap.parse_args().paper_cohort)
