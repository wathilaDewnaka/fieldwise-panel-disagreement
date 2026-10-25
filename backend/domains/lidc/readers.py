"""Who actually annotated each nodule, and is the resulting panel real?

    python -m backend.domains.lidc.readers    -> reports/annotation_quality.csv

`pylidc` does not record reader identity: its `Annotation` table knows the
annotation and the scan, but not which of the four radiologists produced it. So
`len(cluster)` counts CONTOURS, and a reader who drew twice is counted as two
independent opinions. That inflates the denominator of every agreement quantity
this project reports — `agreement_frac`, `opinion_dist`, `disagreement` are all
normalised by the panel size.

IDENTITY comes from the release XML, which pairs one `LidcReadMessage` per scan
with four `<readingSession>` elements, one per radiologist. `servicingRadiologistID`
is `anonymous` throughout the public release, so no identity is stable ACROSS
scans — and none is needed, because agreement is only ever computed within one
nodule of one scan. The reader's ORDINAL within its scan (0..3) is sufficient and
is the strongest identity the data supports. The join back to pylidc is
`<noduleID>` -> `Annotation._nodule_id`; verified over all 1018 CT scans, that id
is unique within a scan (0 collisions) and every one of the 6859 annotations
resolves.

CLUSTERS can then be checked. `pylidc.Scan.cluster_annotations()` groups contours
by proximity; it is a clustering, not a ground truth, and fails two ways:

  DUPLICATE READER  one radiologist contributed two contours, so the panel looks
                    bigger than it is -> keep one contour per reader.
  MERGED NODULES    two nearby lesions fell into one cluster, so the "agreement"
                    map describes which LESION each reader drew rather than how
                    much they agreed -> drop the cluster.

Telling them apart is per-reader, and admits no benign reading: a radiologist does
not contour the same nodule twice in two different places. For a reader appearing
more than once, `separation` is the distance between their own contours divided by
those contours' mean diameter, so above `separation_ratio` (1.0) the contours no
longer overlap and that reader marked two lesions. One reader doing so is a stray
contour; `merge_min_split_readers` (2) or more independently doing so is two
lesions, because the readers agree there are two.

Over all 1392 LIDC-IDRI clusters of three or more contours, 19 have fewer distinct
readers than contours, and only 14 of those are visible as more than four contours
— which is why this counts readers rather than range-checking the total.

Nothing above `main()` imports pylidc: clusters arrive as plain `ClusterAnnotation`
values, so every rule is unit-testable on synthetic input. `prepare.py` keeps the
I/O, exactly as it does for `patches.py`.
"""
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# The LIDC schema's default namespace. ElementTree needs it on every path; the
# CXR schema uses a different one (`.../idri`) and so never matches.
NS = {"n": "http://www.nih.gov"}

# The radiologists on a LIDC panel. The release guarantees at most this many
# reading sessions per scan, which makes "more than four" a detectable integrity
# failure rather than a plausible reading.
MAX_READERS = 4

# Decisions, in increasing order of intervention.
INCLUDE = "include"
DEDUPLICATE = "deduplicate"
EXCLUDE = "exclude"

# Defaults, overridable from configs/preprocessing/preprocess_lidc.yaml.
SEPARATION_RATIO = 1.0
MERGE_MIN_SPLIT_READERS = 2


class ReaderIdentityError(RuntimeError):
    """The XML tree cannot supply reader identity for a scan being processed."""


class AnnotationIntegrityError(RuntimeError):
    """A cluster asserts something the LIDC-IDRI protocol cannot produce."""


# --------------------------------------------------------------------------- #
# reader identity, from the release XML
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScanReaders:
    """One scan's map from `noduleID` to the reading session that produced it."""
    series_uid: str
    source: Path
    n_sessions: int
    by_nodule_id: dict[str, int] = field(default_factory=dict)

    def reader_id(self, nodule_id: str) -> int:
        """The 0-based reading session that annotated `nodule_id`.

        Raises rather than defaulting: a missing id means the XML and the pylidc
        database disagree about this scan, and inventing a reader would put a
        fabricated vote into an agreement surface.
        """
        try:
            return self.by_nodule_id[str(nodule_id)]
        except KeyError:
            raise ReaderIdentityError(
                f"{self.series_uid}: annotation noduleID {nodule_id!r} is not in "
                f"{self.source.name}. The XML tree and the pylidc database "
                f"describe different versions of this scan.") from None


def _parse(path: Path) -> ScanReaders | None:
    """One XML file -> its `ScanReaders`, or None if it is not a CT read.

    Only `<unblindedReadNodule>` elements carrying `<characteristics>` are
    indexed, because those are exactly the ones pylidc promotes to an
    `Annotation`: the release omits characteristics for lesions under 3 mm and
    pylidc drops them. Indexing them anyway would claim coverage of annotations
    that do not exist.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    if not root.tag.endswith("LidcReadMessage"):   # `IdriReadMessage` is CXR
        return None

    uid = root.findtext("n:ResponseHeader/n:SeriesInstanceUid", namespaces=NS)
    if not uid:
        return None

    sessions = root.findall("n:readingSession", NS)
    by_nodule = {}
    for reader_id, session in enumerate(sessions):
        for nodule in session.findall("n:unblindedReadNodule", NS):
            nid = nodule.findtext("n:noduleID", namespaces=NS)
            if nid and nodule.find("n:characteristics", NS) is not None:
                by_nodule[nid.strip()] = reader_id

    return ScanReaders(uid.strip(), path, len(sessions), by_nodule)


def _read_order(path: Path) -> tuple[bool, Path]:
    """Sort key that puts a resubmitted CORRECTION after the reading it replaces.

    25 files in the release describe a series another file also describes. 24 are
    identical re-uploads where the choice cannot matter; the one that differs is a
    correction, e.g. `161-resubmitted-correction-3-9-12.xml` beside `161.xml`. A
    correction is by definition the reading that stands, so sorting it last and
    letting the last write win says that, rather than leaving it to whichever
    filename happens to sort first.
    """
    name = path.name.lower()
    return any(w in name for w in ("resubmitted", "correction")), path


class ReaderIndex:
    """The whole XML tree, indexed by series UID.

    Built once per run and held in memory (~1300 files, a few seconds). It is
    deliberately NOT cached to disk: a stale cache would silently reattribute
    annotations, which is the failure class this module exists to remove.
    """

    def __init__(self, scans: dict[str, ScanReaders]):
        self._scans = scans

    def __len__(self) -> int:
        return len(self._scans)

    @classmethod
    def build(cls, xml_root: Path) -> "ReaderIndex":
        """Index every `LidcReadMessage` under `xml_root`.

        The TCIA release ships annotations as a separate `xml/` folder whose
        subfolders are numbered rather than named after patients, so the tree is
        walked and keyed by the UID found INSIDE each file.
        """
        xml_root = Path(xml_root)
        if not xml_root.is_dir():
            raise ReaderIdentityError(
                f"LIDC annotation XML not found at {xml_root}. Reader identity "
                f"cannot be recovered without it; set `lidc_xml` in "
                f"configs/paths.yaml to the release's `xml/` folder.")

        scans = {}
        for path in sorted(xml_root.rglob("*.xml"), key=_read_order):
            parsed = _parse(path)
            if parsed is not None:
                scans[parsed.series_uid] = parsed      # corrections sort last

        if not scans:
            raise ReaderIdentityError(
                f"no LidcReadMessage XML under {xml_root} — found only CXR reads "
                f"or unparseable files.")
        return cls(scans)

    def for_scan(self, series_uid: str) -> ScanReaders:
        try:
            return self._scans[series_uid]
        except KeyError:
            raise ReaderIdentityError(
                f"no annotation XML for series {series_uid}. Preprocessing "
                f"cannot establish how many independent readers this scan had."
            ) from None

    def reader_ids(self, annotations) -> list[int]:
        """Reader ordinals for a list of pylidc `Annotation` objects.

        Every annotation must belong to one scan — a cluster spanning two would
        mean the caller mixed up its inputs, and the ordinals would not be
        comparable across them.
        """
        uids = {a.scan.series_instance_uid for a in annotations}
        if len(uids) != 1:
            raise ReaderIdentityError(
                f"annotations span {len(uids)} scans; reader ordinals are only "
                f"defined within one scan")
        readers = self.for_scan(uids.pop())
        return [readers.reader_id(a._nodule_id) for a in annotations]

    def anomalies(self) -> list[ScanReaders]:
        """Scans whose XML does not have exactly `MAX_READERS` sessions.

        Reported rather than raised on: an unusual session COUNT only matters if
        it also produces an over-full cluster, and that is checked per cluster
        where the evidence actually is. Two scans in the release qualify.
        """
        return [s for s in self._scans.values() if s.n_sessions != MAX_READERS]


# --------------------------------------------------------------------------- #
# is this cluster one lesion read by N independent radiologists?
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClusterAnnotation:
    """One contour, reduced to what the integrity rules actually need."""
    annotation_id: int
    reader_id: int
    centroid: tuple[float, float, float]
    diameter: float
    malignancy: int


@dataclass(frozen=True)
class ClusterAudit:
    """The verdict on one cluster, and the evidence behind it.

    Every field becomes a column of `reports/annotation_quality.csv`, so a reader
    can audit an individual decision rather than taking the aggregate counts on
    trust.
    """
    patient_id: str
    series_uid: str
    cluster_id: int
    annotation_ids: tuple[int, ...]
    reader_ids: tuple[int, ...]
    n_unique_readers: int
    duplicate_reader_count: int
    split_reader_count: int
    max_separation_ratio: float
    issue: str
    decision: str
    resolution_method: str
    kept_annotation_ids: tuple[int, ...]

    @property
    def nodule_cluster_id(self) -> str:
        """Stable name for this cluster — the stem of its .npz file."""
        return f"{self.patient_id}_n{self.cluster_id:02d}"

    @property
    def n_annotations(self) -> int:
        return len(self.annotation_ids)

    @property
    def included(self) -> bool:
        return self.decision != EXCLUDE

    @property
    def suspected_merge(self) -> bool:
        return self.issue == "suspected_merged_nodules"

    def as_row(self) -> dict:
        return {
            "scan_id": self.series_uid,
            "patient_id": self.patient_id,
            "nodule_cluster_id": self.nodule_cluster_id,
            "num_annotations": self.n_annotations,
            "num_unique_readers": self.n_unique_readers,
            "duplicate_reader_count": self.duplicate_reader_count,
            "split_reader_count": self.split_reader_count,
            "max_separation_ratio": round(self.max_separation_ratio, 3),
            "suspected_merge": self.suspected_merge,
            "issue": self.issue,
            "decision": self.decision,
            "resolution_method": self.resolution_method,
            "annotation_ids": "|".join(str(i) for i in self.annotation_ids),
            "reader_ids": "|".join(str(i) for i in self.reader_ids),
            "kept_annotation_ids": "|".join(str(i) for i in self.kept_annotation_ids),
            "included_in_primary_analysis": self.included,
        }


def _separation(anns: list[ClusterAnnotation]) -> float:
    """Widest gap between one reader's own contours, in units of their own size.

    A ratio of 1.0 means the two contours have just stopped overlapping. Returns
    0.0 for a reader with a single contour: nothing is separated from anything,
    which must not read as evidence of a merge.
    """
    worst = 0.0
    for i, a in enumerate(anns):
        for b in anns[i + 1:]:
            scale = (a.diameter + b.diameter) / 2.0
            if scale > 0:                       # a degenerate contour has no opinion
                worst = max(worst, math.dist(a.centroid, b.centroid) / scale)
    return worst


def _deduplicate(by_reader: dict[int, list[ClusterAnnotation]]
                 ) -> list[ClusterAnnotation]:
    """One contour per reader: the one this lesion's other readers corroborate.

    The reference is the mean centroid of readers who contributed exactly ONE
    contour — they are unambiguous about where the lesion is, so they define it,
    and a duplicated reader's vote for THAT lesion is whichever contour sits
    nearest. Picking the largest would bias the panel toward whichever reading was
    more generous; picking the first would let database insertion order decide a
    training target.

    Returns empty when every reader duplicated: there is then no uncontaminated
    reference, and guessing one would be inventing the answer.
    """
    singles = [a for anns in by_reader.values() if len(anns) == 1 for a in anns]
    if not singles:
        return []

    ref = tuple(sum(a.centroid[k] for a in singles) / len(singles) for k in range(3))
    # `annotation_id` breaks ties so the choice is deterministic across runs.
    return [min(by_reader[r], key=lambda a: (math.dist(a.centroid, ref),
                                             a.annotation_id))
            for r in sorted(by_reader)]


def _resolve(by_reader, split_readers, n_unique, min_readers, merge_min_split_readers
             ) -> tuple[str, str, str, list[ClusterAnnotation]]:
    """The rules, as guard clauses -> (issue, decision, method, kept contours)."""
    if len(split_readers) >= merge_min_split_readers:
        # Two lesions in one cluster. Deduplicating would keep one contour per
        # reader but leave them pointing at DIFFERENT lesions, which is worse than
        # dropping the sample: it manufactures a contested region.
        return "suspected_merged_nodules", EXCLUDE, "excluded_suspected_merge", []

    if any(len(anns) > 1 for anns in by_reader.values()):
        kept = _deduplicate(by_reader)
        if not kept:
            return "duplicate_reader_no_reference", EXCLUDE, "excluded_no_reference", []
        if len(kept) < min_readers:
            return "too_few_unique_readers", EXCLUDE, "excluded_after_deduplication", []
        return ("duplicate_reader", DEDUPLICATE,
                "nearest_to_single_contour_readers", kept)

    if n_unique < min_readers:
        return "too_few_unique_readers", EXCLUDE, "excluded_below_min_readers", []

    return "none", INCLUDE, "none", [a for anns in by_reader.values() for a in anns]


def audit_cluster(patient_id: str, series_uid: str, cluster_id: int,
                  annotations: list[ClusterAnnotation], *,
                  min_readers: int = 3,
                  max_readers: int = MAX_READERS,
                  separation_ratio: float = SEPARATION_RATIO,
                  merge_min_split_readers: int = MERGE_MIN_SPLIT_READERS,
                  ) -> ClusterAudit:
    """Decide whether one cluster is a usable multi-reader nodule.

    Raises `AnnotationIntegrityError` only for the impossible case — more distinct
    readers than the protocol has. Everything else is a recorded decision, because
    excluding data must be visible in the report rather than hidden in an
    exception nobody sees.
    """
    if not annotations:
        raise AnnotationIntegrityError(
            f"{patient_id} cluster {cluster_id}: no annotations")

    by_reader: dict[int, list[ClusterAnnotation]] = defaultdict(list)
    for a in annotations:
        by_reader[a.reader_id].append(a)
    n_unique = len(by_reader)

    if n_unique > max_readers:
        # Not recoverable and not expected: a LIDC-IDRI panel is at most four.
        # More distinct ordinals means the XML -> pylidc join is wrong, and every
        # agreement number built on it would be wrong too.
        raise AnnotationIntegrityError(
            f"{patient_id} cluster {cluster_id}: {n_unique} distinct readers, but "
            f"a LIDC-IDRI panel has at most {max_readers}. Reader identity was "
            f"mis-assigned; refusing to build an agreement surface from it.")

    separations = {r: _separation(anns) for r, anns in by_reader.items()}
    split_readers = [r for r, ratio in separations.items() if ratio > separation_ratio]
    issue, decision, method, kept = _resolve(
        by_reader, split_readers, n_unique, min_readers, merge_min_split_readers)

    return ClusterAudit(
        patient_id=patient_id, series_uid=series_uid, cluster_id=cluster_id,
        annotation_ids=tuple(a.annotation_id for a in annotations),
        reader_ids=tuple(a.reader_id for a in annotations),
        n_unique_readers=n_unique,
        duplicate_reader_count=sum(1 for a in by_reader.values() if len(a) > 1),
        split_reader_count=len(split_readers),
        max_separation_ratio=max(separations.values(), default=0.0),
        issue=issue, decision=decision, resolution_method=method,
        kept_annotation_ids=tuple(a.annotation_id for a in kept))


# --------------------------------------------------------------------------- #
# reports/annotation_quality.csv
# --------------------------------------------------------------------------- #
def quality_report(audits: list[ClusterAudit], path: Path) -> pd.DataFrame:
    """One row per audited cluster.

    Written even when nothing was wrong: a report that only appears on failure
    cannot be cited as evidence that the check ran.
    """
    df = pd.DataFrame([a.as_row() for a in audits])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def summarise(df: pd.DataFrame) -> str:
    """The counts this report exists to let someone quote."""
    miscounted = df[df.num_annotations != df.num_unique_readers]
    hidden = int((miscounted.num_annotations <= MAX_READERS).sum())
    return "\n".join([
        f"clusters audited                       {len(df)}",
        f"contours != distinct readers           {len(miscounted)}",
        f"  visible as >{MAX_READERS} contours               {len(miscounted) - hidden}",
        f"  indistinguishable from a legal panel {hidden}",
        "", df.decision.value_counts().to_string(),
        "", df.issue.value_counts().to_string(),
    ])


def main() -> None:
    """Regenerate the report on its own, from the pylidc database and the XML.

    `prepare.py` writes this file as a by-product of preprocessing. Running it
    standalone reads no DICOM and rebuilds no .npz, so the audit can be reviewed
    BEFORE committing to a multi-hour re-run — and it is the same `audit_cluster`
    either way, so the report cannot drift from the code that built the data.
    """
    # Imported here, not at module scope, so everything above stays testable
    # without pylidc installed. `prepare` must come BEFORE any pylidc call: it
    # carries the compatibility aliases pylidc 0.2.3 needs on a modern
    # NumPy/Python, without which `cluster_annotations()` dies on `np.int`.
    import pylidc as pl
    from tqdm import tqdm

    from backend.config import ROOT
    from backend.domains.lidc.prepare import audit_scan, xml_root

    ap = argparse.ArgumentParser(
        description="Audit LIDC reader identity and cluster integrity.")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "reports" / "annotation_quality.csv")
    ap.add_argument("--xml", type=Path, default=None,
                    help="annotation XML root (default: lidc_xml in paths.yaml)")
    ap.add_argument("--max-patients", type=int, default=None,
                    help="truncate to the first N scans for a quick check")
    args = ap.parse_args()

    readers = ReaderIndex.build(args.xml or xml_root())
    print(f"reader index: {len(readers)} scans")
    for odd in readers.anomalies():
        print(f"  [note] {odd.source.name}: {odd.n_sessions} reading sessions "
              f"(expected {MAX_READERS})")

    scans = pl.query(pl.Scan).all()
    if args.max_patients:
        scans = scans[:args.max_patients]

    audits = []
    for scan in tqdm(scans, desc="scans"):
        try:
            audits.extend(audit_scan(scan, readers))
        except Exception as e:                   # pragma: no cover - data dependent
            tqdm.write(f"[skip] {scan.patient_id}: {type(e).__name__}: {e}")

    df = quality_report(audits, args.out)
    print(f"\n{summarise(df)}\n\nWrote {args.out}")

    flagged = df[df.issue != "none"]
    if len(flagged):
        cols = ["nodule_cluster_id", "num_annotations", "num_unique_readers",
                "split_reader_count", "max_separation_ratio", "issue", "decision"]
        print(f"\n{len(flagged)} flagged cluster(s):")
        print(flagged[cols].to_string(index=False))


if __name__ == "__main__":
    main()
