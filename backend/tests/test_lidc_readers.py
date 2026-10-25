"""Reader identity and cluster integrity — the rules, on synthetic input.

`readers.py` decides how many INDEPENDENT radiologists a nodule had, which sets
the denominator of every agreement quantity the project reports. These tests fix
that behaviour on constructed clusters so it can be reasoned about without the
DICOM tree, pylidc, or a 200 GB download.

The XML tests build a miniature `LidcReadMessage` for the same reason: the join
from pylidc's `_nodule_id` back to a reading session is the one step that could
silently reattribute a vote, so it is pinned here rather than trusted.
"""
from __future__ import annotations

import pytest

from backend.domains.lidc.readers import (DEDUPLICATE, EXCLUDE, INCLUDE,
                                          AnnotationIntegrityError,
                                          ClusterAnnotation, ReaderIdentityError,
                                          ReaderIndex, audit_cluster)


# --------------------------------------------------------------------------- #
# stale output from a previous run
# --------------------------------------------------------------------------- #
def test_files_a_rerun_no_longer_writes_are_deleted(tmp_path):
    """A nodule a new run EXCLUDES must not survive on disk. `build_manifest`
    globs *.npz and documents that it ignores index.csv, so a leftover file walks
    straight back into training."""
    from backend.domains.lidc.prepare import prune_stale

    for name in ("keep.npz", "excluded.npz"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "manifest.csv").write_text("stale", encoding="utf-8")

    removed = prune_stale(tmp_path, [{"file": "keep.npz"}])

    assert removed == ["excluded.npz"]
    assert (tmp_path / "keep.npz").exists()
    assert not (tmp_path / "excluded.npz").exists()
    assert not (tmp_path / "manifest.csv").exists()   # cache describes the old cohort


def test_pruning_leaves_non_nodule_files_alone(tmp_path):
    from backend.domains.lidc.prepare import prune_stale

    (tmp_path / "keep.npz").write_bytes(b"")
    (tmp_path / "index.csv").write_text("x", encoding="utf-8")
    (tmp_path / "preprocess_provenance.json").write_text("{}", encoding="utf-8")

    prune_stale(tmp_path, [{"file": "keep.npz"}])

    assert (tmp_path / "index.csv").exists()
    assert (tmp_path / "preprocess_provenance.json").exists()

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def ann(aid, reader, x=0.0, diameter=10.0, malignancy=3):
    """One contour on the x axis; y and z fixed so distances are readable."""
    return ClusterAnnotation(annotation_id=aid, reader_id=reader,
                             centroid=(x, 0.0, 0.0), diameter=diameter,
                             malignancy=malignancy)


def audit(anns, **kw):
    return audit_cluster("LIDC-IDRI-0001", "1.2.3", 0, anns, **kw)


# --------------------------------------------------------------------------- #
# the ordinary case
# --------------------------------------------------------------------------- #
def test_four_distinct_readers_are_included_unchanged():
    a = [ann(i, i, x=float(i)) for i in range(4)]
    r = audit(a)
    assert r.decision == INCLUDE
    assert r.n_unique_readers == 4
    assert r.kept_annotation_ids == (0, 1, 2, 3)
    assert r.issue == "none" or r.issue == ""


def test_three_readers_meet_the_minimum():
    assert audit([ann(i, i) for i in range(3)]).decision == INCLUDE


def test_two_readers_fall_below_the_minimum():
    r = audit([ann(0, 0), ann(1, 1)])
    assert r.decision == EXCLUDE
    assert r.n_unique_readers == 2


# --------------------------------------------------------------------------- #
# the bug this module exists for: contours counted as people
# --------------------------------------------------------------------------- #
def test_a_reader_who_drew_twice_is_one_reader_not_two():
    """Five contours, four radiologists. The panel size is four."""
    a = [ann(0, 0), ann(1, 1), ann(2, 2), ann(3, 3), ann(4, 1, x=2.0)]
    r = audit(a)
    assert r.n_annotations == 5
    assert r.n_unique_readers == 4          # NOT 5
    assert r.decision == DEDUPLICATE
    assert len(r.kept_annotation_ids) == 4


def test_deduplication_keeps_the_contour_the_other_readers_corroborate():
    """Reader 1's two contours: one on the lesion, one 40 units away.

    Readers 0, 2 and 3 all put the lesion near x=0, so the vote for THIS lesion
    is reader 1's contour at x=1, not the distant one.
    """
    near, far = ann(10, 1, x=1.0), ann(11, 1, x=40.0)
    a = [ann(0, 0, x=0.0), ann(2, 2, x=0.0), ann(3, 3, x=0.0), near, far]
    r = audit(a)
    assert 10 in r.kept_annotation_ids
    assert 11 not in r.kept_annotation_ids


def test_deduplication_is_deterministic_under_input_order():
    a = [ann(0, 0), ann(1, 1), ann(2, 2), ann(3, 3), ann(4, 1, x=2.0)]
    first = audit(a).kept_annotation_ids
    assert audit(list(reversed(a))).kept_annotation_ids == first


def test_more_than_four_distinct_readers_is_impossible_and_raises():
    """Five DISTINCT reader ordinals cannot come from a four-person panel, so
    the identity join is wrong and no agreement surface should be built."""
    with pytest.raises(AnnotationIntegrityError, match="at most 4"):
        audit([ann(i, i) for i in range(5)])


# --------------------------------------------------------------------------- #
# merged nodules
# --------------------------------------------------------------------------- #
def test_two_readers_each_splitting_their_contours_is_a_merged_cluster():
    """Readers 0 and 1 each drew a contour at x=0 AND one at x=60. Both agree
    there are two lesions, so this cluster is not one nodule."""
    a = [ann(0, 0, x=0.0), ann(1, 0, x=60.0),
         ann(2, 1, x=0.0), ann(3, 1, x=60.0),
         ann(4, 2, x=0.0), ann(5, 3, x=0.0)]
    r = audit(a)
    assert r.suspected_merge
    assert r.split_reader_count == 2
    assert r.decision == EXCLUDE
    assert r.kept_annotation_ids == ()


def test_one_reader_splitting_is_a_duplicate_not_a_merge():
    """A single reader's stray second contour is not evidence of two lesions —
    the other three readers saw only one."""
    a = [ann(0, 0, x=0.0), ann(1, 0, x=60.0),
         ann(2, 1, x=0.0), ann(3, 2, x=0.0), ann(4, 3, x=0.0)]
    r = audit(a)
    assert not r.suspected_merge
    assert r.decision == DEDUPLICATE


def test_close_duplicate_contours_do_not_count_as_split():
    """Two contours 2 units apart on a 10-unit lesion overlap heavily: the same
    nodule drawn twice, not two lesions."""
    a = [ann(0, 0, x=0.0), ann(1, 0, x=2.0),
         ann(2, 1, x=0.0), ann(3, 2, x=0.0), ann(4, 3, x=0.0)]
    r = audit(a)
    assert r.split_reader_count == 0
    assert r.decision == DEDUPLICATE


def test_every_reader_duplicated_leaves_no_reference_and_is_excluded():
    """All four readers drew twice, but close together, so the merge rule does
    not fire. There is then no single-contour reader to anchor deduplication."""
    a = [ann(2 * i + j, i, x=float(j)) for i in range(4) for j in range(2)]
    r = audit(a)
    assert r.decision == EXCLUDE
    assert r.resolution_method == "excluded_no_reference"


def test_merge_threshold_is_configurable():
    a = [ann(0, 0, x=0.0), ann(1, 0, x=60.0),
         ann(2, 1, x=0.0), ann(3, 2, x=0.0), ann(4, 3, x=0.0)]
    assert not audit(a).suspected_merge
    assert audit(a, merge_min_split_readers=1).suspected_merge


# --------------------------------------------------------------------------- #
# what the report records
# --------------------------------------------------------------------------- #
def test_audit_row_carries_the_full_provenance():
    a = [ann(0, 0), ann(1, 1), ann(2, 2), ann(3, 3), ann(4, 1, x=2.0)]
    row = audit(a).as_row()
    assert row["num_annotations"] == 5
    assert row["num_unique_readers"] == 4
    assert row["duplicate_reader_count"] == 1
    assert row["nodule_cluster_id"] == "LIDC-IDRI-0001_n00"
    assert row["included_in_primary_analysis"] is True
    assert row["reader_ids"] == "0|1|2|3|1"
    assert row["decision"] == DEDUPLICATE


def test_excluded_clusters_say_so_in_the_row():
    a = [ann(0, 0, x=0.0), ann(1, 0, x=60.0),
         ann(2, 1, x=0.0), ann(3, 1, x=60.0),
         ann(4, 2, x=0.0), ann(5, 3, x=0.0)]
    row = audit(a).as_row()
    assert row["included_in_primary_analysis"] is False
    assert row["issue"] == "suspected_merged_nodules"


# --------------------------------------------------------------------------- #
# the XML -> reader join
# --------------------------------------------------------------------------- #
LIDC_XML = """<?xml version="1.0" encoding="UTF-8"?>
<LidcReadMessage xmlns="http://www.nih.gov">
 <ResponseHeader><SeriesInstanceUid>1.2.3</SeriesInstanceUid></ResponseHeader>
 <readingSession>
  <servicingRadiologistID>anonymous</servicingRadiologistID>
  <unblindedReadNodule><noduleID>A1</noduleID>
    <characteristics><malignancy>3</malignancy></characteristics></unblindedReadNodule>
  <unblindedReadNodule><noduleID>tiny</noduleID></unblindedReadNodule>
 </readingSession>
 <readingSession>
  <unblindedReadNodule><noduleID>B1</noduleID>
    <characteristics><malignancy>4</malignancy></characteristics></unblindedReadNodule>
 </readingSession>
</LidcReadMessage>
"""

CXR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<IdriReadMessage xmlns="http://www.nih.gov/idri">
 <ResponseHeader><CTSeriesInstanceUid>9.9.9</CTSeriesInstanceUid></ResponseHeader>
</IdriReadMessage>
"""


@pytest.fixture
def xml_dir(tmp_path):
    (tmp_path / "158.xml").write_text(LIDC_XML, encoding="utf-8")
    (tmp_path / "068.xml").write_text(CXR_XML, encoding="utf-8")
    (tmp_path / "broken.xml").write_text("<not-xml", encoding="utf-8")
    return tmp_path


def test_reading_session_order_is_the_reader_identity(xml_dir):
    idx = ReaderIndex.build(xml_dir)
    readers = idx.for_scan("1.2.3")
    assert readers.reader_id("A1") == 0
    assert readers.reader_id("B1") == 1
    assert readers.n_sessions == 2


def test_nodules_without_characteristics_are_not_indexed(xml_dir):
    """pylidc drops sub-3mm lesions, which carry no characteristics block. The
    index must not claim to cover annotations that do not exist."""
    with pytest.raises(ReaderIdentityError):
        ReaderIndex.build(xml_dir).for_scan("1.2.3").reader_id("tiny")


def test_cxr_reads_and_unparseable_files_are_skipped(xml_dir):
    idx = ReaderIndex.build(xml_dir)
    assert len(idx) == 1
    with pytest.raises(ReaderIdentityError):
        idx.for_scan("9.9.9")                    # the CXR series was not indexed


def test_an_unknown_scan_raises_rather_than_guessing(xml_dir):
    with pytest.raises(ReaderIdentityError, match="no annotation XML"):
        ReaderIndex.build(xml_dir).for_scan("4.5.6")


def test_a_missing_xml_tree_is_an_actionable_error(tmp_path):
    with pytest.raises(ReaderIdentityError, match="configs/paths.yaml"):
        ReaderIndex.build(tmp_path / "nope")


def test_a_resubmitted_correction_supersedes_the_original(tmp_path):
    """Both files describe series 1.2.3, but they disagree about who read B1.
    The correction is the reading that stands, whichever way the names sort."""
    original = LIDC_XML
    corrected = LIDC_XML.replace(
        "<readingSession>\n  <unblindedReadNodule><noduleID>B1</noduleID>",
        "<readingSession/>\n <readingSession>\n"
        "  <unblindedReadNodule><noduleID>B1</noduleID>")
    (tmp_path / "161.xml").write_text(original, encoding="utf-8")
    (tmp_path / "161-resubmitted-correction.xml").write_text(corrected,
                                                             encoding="utf-8")
    idx = ReaderIndex.build(tmp_path)
    assert idx.for_scan("1.2.3").reader_id("B1") == 2       # corrected, not 1


def test_identical_duplicates_give_the_same_index_whichever_wins(tmp_path):
    """24 of the release's 25 duplicated files are identical re-uploads. Which
    one the index keeps is therefore unobservable — and that, not the filename,
    is what has to hold."""
    (tmp_path / "082.xml").write_text(LIDC_XML, encoding="utf-8")
    (tmp_path / "083.xml").write_text(LIDC_XML, encoding="utf-8")
    idx = ReaderIndex.build(tmp_path)
    assert idx.for_scan("1.2.3").by_nodule_id == {"A1": 0, "B1": 1}


def test_an_xml_tree_with_no_ct_reads_is_an_error(tmp_path):
    (tmp_path / "068.xml").write_text(CXR_XML, encoding="utf-8")
    with pytest.raises(ReaderIdentityError, match="only CXR"):
        ReaderIndex.build(tmp_path)
