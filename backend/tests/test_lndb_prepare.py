"""LNDb preprocessing: the reader join, MetaImage reading and the voxel geometry,
on tiny synthetic inputs (the release itself is 24 GB and licensed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.domains.lndb import prepare as L


def _csvs(tmp_path, misalign=False):
    tn = pd.DataFrame({"LNDbID": [1, 1, 1, 1], "RadID": [1, 2, 3, 1],
                       "FindingID": [1, 1, 1, 2], "x": 0.0, "y": 0.0, "z": 0.0,
                       "Nodule": [1, 1, 1, 1], "Volume": 100.0, "Text": [5, 4, 5, 3]})
    ch = pd.DataFrame({"LNDbID": [1, 1, 1, 1], "RadID": [4, 5, 2, 4],     # GLOBAL readers
                       "FindingID": [1, 1, 1, 2] if not misalign else [2, 1, 1, 1],
                       "calcification": 6, "internalStructure": 1,
                       "lobulation": [1, 2, 1, 1], "malignancy": [2, 4, 0, 3],
                       "margin": 4, "sphericity": 3, "spiculation": 1, "subtlety": 5})
    gt = pd.DataFrame({"LNDbID": [1, 1], "RadID": ["1,2,3", "1"], "RadFindingID": ["1,1,1", "2"],
                       "FindingID": [1, 2], "x": 1.0, "y": 2.0, "z": 3.0, "AgrLevel": [3, 1],
                       "Nodule": [1, 1], "Volume": 100.0, "Text": 5.0})
    tn.to_csv(tmp_path / "trainNodules.csv", index=False)
    ch.to_csv(tmp_path / "chars_trainNodules.csv", index=False)
    gt.to_csv(tmp_path / "trainNodules_gt.csv", index=False)
    return tmp_path


def test_readers_are_joined_by_row_and_unrated_readers_dropped(tmp_path):
    pan = L.panels(_csvs(tmp_path), min_readers=2)
    assert len(pan) == 1                                  # finding 2 has one reader
    r = pan.iloc[0]
    assert r.slots == [1, 2] and r.readers == [4, 5]      # slot 3 rated malignancy 0
    # LIDC field order; texture comes from trainNodules' `Text`
    assert r.ratings.tolist() == [[5, 3, 4, 1, 1, 5, 2], [5, 3, 4, 2, 1, 4, 4]]


def test_a_misaligned_characteristics_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="row-aligned"):
        L.panels(_csvs(tmp_path, misalign=True))


def test_read_mhd_and_column_major_direction(tmp_path):
    vol = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)     # (Z, Y, X)
    vol.tofile(tmp_path / "ct.raw")
    # a rotation about z by 90 degrees, written as MetaIO writes it (column by column)
    (tmp_path / "ct.mhd").write_text(
        "ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\n"
        "CompressedData = False\nTransformMatrix = 0 1 0 -1 0 0 0 0 1\n"
        "Offset = 10 20 30\nElementSpacing = 0.5 0.5 2\nDimSize = 4 3 2\n"
        "ElementType = MET_SHORT\nElementDataFile = ct.raw\n")
    arr, sp, off, d = L.read_mhd(tmp_path / "ct.mhd")
    assert arr.shape == (2, 3, 4) and (arr == vol).all()
    # voxel x-axis points along world +y: column 0 of the direction matrix
    assert np.allclose(d[:, 0], [0, 1, 0])
    vox = np.array([2, 1, 1])
    world = off + d @ (vox * sp)
    assert (L.world_to_voxel(world, sp, off, d) == vox).all()


def _lidc_report_built() -> bool:
    from pathlib import Path
    from backend.config import paths
    root = getattr(paths(), "lidc_report_processed", None)
    return bool(root) and (Path(root) / "preprocess_provenance.json").is_file()


@pytest.mark.skipif(not _lidc_report_built(),
                    reason="needs the preprocessed LIDC-IDRI report-level cohort")
def test_scales_are_lidcs_and_the_profile_uses_them():
    s = L.lidc_scales()
    assert s.shape == (7,) and (s > 0).all()


def test_lndb_folds_are_patient_disjoint_and_class_balanced():
    """Training on LNDb: no patient in two of train/val/test, and every test fold
    holds positives (the release is 20% positive)."""
    from backend.config import paths
    from field_disagreement import splits as S
    from backend.domains.datasets import DATASETS
    from backend.config import ROOT
    proc = paths().lndb_report_processed
    if not (proc / "index.csv").exists():
        pytest.skip("LNDb not processed on this machine")
    df = S.build_manifest(proc)
    import json
    assign = json.loads((ROOT / DATASETS["lndb"].folds).read_text())["assignment"]
    for k in range(5):
        sp = S.make_fold(df, assign, k, 5)
        S.assert_no_leakage(df, sp)
        lab = df.set_index("file").loc[sp["test"], "label"]
        assert (lab == 1).sum() >= 5 and (lab == 0).sum() >= 20
