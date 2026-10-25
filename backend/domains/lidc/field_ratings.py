"""The panel's disagreement on each of LIDC-IDRI's seven ordinal characteristics.

    python -m backend.domains.lidc.field_ratings      # lidc-idri -> lidc_multi

Copies every processed nodule (`paths().processed`) to `paths().lidc_multi_processed`
and adds two arrays, read from pylidc's annotation table for the readers already
recorded in the sample (`annotation_ids`, one per distinct reader):

    d_attributes     (7,) the characteristic names, in `report_condition.LIDC_ATTRS`
                     order (malignancy last)
    d_per_attribute  (7,) float32, the population s.d. (divisor R) of the readers'
                     1..5 ratings of each characteristic

`backend.domains.report_condition --cohort lidc` builds the report-level cohort from
this folder. Nothing else in the sample changes. Needs the [lidc] environment
(pylidc), but not the DICOM images: the ratings come from pylidc's own database.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from backend.config import paths
from backend.domains.report_condition import LIDC_ATTRS


def per_attribute_sd(annotation_ids) -> np.ndarray:
    """(7,) population s.d. of the readers' ratings of each characteristic."""
    import pylidc as pl
    ids = [int(a) for a in annotation_ids]
    anns = pl.query(pl.Annotation).filter(pl.Annotation.id.in_(ids)).all()
    if len(anns) != len(ids):
        raise ValueError(f"pylidc returned {len(anns)} of {len(ids)} annotations {ids}")
    ratings = np.asarray([[getattr(a, k) for k in LIDC_ATTRS] for a in anns], float)
    return ratings.std(axis=0)


def build(src: Path | None = None, dst: Path | None = None) -> Path:
    p = paths()
    src = Path(src or p.processed)
    dst = Path(dst or p.lidc_multi_processed)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npz"))
    for f in tqdm(files, desc="nodules"):
        z = dict(np.load(f, allow_pickle=True))
        z["d_attributes"] = np.asarray(LIDC_ATTRS)
        z["d_per_attribute"] = per_attribute_sd(z["annotation_ids"]).astype(np.float32)
        np.savez_compressed(dst / f.name, **z)
    for extra in ("index.csv", "manifest.csv", "preprocess_provenance.json"):
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)
    print(f"{len(files)} nodules -> {dst}")
    return dst


if __name__ == "__main__":
    build()
