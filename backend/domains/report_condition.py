"""The REPORT-LEVEL disagreement condition of a cohort.

A reader's report on a case is more than one rating: LIDC's
readers score seven ordinal characteristics of every nodule, and Chákṣu's
experts record ten measurements of every optic disc. How much the panel
disagrees about a case is a property of the whole report. This module writes a
CONDITION of an existing cohort (same images, labels and frozen split) whose
`disagreement` is the report-level panel disagreement, and which keeps

    d_single    the single-attribute d (LIDC malignancy, Chákṣu VCDR
                on the 0.1 grid), so every rung of the ladder trains exactly as it
                does on the parent cohort;
    d_profile   (P,) the panel's disagreement on EACH attribute, each in units of
                that attribute's spread over the cohort;
    disagreement = mean(d_profile), the report-level target.

Per-attribute disagreement:
    LIDC     s.d. of the readers' 1..5 ratings (7 ordinal characteristics;
             internalStructure and calcification are nominal and excluded).
    Chákṣu   s.d. of the three cup-to-disc ratios; coefficient of variation
             (s.d. / mean) of the seven size measurements, which are in pixels
             of devices with different resolutions and must be scale-free.
Each is divided by its s.d. over the whole cohort (written to
preprocess_provenance.json), so no attribute dominates by its units.

These constants are FITTED ON THE WHOLE COHORT, test folds included - label
information only, no model output, but not train-only. Measured cost
(refitting the constants on each fold's training items): rank agreement of the report-level target >= 0.9998 on
every test fold, at most 5 of ~270 test items change contested status at the
80th-percentile `contested_d`, and the per-field primary is invariant to the
rescaling by construction. Kept as built so the paper's runs stay
reproducible; a fresh study should fit the scales per fold.

No mask is opened: LIDC's ratings come from `lidc_multi` (built from the
processed nodules by `backend.domains.lidc.field_ratings`, via pylidc's
annotations), Chákṣu's measurements from the decision CSVs that `vcdr_opinion` already reads,
read inside the release zips without extracting anything.

    python -m backend.domains.report_condition --cohort lidc
    python -m backend.domains.report_condition --cohort chaksu
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.config import paths

LIDC_ATTRS = ("subtlety", "sphericity", "margin", "lobulation", "spiculation",
              "texture", "malignancy")
CHAKSU_RATIOS = ("ACDR", "VCDR", "HCDR")
CHAKSU_SIZES = ("Disc Area", "Cup Area", "Rim Area", "Disc Height", "Disc Width",
                "Cup Height", "Cup Width")
CHAKSU_ATTRS = CHAKSU_RATIOS + CHAKSU_SIZES


def chaksu_fields(raw: Path) -> dict[str, dict[str, dict[str, str]]]:
    """series_uid -> expert -> the full CSV row."""
    out: dict = defaultdict(dict)
    for split in ("Train", "Test"):
        zf = zipfile.ZipFile(raw / f"{split}.zip")
        pat = re.compile(rf"{split}/6.0_Glaucoma_Decision/(Expert \d)/(Bosch|Forus|Remidio)\.csv$")
        for member in zf.namelist():
            m = pat.match(member)
            if not m:
                continue
            expert, device = m.groups()
            text = zf.read(member).decode("utf-8-sig", "replace")
            for row in csv.DictReader(io.StringIO(text)):
                row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
                name = (row.get("Images") or "").split("-")[0].strip()
                if name:
                    out[f"{device}_{Path(name).stem}"][expert] = row
    return dict(out)


def chaksu_raw_profile(rows: dict[str, dict[str, str]]) -> np.ndarray:
    """(10,) unnormalised disagreement per attribute; NaN where < 2 valid values."""
    prof = []
    for a in CHAKSU_ATTRS:
        v = []
        for r in rows.values():
            try:
                x = float(r.get(a, ""))
            except ValueError:
                continue
            if np.isfinite(x) and x > 0:
                v.append(x)
        v = np.asarray(v, float)
        if len(v) < 2:
            prof.append(np.nan)
        elif a in CHAKSU_RATIOS:
            prof.append(v.std())
        else:
            prof.append(v.std() / v.mean())
    return np.asarray(prof, float)


def chaksu_raw_consensus(rows: dict[str, dict[str, str]]) -> np.ndarray:
    """(10,) the panel's mean of each recorded measurement; NaN where none valid."""
    out = []
    for a in CHAKSU_ATTRS:
        v = []
        for r in rows.values():
            try:
                x = float(r.get(a, ""))
            except ValueError:
                continue
            if np.isfinite(x) and x > 0:
                v.append(x)
        out.append(np.mean(v) if v else np.nan)
    return np.asarray(out, float)


def lidc_raw_consensus(annotation_ids) -> np.ndarray:
    """(7,) the panel's mean rating of each ordinal characteristic (pylidc)."""
    import pylidc as pl
    anns = pl.query(pl.Annotation).filter(
        pl.Annotation.id.in_([int(a) for a in annotation_ids])).all()
    return np.asarray([np.mean([getattr(a, k) for a in anns]) for k in LIDC_ATTRS], float)


def build(cohort: str, out: Path | None = None) -> Path:
    p = paths()
    if cohort == "lidc":
        src, attrs = Path(p.lidc_multi_processed), LIDC_ATTRS
        dst = Path(out or p.lidc_report_processed)
    elif cohort == "chaksu":
        src, attrs = Path(p.chaksu_vcdr_processed), CHAKSU_ATTRS
        dst = Path(out or p.chaksu_report_processed)
        table = chaksu_fields(Path(p.chaksu_raw))
    else:
        raise ValueError(cohort)
    files = sorted(src.glob("*.npz"))
    raw, single, cons = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if cohort == "lidc":
            names = [str(s) for s in z["d_attributes"]]
            if tuple(names) != LIDC_ATTRS:
                raise ValueError(f"{f.name}: attributes {names} != {LIDC_ATTRS}")
            raw.append(np.asarray(z["d_per_attribute"], float))
            single.append(float(z["d_per_attribute"][-1]))        # malignancy
            cons.append(lidc_raw_consensus(z["annotation_ids"]))
        else:
            uid = str(z["series_uid"])
            if uid not in table:
                raise KeyError(f"{uid}: no expert CSV rows")
            raw.append(chaksu_raw_profile(table[uid]))
            single.append(float(z["disagreement"]))                # VCDR, 0.1 grid
            cons.append(chaksu_raw_consensus(table[uid]))
    R = np.stack(raw)
    fill = np.nanmedian(R, axis=0)
    n_fill = int(np.isnan(R).sum())
    R = np.where(np.isnan(R), fill, R)
    scale = R.std(axis=0)
    if (scale <= 0).any():
        raise ValueError(f"an attribute never varies: {dict(zip(attrs, scale))}")
    prof = R / scale
    comp = prof.mean(axis=1)
    # the panel's CONSENSUS report, standardised per attribute (a training signal only;
    # the evaluated target is `disagreement`)
    C = np.stack(cons)
    C = np.where(np.isnan(C), np.nanmedian(C, axis=0), C)
    C = (C - C.mean(axis=0)) / (C.std(axis=0) + 1e-9)

    dst.mkdir(parents=True, exist_ok=True)
    for f, pr, c, s, cc in zip(files, prof, comp, single, C):
        z = dict(np.load(f, allow_pickle=True))
        z["d_single"] = np.float32(s)
        z["d_profile"] = pr.astype(np.float32)
        z["d_profile_attributes"] = np.asarray(attrs)
        z["d_consensus"] = cc.astype(np.float32)
        z["disagreement"] = np.float32(c)
        np.savez_compressed(dst / f.name, **z)
    for extra in ("index.csv", "manifest.csv"):
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)
    prov = {}
    if (src / "preprocess_provenance.json").exists():
        prov = json.loads((src / "preprocess_provenance.json").read_text(encoding="utf-8"))
    prov["report_condition"] = {
        "parent": str(src), "attributes": list(attrs),
        "per_attribute_scale": dict(zip(attrs, map(float, scale))),
        "missing_filled_with_cohort_median": n_fill,
        "items": len(files),
        "rho_report_vs_single": float(_spearman(comp, np.asarray(single))),
    }
    (dst / "preprocess_provenance.json").write_text(json.dumps(prov, indent=2),
                                                     encoding="utf-8")
    print(f"{cohort}: {len(files)} items -> {dst}; filled {n_fill}; "
          f"rho(report, single) {prov['report_condition']['rho_report_vs_single']:+.3f}; "
          f"80th pct of report d {np.quantile(comp, 0.8):.4f}")
    return dst


def _spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", required=True, choices=("lidc", "chaksu"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build(a.cohort, Path(a.out) if a.out else None)
