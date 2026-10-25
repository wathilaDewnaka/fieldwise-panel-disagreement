"""Chákṣu with GRADED expert opinions: each expert's vertical cup-to-disc ratio.

    python -m backend.domains.chaksu.vcdr_opinion            # writes data/processed/chaksu_vcdr

Why this condition exists
-------------------------
Every expert's glaucoma vote in Chákṣu is a THRESHOLD on that expert's own VCDR:
over all 6,700 (image, expert) pairs the largest VCDR voted "normal" is 0.600 and
the smallest voted "glaucoma suspect" is 0.601 (AUC 1.000). The binary vote is a
lossy projection of a graded clinical measurement, exactly as LIDC's binary
malignancy label is a projection of the 1-5 ratings. On the binary projection the
panel's disagreement is a function of the vote COUNT, so it is the same ranking
as a perfectly calibrated confidence. On the graded opinion
it is not: a perfect confidence correlates only +0.08 with the panel's s.d.

What is and is not used
-----------------------
The VCDR values are read from the SAME decision CSVs `prepare.py` reads the votes
from (`6.0_Glaucoma_Decision/Expert N/<device>.csv`, column `VCDR`), recorded by
each expert beside their decision. No mask is opened, and the model still sees
only the image. The image, label (the panel's majority vote), folds, patient ids
and every spatial field are copied byte-for-byte from `data/processed/chaksu`, so
this is a second CONDITION of Chákṣu, paired item-for-item with it.

Opinion encoding: VCDR on the clinical 0.1 grid -> levels 1..10 (K=10), the way
clinicians record a CDR. `disagreement` is the population s.d. of those levels,
so it is an exact functional of `opinion_dist`, as on LIDC.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.config import paths

K = 10          # VCDR 0.1 .. 1.0 on the clinical 0.1 grid
DECISIONS = "6.0_Glaucoma_Decision"


def read_vcdr(raw: Path) -> dict[str, list[float]]:
    """series_uid ("<device>_<stem>") -> the experts' VCDR values."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for split in ("Train", "Test"):
        zf = zipfile.ZipFile(raw / f"{split}.zip")
        pat = re.compile(rf"{split}/{DECISIONS}/(Expert \d)/(Bosch|Forus|Remidio)\.csv$")
        for member in zf.namelist():
            m = pat.match(member)
            if not m:
                continue
            expert, device = m.groups()
            reader = csv.DictReader(io.StringIO(zf.read(member).decode("utf-8-sig", "replace")))
            for row in reader:
                row = {k.strip(): v for k, v in row.items() if k}
                name = (row.get("Images") or "").split("-")[0].strip()
                try:
                    v = float(row.get("VCDR", ""))
                except ValueError:
                    continue
                if name and np.isfinite(v) and 0.0 < v <= 1.5:
                    out[f"{device}_{Path(name).stem}"][expert] = v
    return {k: list(v.values()) for k, v in out.items()}


def targets(vcdr: list[float]) -> tuple[np.ndarray, float]:
    levels = np.clip(np.round(np.asarray(vcdr) * 10).astype(int), 1, K)
    hist = np.bincount(levels - 1, minlength=K).astype(np.float32)
    return hist / hist.sum(), float(np.std(levels.astype(np.float64)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    p = paths()
    src = Path(p.chaksu_processed)
    dst = Path(a.out or getattr(p, "chaksu_vcdr_processed", "data/processed/chaksu_vcdr"))
    dst.mkdir(parents=True, exist_ok=True)
    table = read_vcdr(Path(p.chaksu_raw))
    written, missing, d_all = 0, [], []
    for f in sorted(src.glob("*.npz")):
        v = table.get(f.stem)
        if not v or len(v) < 4:
            missing.append(f.stem)
            continue
        d = dict(np.load(f, allow_pickle=True))
        q, s = targets(v)
        d["opinion_dist"], d["disagreement"] = q, np.float32(s)
        d["rating_entropy"] = np.float32(-(q[q > 0] * np.log(q[q > 0])).sum())
        d["vcdr"] = np.asarray(v, dtype=np.float32)
        np.savez_compressed(dst / f.name, **d)
        written += 1
        d_all.append(s)
    # the index is what the split machinery reads; carry it over with the new
    # targets so nothing downstream sees the binary-vote columns
    import pandas as pd
    index = pd.read_csv(src / "index.csv")
    index = index[index.file.map(lambda f: (dst / f).exists())].copy()
    tq = {f: targets(table[Path(f).stem]) for f in index.file}
    index["opinion_dist"] = [ "|".join(f"{x:.3f}" for x in tq[f][0]) for f in index.file]
    index["disagreement"] = [round(tq[f][1], 4) for f in index.file]
    index["vcdr"] = ["|".join(f"{x:.3f}" for x in table[Path(f).stem]) for f in index.file]
    index.to_csv(dst / "index.csv", index=False)
    for extra in ("manifest.csv", "preprocess_provenance.json"):
        if (src / extra).exists():
            (dst / extra).write_bytes((src / extra).read_bytes())
    d_all = np.asarray(d_all)
    print(f"wrote {written} items to {dst}; {len(missing)} without >=4 VCDRs: {missing[:5]}")
    print("disagreement (s.d. of 0.1-VCDR levels): "
          + ", ".join(f"q{int(q * 100)} {np.quantile(d_all, q):.3f}" for q in (.5, .75, .85, .9)))
    for t in (0.5, 0.75, 0.8, 1.0):
        print(f"  d >= {t}: {100 * (d_all >= t - 1e-9).mean():.1f}%")


if __name__ == "__main__":
    main()
