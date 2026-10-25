"""Per-field rating histograms: the target of the per-field opinion-head arm (P_emd).

    python -m backend.domains.field_opinions --cohort lidc     # lidc_report, via pylidc
    python -m backend.domains.field_opinions --cohort chaksu   # chaksu_report, release CSVs
    python -m backend.domains.field_opinions --cohort lndb     # lndb_report, release CSVs

Adds `opinion_fields` (P, K) float32 to every .npz in place: for each report field,
the share of the panel's readers at each of K levels. Nothing else is touched, so
every existing arm reads the same samples as before.

Levels, decided from the readers' values alone, before any run:
  * LIDC-IDRI / LNDb: the ordinal ratings 1..5 (K = 5), as the malignancy histogram.
  * Chákṣu ratios (ACDR, VCDR, HCDR): the 0.1 grid mapped to levels 1..10, exactly
    as the diagnostic-field (VCDR) histogram of the prior-art arms.
  * Chákṣu sizes: each expert's value divided by the panel's mean (the sizes are in
    pixels of three cameras, which is why their dispersion is a coefficient of
    variation), cut at the deciles of all such relative values in the cohort into
    K = 10 levels. The histogram's spread then tracks the field's CV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend.config import paths

K_LIDC, K_CHAKSU = 5, 10


def _hist(levels: np.ndarray, k: int) -> np.ndarray:
    """(R, P) integer levels 0..k-1 -> (P, k) shares."""
    out = np.zeros((levels.shape[1], k), np.float32)
    for j in range(levels.shape[1]):
        out[j] = np.bincount(levels[:, j], minlength=k)[:k] / levels.shape[0]
    return out


def lidc_ratings(annotation_ids) -> np.ndarray:
    """(R, 7) per-reader ratings in LIDC field order, from pylidc's bundled database."""
    import pylidc as pl
    from backend.domains.report_condition import LIDC_ATTRS
    anns = pl.query(pl.Annotation).filter(
        pl.Annotation.id.in_([int(a) for a in annotation_ids])).all()
    if len(anns) != len(annotation_ids):
        raise ValueError(f"pylidc returned {len(anns)} of {len(annotation_ids)} annotations")
    return np.array([[getattr(a, f) for f in LIDC_ATTRS] for a in anns], int)


def build(cohort: str) -> dict:
    P = paths()
    if cohort == "lidc":
        folder = Path(P.lidc_report_processed)
    elif cohort == "chaksu":
        folder = Path(P.chaksu_report_processed)
    elif cohort == "lndb":
        folder = Path(P.lndb_report_processed)
    else:
        raise ValueError(cohort)
    files = sorted(folder.glob("*.npz"))

    if cohort == "chaksu":
        from backend.domains.report_condition import (CHAKSU_ATTRS, CHAKSU_RATIOS,
                                                      chaksu_fields)
        table = chaksu_fields(Path(P.chaksu_raw))
        vals = {}
        for f in files:
            uid = str(np.load(f, allow_pickle=True)["series_uid"])
            m = np.full((len(table[uid]), len(CHAKSU_ATTRS)), np.nan)
            for r, row in enumerate(table[uid].values()):
                for j, a in enumerate(CHAKSU_ATTRS):
                    try:
                        x = float(row.get(a, ""))
                    except ValueError:
                        continue
                    if np.isfinite(x) and x > 0:
                        m[r, j] = x
            vals[f.name] = m
        n_ratio = len(CHAKSU_RATIOS)
        rel = np.concatenate([(m[:, n_ratio:] / np.nanmean(m[:, n_ratio:], 0))
                              for m in vals.values()])
        cuts = [np.nanquantile(rel[:, j], np.linspace(0.1, 0.9, 9))
                for j in range(rel.shape[1])]
    elif cohort == "lndb":
        from backend.domains.lndb.prepare import panels
        pan = panels(Path(P.lndb_raw) / "csv")
        lndb = {f"LNDb-{r.lndb:04d}_n{r.finding:02d}.npz": r.ratings for r in pan.itertuples()}

    n = 0
    for f in files:
        z = dict(np.load(f, allow_pickle=True))
        if cohort == "lidc":
            hist = _hist(lidc_ratings(z["annotation_ids"]) - 1, K_LIDC)
        elif cohort == "lndb":
            hist = _hist(np.asarray(lndb[f.name], int) - 1, K_LIDC)
        else:
            # each field over the experts who RECORDED it (a blank is not a vote)
            m = vals[f.name]
            hist = np.zeros((m.shape[1], K_CHAKSU), np.float32)
            for j in range(m.shape[1]):
                col = m[np.isfinite(m[:, j]), j]
                if not len(col):
                    hist[j] = 1.0 / K_CHAKSU
                    continue
                if j < n_ratio:
                    lev = np.clip(np.round(col * 10).astype(int), 1, 10) - 1
                else:
                    lev = np.searchsorted(cuts[j - n_ratio], col / np.mean(col))
                hist[j] = np.bincount(lev, minlength=K_CHAKSU)[:K_CHAKSU] / len(col)
        z["opinion_fields"] = hist
        np.savez_compressed(f, **z)
        n += 1
    return {"cohort": cohort, "items": n, "shape": list(hist.shape)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cohort", required=True, choices=("lidc", "chaksu", "lndb"))
    print(build(ap.parse_args().cohort))


if __name__ == "__main__":
    main()
