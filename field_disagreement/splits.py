from __future__ import annotations
import hashlib
import json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd


"""Subject-disjoint cross-validation: who is in which fold, and nothing else.

Pure numpy / pandas: NO torch, so it can be unit-tested without a GPU stack.

Protocol:
  * split BY SUBJECT, never by item — items from one subject are correlated, so
    a subject appearing in both train and test leaks;
  * `n_splits` folds, no subject in two of them;
  * per fold: train / val / test over the binary labels only;
  * AMBIGUOUS items (the panel declined to call them — `spectrum.median_label`'s
    code) are NEVER a binary training target. They are held out and evaluated
    separately, attached to their subject's TEST fold so the ambiguous-case
    evaluation stays subject-disjoint from training.

The assignment greedily balances the negative / positive / ambiguous counts
across folds while keeping every subject's items together.

The labels are NEGATIVE / POSITIVE, not a domain's own class names: which
condition "positive" denotes is the domain's fact, and naming it here would make
the fold arithmetic read as though it only applied to one cohort. `BENIGN` and
`MALIGNANT` were what these were called while this was a single-cohort project.

The subject column is `subject_id`, and `patient_id` is accepted as its
historical name so a manifest cached by an earlier run still loads. The manifest
BUILDER, by contrast, is domain work: `build_manifest` reads a specific .npz
field layout and lives here only for the cohorts that already use it — a domain
with its own storage supplies the DataFrame directly and everything below works
on it unchanged.
"""

NEGATIVE, POSITIVE, AMBIGUOUS = 0, 1, -1
BINARY_LABELS = (NEGATIVE, POSITIVE)

SUBJECT_COLS = ("subject_id", "patient_id")


def subject_column(df: pd.DataFrame) -> str:
    """Which column identifies the subject — the canonical name, or the
    historical one a cached manifest may still carry."""
    for name in SUBJECT_COLS:
        if name in df.columns:
            return name
    raise KeyError(
        f"no subject column: expected one of {list(SUBJECT_COLS)}, "
        f"got {list(df.columns)}")


def subjects(df: pd.DataFrame):
    """The subject of every row, as strings."""
    return df[subject_column(df)].astype(str)


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def build_manifest(processed_dir, rebuild: bool = False) -> pd.DataFrame:
    """Scan every .npz in `processed_dir` and return a manifest
    (file, patient_id, label, n_readers). Cached to manifest.csv.

    The one function here that assumes a STORAGE LAYOUT — a directory of .npz
    files carrying those field names — which is why the field names keep their
    original spelling: they are what is written on disk, and renaming them here
    would not rename them there. It is a convenience for the cohorts already in
    that format, not part of the split contract.

    Everything else in this module takes the DataFrame, so a domain storing its
    cohort any other way builds the frame itself (columns: `file`, `subject_id`,
    `label`) and never calls this."""
    processed_dir = Path(processed_dir)
    cache = processed_dir / "manifest.csv"
    if cache.exists() and not rebuild:
        return pd.read_csv(cache)

    rows = []
    for f in sorted(processed_dir.glob("*.npz")):
        with np.load(f, allow_pickle=True) as d:
            keys = d.files
            n_readers = (int(d["n_readers"]) if "n_readers" in keys
                         else int(len(d["malignancy"])))
            rows.append({
                "file": f.name,
                "patient_id": str(d["patient_id"]),
                "label": int(d["label"]),
                "n_readers": n_readers,
            })
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


# --------------------------------------------------------------------------- #
# patient-level fold assignment
# --------------------------------------------------------------------------- #
def stratified_patient_kfold(df: pd.DataFrame, n_splits: int = 5,
                             seed: int = 42) -> dict[str, int]:
    """Assign each patient to exactly one fold in [0, n_splits), greedily
    balancing benign, malignant AND score-3 nodule counts (and patient counts)
    across folds. Every patient's nodules stay together. Returns
    {patient_id: fold}."""
    # per-subject class counts over the WHOLE manifest
    stat = (df.assign(
                subject=subjects(df),
                is_neg=(df.label == NEGATIVE).astype(int),
                is_pos=(df.label == POSITIVE).astype(int),
                is_amb=(df.label == AMBIGUOUS).astype(int))
              .groupby("subject")[["is_neg", "is_pos", "is_amb"]].sum())

    # place harder-to-balance subjects first (more items), shuffle ties by seed
    stat["bm"] = stat.is_neg + stat.is_pos
    order = stat.sample(frac=1.0, random_state=int(seed)).sort_values(
        ["bm", "is_amb", "is_pos"], ascending=False, kind="mergesort").index.tolist()

    load = np.zeros((n_splits, 3), dtype=np.int64)   # [neg, pos, amb] per fold
    pcount = np.zeros(n_splits, dtype=np.int64)      # subjects per fold
    assign: dict[str, int] = {}
    for pid in order:
        add = np.array([stat.at[pid, "is_neg"],
                        stat.at[pid, "is_pos"],
                        stat.at[pid, "is_amb"]])
        best_k, best_cost = 0, None
        for k in range(n_splits):
            tl = load.copy();  tl[k] += add
            pc = pcount.copy(); pc[k] += 1
            bm = tl[:, 0] + tl[:, 1]
            cost = ((bm.max() - bm.min())                    # binary-total balance
                    + (tl[:, 1].max() - tl[:, 1].min())      # positive balance
                    + (tl[:, 2].max() - tl[:, 2].min())      # ambiguous balance
                    + (pc.max() - pc.min()))                 # subject-count balance
            if best_cost is None or cost < best_cost:
                best_cost, best_k = cost, k
        assign[pid] = best_k
        load[best_k] += add
        pcount[best_k] += 1
    return assign


# --------------------------------------------------------------------------- #
# freezing the assignment
# --------------------------------------------------------------------------- #
def manifest_fingerprint(df: pd.DataFrame) -> str:
    """Short digest of everything `stratified_patient_kfold` actually reads: the
    per-patient class counts. Two manifests with the same fingerprint produce the
    same folds, whatever order the files were scanned in."""
    stat = (df.assign(subject=subjects(df),
                      b=(df.label == NEGATIVE).astype(int),
                      m=(df.label == POSITIVE).astype(int),
                      a=(df.label == AMBIGUOUS).astype(int))
              .groupby("subject")[["b", "m", "a"]].sum().sort_index())
    payload = "\n".join(f"{pid},{r.b},{r.m},{r.a}" for pid, r in stat.iterrows())
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_fold_assignment(path, patient_fold: dict[str, int], df: pd.DataFrame,
                         seed: int, n_splits: int) -> dict:
    """Write the assignment to JSON so it is a committed artefact rather than
    something recomputed at import time. The fingerprint of the manifest it was
    derived from travels with it, so `load_fold_assignment` can refuse to pair a
    frozen split with data it was not built for."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "created": date.today().isoformat(),
        "seed": int(seed),
        "n_splits": int(n_splits),
        "n_patients": len(patient_fold),
        "n_items": int(len(df)),
        "manifest_fingerprint": manifest_fingerprint(df),
        "assignment": {str(k): int(v) for k, v in sorted(patient_fold.items())},
    }
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return blob


def load_fold_assignment(path, df: pd.DataFrame | None = None,
                         strict: bool = True) -> dict[str, int]:
    """Read a frozen assignment. When `df` is given the manifest is checked
    against the fingerprint stored in the file: a mismatch means the data
    changed under a split that is supposed to be fixed, which silently
    invalidates every result compared across it, so it raises.

    `strict=False` downgrades the fingerprint check to a warning but still
    requires every subject in `df` to have an entry — use it only for a
    deliberate subset (a smoke run), never for a reported number."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    assign = {str(k): int(v) for k, v in blob["assignment"].items()}
    if df is None:
        return assign

    missing = sorted(set(subjects(df)) - set(assign))
    if missing:
        raise KeyError(
            f"{len(missing)} subject(s) are not in the frozen split {path} "
            f"(first few: {missing[:5]}). Re-freeze the folds if the dataset grew.")

    got = manifest_fingerprint(df)
    want = blob.get("manifest_fingerprint")
    if want and got != want:
        msg = (f"manifest fingerprint {got} != {want} recorded in {path}: the data "
               f"changed since the folds were frozen, so results across this split "
               f"are no longer comparable. Re-run preprocessing to restore it, or "
               f"re-freeze deliberately.")
        if strict:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg, stacklevel=2)
    return assign


def make_fold(df: pd.DataFrame, patient_fold: dict[str, int], fold_id: int,
              n_splits: int = 5) -> dict[str, list[str]]:
    """Rotation scheme for one CV fold:
        test = fold_id ; val = (fold_id + 1) % n_splits ; train = the rest.

    Returns file lists for train/val/test (binary only) and `ambiguous` (the
    TEST subjects' items the panel declined to call). `score3` is the same list
    under the name this project's notebooks already read it by — one list, two
    keys, so a caller cannot pick up a stale second copy."""
    test_f = fold_id % n_splits
    val_f = (fold_id + 1) % n_splits
    train_f = [k for k in range(n_splits) if k not in (test_f, val_f)]

    fold_of = subjects(df).map(patient_fold)

    def files(folds, labels):
        m = fold_of.isin(folds) & df["label"].isin(labels)
        return df.loc[m, "file"].tolist()

    ambiguous = files([test_f], [AMBIGUOUS])
    return {
        "train":  files(train_f, BINARY_LABELS),
        "val":    files([val_f],  BINARY_LABELS),
        "test":   files([test_f], BINARY_LABELS),
        "ambiguous": ambiguous,
        "score3": ambiguous,
    }


def assert_no_leakage(df: pd.DataFrame, splits: dict[str, list[str]]) -> None:
    """Raise if any subject appears in more than one of train/val/test, or if an
    ambiguous-eval subject is not among the test subjects."""
    pf = df.assign(subject=subjects(df)).set_index("file")["subject"]
    pat = {k: set(pf[f] for f in v) for k, v in splits.items()}
    assert pat["train"].isdisjoint(pat["val"]),  "subject leak: train/val"
    assert pat["train"].isdisjoint(pat["test"]), "subject leak: train/test"
    assert pat["val"].isdisjoint(pat["test"]),   "subject leak: val/test"
    # Ambiguous items are only ever evaluated, never trained: their subjects must
    # not appear in train or val (they ride with the held-out test fold).
    assert pat["ambiguous"].isdisjoint(pat["train"]), "ambiguous leak into train"
    assert pat["ambiguous"].isdisjoint(pat["val"]),   "ambiguous leak into val"


def fold_summary(df: pd.DataFrame, patient_fold: dict[str, int],
                 n_splits: int = 5) -> pd.DataFrame:
    """Per-fold subject / negative / positive / ambiguous counts (sanity table).

    The legacy column names ride along so the existing report tables keep
    resolving; they are the same numbers under their single-cohort names."""
    subj = subjects(df)
    fold_of = subj.map(patient_fold)
    rows = []
    for k in range(n_splits):
        m = fold_of == k
        sub = df[m]
        n_neg = int((sub.label == NEGATIVE).sum())
        n_pos = int((sub.label == POSITIVE).sum())
        n_amb = int((sub.label == AMBIGUOUS).sum())
        rows.append({
            "fold": k,
            "subjects": subj[m].nunique(),
            "negative": n_neg, "positive": n_pos, "ambiguous": n_amb,
            "patients": subj[m].nunique(),
            "benign": n_neg, "malignant": n_pos, "score3": n_amb,
        })
    return pd.DataFrame(rows)
