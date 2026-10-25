"""The study's EXTERNAL TEST: the LIDC-trained arms, unchanged, on LNDb.

Every arm was trained five times on LIDC-IDRI (one network per cross-validation
fold). Here each of those five networks scores every LNDb nodule, with no
retraining, no recalibration and no refitted constant: the conformal threshold is
the one fitted on that fold's LIDC validation set, the per-field scales are
LIDC's, and the input pipeline (size, normalisation, the deterministic test-time
decentring) is the LIDC test loader's.

    score   forward the five networks of each arm over LNDb and write one result
            file per (arm, network), in the run-folder format study_results reads
    report  the per-field primary on LNDb: each arm's Spearman with the panel's
            disagreement on each field, averaged over the fields and over the five
            networks, and ours minus each arm with a patient-clustered bootstrap
            whose resamples are SHARED by every network and every arm (they all
            score the same items, so independent draws per network would treat
            five correlated reads as five independent ones)

The main analysis uses the nodules with a binary panel label (median rating not
3), as the LIDC test folds do; `include_ambiguous=True` adds the median-3 nodules,
which carry a disagreement profile but no label.

    from backend.notebooks.shared import study_external as EXT
    EXT.score(cfg, cfg.out, EXT.find("lndb_report"), device)
    REP = EXT.report(cfg.out)
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from field_disagreement import build_model, model_input_spec
from field_disagreement.config import ExperimentConfig
from field_disagreement.triage.ladder import TRAINED_ARMS, ships

from backend.domains.adapter import PatchDataset

from . import study_prior_art as X
from . import panel as PANEL
from .study_arms import fold_tag
from .study_results import (ARMS, CLAIM, FIELD_NAMES, RIVALS, field_rho, spearman)
from .evaluate import predict_full
from .study import BASE_SEED

N_BOOT = 2000
EXT_DIR = "external_{name}"


# --------------------------------------------------------------------------- #
# where the external cohort is
# --------------------------------------------------------------------------- #
def find(name: str = "lndb_report") -> Path:
    """The processed external cohort: paths.yaml first, then any attached Kaggle
    input or Drive folder holding `<name>/index.csv`."""
    try:
        from backend.config import paths
        p = Path(getattr(paths(), f"{name}_processed"))
        if (p / "index.csv").exists():
            return p
    except Exception:
        pass
    # Colab: `<name>.zip` in the project's data/processed on Drive, unpacked once
    for z in sorted(glob.glob(f"/content/drive/MyDrive/*/data/processed/{name}.zip")):
        dest = Path("/content/data") / name
        if not (dest / "index.csv").exists():
            import zipfile
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest.parent / f"_{name}")
            hit = next(Path(dest.parent / f"_{name}").rglob("index.csv")).parent
            hit.rename(dest)
        return dest
    # depth-limited: a recursive walk of a whole Drive mount can take many minutes
    for root in ("/kaggle/input", "/content/drive/MyDrive", "/content"):
        for depth in range(5):
            hits = glob.glob(os.path.join(root, *(["*"] * depth), name, "index.csv"))
            if hits:
                return Path(hits[0]).parent
    raise FileNotFoundError(
        f"no processed `{name}` found. Build it with `python -m backend.domains.lndb.prepare` "
        f"and attach the folder (keep it private: LNDb is CC BY-NC-ND).")


def _files(ext: Path, include_ambiguous: bool) -> list[str]:
    import pandas as pd
    idx = pd.read_csv(ext / "index.csv")
    if not include_ambiguous:
        idx = idx[idx.label.isin([0, 1])]
    return idx.file.tolist()


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _network(cfg, arm: str, n_fields: int):
    """The arm's architecture exactly as `study_arms.build_arm` builds it."""
    spec = TRAINED_ARMS[arm]
    width = (n_fields if spec.target == "d_profile" else
             n_fields * int(cfg.n_ratings) if spec.target == "opinion_fields" else spec.regress)
    if spec.aux_fields:
        width += n_fields * int(cfg.n_ratings)
    if spec.head == "evidential":
        width = 2
    model = build_model(cfg.model, pretrained=False,
                        opinion_head=spec.opinion != "none",
                        n_ratings=cfg.n_ratings, dispersion_head=width)
    return X.EvidentialNet(model) if spec.head == "evidential" else model


def _weights(run_dir: Path, cfg, arm: str, fold: int, seed: int) -> Path:
    return run_dir / "checkpoints" / f"{cfg.model}_{fold_tag(arm, fold, seed)}.best.pt"


def _lidc_result(run_dir: Path, cfg, arm: str, fold: int, seed: int) -> dict:
    p = run_dir / "checkpoints" / f"{cfg.model}_{fold_tag(arm, fold, seed)}.result.json"
    return json.loads(p.read_text()) if p.exists() else {}


def score(cfg, run_dir, ext_dir, device, arms=ARMS, folds=range(5), seed: int | None = None,
          name: str = "lndb", include_ambiguous: bool = False) -> Path:
    """Write `<run_dir>/external_<name>[_all]/checkpoints/<model>_<arm>_fold<k>.result.json`.
    An arm or fold without saved LIDC weights is skipped and listed."""
    run_dir, ext_dir = Path(run_dir), Path(ext_dir)
    seed = int(cfg.seed if seed is None else seed)
    out = run_dir / (EXT_DIR.format(name=name) + ("_all" if include_ambiguous else "")) / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    files = _files(ext_dir, include_ambiguous)
    size, norm = model_input_spec(cfg.model)
    data = ExperimentConfig.of(cfg).data
    ds = PatchDataset(ext_dir, files, train=False, img_size=size, normalize=norm,
                      shift_frac=data.shift_frac, shift_seed=data.shift_seed)
    loader = DataLoader(ds, batch_size=data.batch_size, shuffle=False,
                        num_workers=data.num_workers)
    first = np.load(ext_dir / files[0], allow_pickle=True)
    n_fields = int(np.asarray(first["d_profile"]).shape[0])
    profile = np.stack([np.load(ext_dir / f, allow_pickle=True)["d_profile"] for f in files])
    clusters = [str(np.load(ext_dir / f, allow_pickle=True)["patient_id"]) for f in files]
    missing = []
    for arm in arms:
        for k in folds:
            w = _weights(run_dir, cfg, arm, k, seed)
            if not w.exists():
                missing.append(w.name)
                continue
            if (out / f"{cfg.model}_{fold_tag(arm, k, seed)}.result.json").exists():
                continue                    # scored in an earlier session
            model = _network(cfg, arm, n_fields).to(device)
            model.load_state_dict(torch.load(w, map_location=device))
            model.eval()
            pred = predict_full(model, loader, device, cfg.mc_passes, cfg.seed)
            if TRAINED_ARMS[arm].head == "evidential" and pred.get("dispersion") is not None:
                pred["epistemic"] = np.asarray(pred.pop("dispersion"), float).ravel()
            if TRAINED_ARMS[arm].target == "opinion_fields" and pred.get("dispersion") is not None:
                from .study_arms import _field_spreads
                pred["dispersion"] = _field_spreads(pred["dispersion"], cfg.n_ratings)
            from .study_arms import _drop_aux
            _drop_aux(pred, arm, cfg)
            qhat = _lidc_result(run_dir, cfg, arm, k, seed).get("conf_qhat")
            p = np.asarray(pred["probs"], float).ravel()
            y = np.asarray(pred["labels"], float).ravel() if pred.get("labels") is not None \
                else np.array([np.load(ext_dir / f)["label"] for f in files], float)
            res = {"arm": arm, "lidc_fold": k, "seed": seed, "cohort": name,
                   "n_items": len(files), "conf_qhat_from_lidc": qhat,
                   "test_probs": p.tolist(), "test_labels": y.tolist(),
                   "test_disagreement": np.asarray(pred["disagreement"], float).ravel().tolist(),
                   "test_clusters": clusters, "test_d_profile": profile.astype(float).tolist()}
            if pred.get("dispersion") is not None and np.ndim(pred["dispersion"]) == 2:
                res["test_panel_disp_fields"] = np.asarray(pred["dispersion"], float).tolist()
            for sig, v in PANEL.panel_signals(pred, q_hat=qhat).items():
                res[f"test_{sig}"] = np.asarray(v, float).ravel().tolist()
            (out / f"{cfg.model}_{fold_tag(arm, k, seed)}.result.json").write_text(json.dumps(res))
            print(f"  {arm} LIDC fold {k}: {len(files)} {name} nodules scored", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if missing:
        print(f"  no LIDC weights for {len(missing)} networks (skipped): {missing[:6]}"
              + (" ..." if len(missing) > 6 else ""))
    return out.parent


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def load(ext_run) -> dict:
    """{arm: {(seed, lidc_fold): result}} from an external run folder: every
    LIDC network of every training seed run so far is one network of the arm."""
    out: dict = {}
    for f in sorted(Path(ext_run, "checkpoints").glob("*.result.json")):
        r = json.loads(f.read_text())
        out.setdefault(r["arm"], {})[(int(r.get("seed", BASE_SEED)), int(r["lidc_fold"]))] = r
    return out


def _fields(r: dict, arm: str, P: int) -> np.ndarray:
    if r.get("test_panel_disp_fields") is not None:
        return np.asarray(r["test_panel_disp_fields"], float)
    return np.repeat(np.asarray(r[f"test_{ships(arm)}"], float)[:, None], P, axis=1)


def _auc(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    m = np.isin(y, (0, 1))
    y, p = y[m], p[m]
    if y.min() == y.max():
        return float("nan")
    r = np.argsort(np.argsort(p)) + 1.0
    n1 = (y == 1).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def report(run_dir_or_ext, claim: str = CLAIM, rivals=RIVALS, n_boot: int = N_BOOT,
           seed: int = 0, name: str = "lndb", verbose: bool = True) -> dict:
    """The per-field primary on the external cohort, pooled over the five LIDC
    networks of each arm, with shared patient-clustered bootstrap draws."""
    ext = Path(run_dir_or_ext)
    if not ext.name.startswith("external_"):              # a run folder: its external part
        ext = ext / EXT_DIR.format(name=name)
    res = load(ext)
    if claim not in res:
        raise FileNotFoundError(f"no {claim} results under {ext}")
    nets = sorted(res[claim])
    ref = res[claim][nets[0]]
    D = np.asarray(ref["test_d_profile"], float)
    N, P = D.shape
    cl = np.asarray(ref["test_clusters"])
    groups = [np.flatnonzero(cl == c) for c in np.unique(cl)]
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
             for _ in range(n_boot)]
    arms = [claim] + [a for a in rivals if a in res]

    def rho_fields(arm, k, b=None):
        S = _fields(res[arm][k], arm, P)
        return field_rho(S if b is None else S[b], D if b is None else D[b])

    point, per_field, per_net, boot = {}, {}, {}, {}
    for a in arms:
        ks = [k for k in nets if k in res[a]]
        pf = np.mean([rho_fields(a, k) for k in ks], axis=0)
        per_field[a] = pf
        point[a] = float(pf.mean())
        per_net[a] = {k: float(rho_fields(a, k).mean()) for k in ks}
        boot[a] = np.array([np.mean([rho_fields(a, k, b).mean() for k in ks]) for b in boots])
    contrast = {}
    for a in arms[1:]:
        d = boot[claim] - boot[a]
        lo, hi = np.percentile(d, [2.5, 97.5])
        won = sum(per_net[claim][k] > per_net[a][k] for k in per_net[a] if k in per_net[claim])
        contrast[a] = {"delta": point[claim] - point[a], "lo": float(lo), "hi": float(hi),
                       "won": int(won), "n_nets": len(per_net[a])}
    y = np.asarray(ref["test_labels"], float)
    auc = {a: float(np.mean([_auc(y, res[a][k]["test_probs"]) for k in res[a]])) for a in arms}
    report_level = {a: float(np.mean([spearman(np.asarray(res[a][k][f"test_{ships(a)}"]),
                                               np.asarray(res[a][k]["test_disagreement"]))
                                      for k in res[a]])) for a in arms}
    out = {"cohort": name, "n_items": N, "n_patients": len(groups), "n_fields": P,
           "nets": nets, "rho": point, "per_field": per_field, "per_net": per_net,
           "contrast": contrast, "auc": auc, "report_level": report_level,
           "labels": {int(v): int((y == v).sum()) for v in np.unique(y)}}
    if verbose:
        print_report(out, claim)
    return out


def print_report(rep: dict, claim: str = CLAIM) -> None:
    names = FIELD_NAMES.get(rep["n_fields"], [f"f{j}" for j in range(rep["n_fields"])])
    print(f"\nEXTERNAL TEST on {rep['cohort'].upper()}: {rep['n_items']} nodules, "
          f"{rep['n_patients']} patients, labels {rep['labels']}; each arm's five LIDC "
          f"networks, unchanged")
    print(f"  {'arm':8s}{'per-field rho':>14s}{'ours minus it [95% CI]':>30s}"
          f"{'nets won':>10s}{'report rho':>12s}{'AUC':>8s}")
    for a, r in rep["rho"].items():
        c = rep["contrast"].get(a)
        ci = (f"{c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]" if c else "")
        won = f"{c['won']}/{c['n_nets']}" if c else ""
        print(f"  {a:8s}{r:14.3f}{ci:>30s}{won:>10s}"
              f"{rep['report_level'][a]:12.3f}{rep['auc'][a]:8.3f}")
    print("  per field (ours):  " + ", ".join(
        f"{n} {v:.3f}" for n, v in zip(names, rep["per_field"][claim])))
