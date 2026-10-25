"""What a study run covers.

    run_scope             which folds this run covers (smoke / probe / n_folds)
    screen_arms           which of the declared arms a run may train

Both decide WHAT runs, from the config alone, so the scope of a run is in its own
config record rather than in an edited notebook cell. Nothing here reads a
notebook global.
"""
from __future__ import annotations

from types import SimpleNamespace

__all__ = ["run_scope", "fold_count_note", "screen_arms", "SMOKE_SCHEDULE", "BASE_SEED"]

# The training seed whose cache tags and result files carry no `_s<seed>` suffix.
BASE_SEED = 42


# --------------------------------------------------------------------------- #
# what runs
# --------------------------------------------------------------------------- #
SMOKE_SCHEDULE = {"epochs": 3, "warmup": 1}


def run_scope(cfg) -> SimpleNamespace:
    """The folds this run covers, from the config alone.

      smoke: true    a short sanity run on fold 0 (edits `cfg`'s schedule)
      probe: true    one fold, `probe_fold` (default 0) - a SCREEN
      otherwise      the first `n_folds` (default: every fold of the frozen
                     split) - so a 3-fold read is a strict prefix of the 5-fold one

    The smoke schedule is SMOKE_SCHEDULE, with any flat `smoke_overrides:`
    mapping in the config on top.

    Config-driven so the scope of a run is in its own config record rather than
    in an edited notebook cell. Returns (folds, probe, smoke, n_folds, note);
    `note` is the warning to print when the run is too small for the fold-level
    paired test, else ''.
    """
    probe = bool(getattr(cfg, "probe", False))
    smoke = bool(getattr(cfg, "smoke", False))
    n_folds = int(getattr(cfg, "n_folds", None) or getattr(cfg, "n_splits", 5))
    if smoke:
        over = {**SMOKE_SCHEDULE, **dict(getattr(cfg, "smoke_overrides", None) or {})}
        for key, value in over.items():
            setattr(cfg, key, value)
        folds = [0]
    elif probe:
        folds = [int(getattr(cfg, "probe_fold", 0))]
    else:
        folds = list(range(n_folds))
    return SimpleNamespace(folds=folds, probe=probe, smoke=smoke,
                           n_folds=n_folds, note=fold_count_note(folds))


def fold_count_note(folds) -> str:
    """The warning a run with fewer than 3 folds must print, or ''.

    With n < 3 a fold-level paired interval is undefined (NaN), and
    a table of NaN CIs is exactly what gets quoted without the caveat."""
    if len(folds) >= 3:
        return ""
    return "\n".join([
        f"\n  !! {len(folds)}-FOLD RUN: the fold-level paired table will show nan CIs.",
        "     Read the SUBJECT-CLUSTERED BOOTSTRAP instead (it pools per-item",
        "     scores and resamples subjects). Treat the result as a DIRECTION:",
        "     one fold cannot separate a small effect from seed-to-seed noise."])


def screen_arms(cfg, declared, optional=()) -> list:
    """The arms this run trains: `cfg.screen_arms`, or every declared arm.

    A screen may only ever be a SUBSET of the declared study, so an arm the
    study did not declare cannot enter through it. `optional` arms (the
    ablations) are declared too, but trained only when named."""
    declared, optional = list(declared), list(optional)
    screen = list(getattr(cfg, "screen_arms", None) or declared)
    unknown = [m for m in screen if m not in declared + optional]
    if unknown:
        raise ValueError(
            f"screen_arms {unknown} are not declared arms of this study "
            f"(declared = {declared + optional}); a screen must be a SUBSET of the study")
    extra = [m for m in screen if m in optional]
    if extra:
        print(f"[screen] ablation arms {extra}: reported beside the study, not in its tables")
    if [m for m in screen if m in declared] != declared:
        print(f"[screen] {len(screen) - len(extra)} of {len(declared)} study arms: "
              f"{[m for m in screen if m in declared]}")
        print("[screen] a SUBSET of the study's arms - report it as the ladder only "
              "together with the study's other arms")
    return screen

