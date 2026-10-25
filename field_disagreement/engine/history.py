"""What a trained unit's per-epoch `history` says about its kept checkpoint.

`ExperimentRunner.fit` writes one row per epoch (loss, val AUC, smoothed val AUC,
and any `val_monitor` numbers) and the caller stores it, with `selection`, in the
unit's result. This reads one such result back: which epoch was kept, where val
AUC peaked, how far selected val AUC sits above test AUC, and - when a monitor
logged a secondary signal - how much of that signal's best the kept epoch had."""
from __future__ import annotations


__all__ = ["selection_summary", "signal_learned_by"]


def selection_summary(result: dict, signal: str = "val_rho_fields",
                      test_key: str = "auc") -> dict:
    """{kept, peak, gap, signal_share} for one unit's result.

    kept          the epoch selection kept (None if not recorded)
    peak          the epoch of the best smoothed val AUC in `history` (None without it)
    gap           selected val AUC minus `test_key` (NaN if either is missing)
    signal_share  the logged `signal` at the kept epoch over its best anywhere in
                  the run (NaN without it): below 1 means the head was kept before
                  that signal finished improving"""
    sel = result.get("selection") or {}
    hist = result.get("history") or []
    kept = sel.get("epoch")
    peak = (max(hist, key=lambda h: h.get("val_auc_smooth", -1.0))["epoch"]
            if hist else None)
    try:
        gap = float(result["best_val_auc"]) - float(result[test_key])
    except (KeyError, TypeError, ValueError):
        gap = float("nan")
    vals = {h["epoch"]: h.get(signal) for h in hist if h.get(signal) is not None}
    best = max(vals.values()) if vals else float("nan")
    at = vals.get(kept)
    share = float(at / best) if at is not None and best > 0 else float("nan")
    return {"kept": kept, "peak": peak, "gap": gap, "signal_share": share}


def _smooth(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        w = [v for v in values[max(0, i - window + 1): i + 1] if v == v]
        out.append(sum(w) / len(w) if w else float("nan"))
    return out


def signal_learned_by(result: dict, share: float, window: int,
                      signal: str = "val_rho_fields") -> dict:
    """When the logged `signal` (smoothed over `window` epochs, as selection
    smooths val AUC) first reached `share` of its best, and what share it had at
    the kept epoch. {reached: epoch or None, kept_share: float}. Smoothing keeps
    one lucky epoch from setting the bar."""
    hist = [h for h in (result.get("history") or []) if h.get(signal) is not None]
    if not hist:
        return {"reached": None, "kept_share": float("nan")}
    ep = [h["epoch"] for h in hist]
    sm = _smooth([float(h[signal]) for h in hist], window)
    best = max(v for v in sm if v == v)
    if not best > 0:
        return {"reached": None, "kept_share": float("nan")}
    reached = next((e for e, v in zip(ep, sm) if v >= share * best), None)
    kept = (result.get("selection") or {}).get("epoch")
    at = dict(zip(ep, sm)).get(kept)
    return {"reached": reached, "kept_share": float(at / best) if at is not None else float("nan")}
