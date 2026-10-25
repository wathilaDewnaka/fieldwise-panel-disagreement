"""Figures for walking through the study: the data, and what each arm learned.

Every function draws one matplotlib figure and returns it, so a notebook cell is
one call. Nothing here trains, scores or changes a result: the data figures read
the processed cohort, and the result figures read a finished run folder.

    panel_gallery        images with the panel's per-field disagreement
    case_gallery      test cases: the panel's disagreement against each arm's
    referral_curve    share of contested cases each arm sends to a reader
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
ARM_COLOURS = ("#8a8984", "#eb6834", "#1baf7a", "#7a5cc4", "#c2410c", "#0e7490",
               "#a16207", "#2a78d6", "#be185d")


def _plt():
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def to_display(image, normalized: bool = False) -> np.ndarray:
    """(3, H, W) array or tensor -> (H, W, 3) in [0, 1] for imshow.

    A CT patch is three neighbouring slices, so its middle channel is shown as a
    grey image; a photograph is shown in colour."""
    x = np.asarray(image.detach().cpu() if hasattr(image, "detach") else image, float)
    if normalized:
        x = x * _STD + _MEAN
    x = np.clip(x, 0.0, 1.0)
    if x.ndim == 2:
        return np.repeat(x[..., None], 3, axis=-1)
    if np.allclose(x[0], x[2], atol=0.08) and np.allclose(x[0], x[1], atol=0.08):
        return np.repeat(x[1][..., None], 3, axis=-1)
    return np.moveaxis(x, 0, -1)


def is_ct(image) -> bool:
    """True for a three-slice CT patch, False for a colour photograph. Neighbouring
    slices differ pixel by pixel but have the same overall brightness, while a
    photograph's colour channels do not (a fundus image is far redder than blue)."""
    x = np.asarray(image.detach().cpu() if hasattr(image, "detach") else image, float)
    means = x.reshape(x.shape[0], -1).mean(axis=1)
    return bool(np.ptp(means) < 0.05 * max(float(np.abs(means).max()), 1e-6) + 0.02)


def ct_slice(image, normalized: bool = False) -> np.ndarray:
    """The centre slice of a CT patch as a grey (H, W, 3) image."""
    x = np.asarray(image.detach().cpu() if hasattr(image, "detach") else image, float)
    if normalized:
        x = x * _STD + _MEAN
    s = np.clip(x[1], 0.0, 1.0)
    return np.repeat(s[..., None], 3, axis=-1)


def show_image(ax, image, normalized: bool = False) -> None:
    raw = np.asarray(image.detach().cpu() if hasattr(image, "detach") else image, float)
    if normalized:
        raw = raw * _STD + _MEAN
    rgb = ct_slice(raw) if is_ct(raw) else to_display(raw)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _load(processed_dir, name):
    return np.load(Path(processed_dir) / name, allow_pickle=True)


def _files(processed_dir) -> list[str]:
    return sorted(p.name for p in Path(processed_dir).glob("*.npz"))


def pick_by_disagreement(processed_dir, n: int, files=None, seed: int = 0) -> list[str]:
    """`n` files spread over the panel's disagreement: the most and least disputed
    and, between them, evenly spaced quantiles - so a gallery shows the range."""
    files = list(files) if files is not None else _files(processed_dir)
    d = np.array([float(_load(processed_dir, f)["disagreement"]) for f in files])
    order = np.argsort(d, kind="stable")
    pos = np.unique(np.linspace(0, len(order) - 1, n).round().astype(int))
    return [files[order[i]] for i in pos]


# --------------------------------------------------------------------------- #
# the data
# --------------------------------------------------------------------------- #
def describe_cohort(df, processed_dir, field_names=()) -> None:
    """Print what one cohort contains and what a single processed item holds."""
    labels = df["label"].astype(int)
    print(f"items {len(df)}  |  patients {df['patient_id'].nunique()}  |  "
          f"positive {int((labels == 1).sum())}  negative {int((labels == 0).sum())}"
          + (f"  undecided {int((labels < 0).sum())}" if (labels < 0).any() else ""))
    if "n_readers" in df:
        counts = df["n_readers"].value_counts().sort_index()
        print("readers per item: " + ", ".join(f"{int(k)} readers: {int(v)}"
                                               for k, v in counts.items()))
    if "device" in df:
        print("camera: " + ", ".join(f"{k} {v}" for k, v in df["device"].value_counts().items()))
    if len(field_names):
        print(f"report fields ({len(field_names)}): " + ", ".join(field_names))
    z = _load(processed_dir, str(df["file"].iloc[0]))
    print(f"\none item ({df['file'].iloc[0]}):")
    for key in z.files:
        v = z[key]
        shown = (f"{v.dtype} {tuple(v.shape)}" if v.ndim else f"{v.dtype} = {v.item()!r}")
        print(f"  {key:22s} {shown}")


def panel_gallery(processed_dir, files, field_names, title: str = ""):
    """One row per item: the image, and the panel's disagreement on every report
    field (in units of that field's cohort spread)."""
    plt = _plt()
    fig, axes = plt.subplots(len(files), 2, figsize=(7.2, 1.9 * len(files)),
                             gridspec_kw={"width_ratios": [1, 3]}, squeeze=False)
    for i, f in enumerate(files):
        z = _load(processed_dir, f)
        show_image(axes[i, 0], z["image"])
        axes[i, 0].set_title(f"label {int(z['label'])}, report-level d = "
                             f"{float(z['disagreement']):.2f}", fontsize=8)
        prof = np.asarray(z["d_profile"], float)
        axes[i, 1].bar(np.arange(len(prof)), prof, color="#2a78d6")
        axes[i, 1].set_xticks(np.arange(len(prof)))
        axes[i, 1].set_xticklabels(field_names if i == len(files) - 1 else [],
                                   rotation=35, ha="right", fontsize=7)
        axes[i, 1].set_ylim(0, max(3.0, prof.max() * 1.1))
        axes[i, 1].set_ylabel("disagreement", fontsize=7)
        axes[i, 1].grid(axis="y", color="#e4e3df", lw=0.5)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# trained arms, read from the run folder
# --------------------------------------------------------------------------- #
def _col(result: dict, key: str) -> np.ndarray:
    return np.asarray(result[f"test_{key}"], float)


def _pct(x) -> np.ndarray:
    """Within-fold percentile rank in [0, 1] (ties share their mean rank)."""
    from scipy.stats import rankdata
    x = np.asarray(x, float)
    return (rankdata(x, axis=0) - 0.5) / x.shape[0]


def case_gallery(fold_result: dict, processed_dir, field_names, arms: dict,
                    signal_of, n_cases: int, title: str = ""):
    """The `n_cases` test cases the panel disputed most, from one fold.

    Left: the image. Right: for each report field, where the case sits among the
    fold's test cases (percentile) - by the panel's real disagreement, and by
    each per-field arm's prediction. `arms` maps a display label to an arm id;
    `signal_of(arm)` names the column the arm ranks with. A single-signal arm
    gives every field the same value."""
    plt = _plt()
    any_arm = next(iter(fold_result.values()))
    files = [Path(str(f)).name for f in any_arm["test_files"]]
    D = next(np.asarray(r["test_d_profile"], float) for r in fold_result.values()
             if r.get("test_d_profile") is not None)
    d = np.asarray(any_arm["test_disagreement"], float)
    pick = np.argsort(-d, kind="stable")[:n_cases]
    truth = _pct(D)
    pred = {}
    for label, arm in arms.items():
        r = fold_result[arm]
        S = (np.asarray(r["test_panel_disp_fields"], float)
             if r.get("test_panel_disp_fields") is not None
             else np.repeat(_col(r, signal_of(arm))[:, None], D.shape[1], 1))
        pred[label] = _pct(S)
    fig, axes = plt.subplots(n_cases, 2, figsize=(9.0, 2.1 * n_cases),
                             gridspec_kw={"width_ratios": [1, 4]}, squeeze=False)
    w = 0.8 / (1 + len(arms))
    xs = np.arange(D.shape[1])
    for i, k in enumerate(pick):
        z = _load(processed_dir, files[k])
        show_image(axes[i, 0], z["image"])
        axes[i, 0].set_title(f"label {int(any_arm['test_labels'][k])}, d = {d[k]:.2f}",
                             fontsize=8)
        ax = axes[i, 1]
        ax.bar(xs - 0.4 + w / 2, truth[k], w, color="#0b0b0b", label="panel (truth)")
        for j, (label, P) in enumerate(pred.items()):
            ax.bar(xs - 0.4 + w * (j + 1.5), P[k], w,
                   color=ARM_COLOURS[(j + 1) % len(ARM_COLOURS)], label=label)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(xs)
        ax.set_xticklabels(field_names if i == n_cases - 1 else [], rotation=35,
                           ha="right", fontsize=7)
        ax.set_ylabel("percentile", fontsize=7)
        ax.grid(axis="y", color="#e4e3df", lw=0.5)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=len(arms) + 1, loc="upper center",
               bbox_to_anchor=(0.5, 0.965), frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def referral_capture(res: dict, arm: str, signal_of, d0: float, budgets) -> np.ndarray:
    """Mean over folds of the share of contested cases (d > d0) that land in the
    top `budget` share of cases ranked by the arm's routing signal."""
    out = []
    for k in sorted(res):
        r = res[k].get(arm)
        if r is None:
            continue
        s, d = _col(r, signal_of(arm)), np.asarray(r["test_disagreement"], float)
        hot = d > d0
        order = np.argsort(-s, kind="stable")
        row = []
        for b in budgets:
            top = np.zeros(len(s), bool)
            top[order[:max(1, int(round(b * len(s))))]] = True
            row.append(float((top & hot).sum() / max(1, hot.sum())))
        out.append(row)
    return np.mean(out, axis=0)


def referral_curve(res: dict, arms: dict, signal_of, d0: float, budgets,
                      title: str = ""):
    """For a reading budget (share of all cases sent to a reader, ranked by each
    arm's routing signal), the share of the panel's contested cases it catches.
    The dashed diagonal is a random choice of the same size."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    budgets = np.asarray(budgets, float)
    for j, (label, arm) in enumerate(arms.items()):
        ax.plot(100 * budgets, 100 * referral_capture(res, arm, signal_of, d0, budgets),
                marker="o", ms=3, lw=1.6, color=ARM_COLOURS[j % len(ARM_COLOURS)],
                label=label)
    ax.plot([0, 100], [0, 100], ls="--", color="#8a8984", lw=0.8, label="random")
    ax.set_xlabel("cases sent to a reader (%)")
    ax.set_ylabel("contested cases caught (%)")
    ax.set_xlim(0, 100 * budgets.max())
    ax.set_ylim(0, 100)
    ax.grid(color="#e4e3df", lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
