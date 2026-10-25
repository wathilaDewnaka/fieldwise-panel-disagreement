"""Illustrative cases: one lung nodule and one fundus image, with the panel's recorded
and Field-disp's predicted disagreement on every report field.

The case is chosen by a fixed rule, not by eye: among the contested test items of
seed 42 (report-level d above the cohort's contested cut d0), pooled over the five
test folds, the item whose predicted profile agrees with its recorded profile at
the MEDIAN Spearman correlation (the lower median when the count is even). Both
profiles are shown as within-fold percentiles, so every field is on one scale. The
two fields with the highest predicted disagreement are marked as the fields a
second reader would be pointed to.

    python fig_cases.py      (writes fig_cases.pdf; needs the stored predictions
                              and the processed report-level cohorts, see README)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "paper" / "analysis")]
import io, contextlib  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    import ms_subsets_w as MS  # noqa: E402
R = MS.R

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY = "#2a78d6", "#b8b6b0"
FIELDS = {"LIDC": ["Subtlety", "Sphericity", "Margin", "Lobulation", "Spiculation",
                   "Texture", "Malignancy"],
          "Chaksu": ["ACDR", "VCDR", "HCDR", "Disc area", "Cup area", "Rim area",
                     "Disc height", "Disc width", "Cup height", "Cup width"]}
TITLE = {"LIDC": "LIDC-IDRI nodule", "Chaksu": "Chákṣu fundus image"}


def pct(x):
    return np.column_stack([(rankdata(x[:, j]) - 0.5) / len(x) for j in range(x.shape[1])])


def pick(coh):
    res = R.load_run(MS.COH[coh][42], 42)
    d0 = R.cohort_d0(MS.COH[coh][42])
    rows = []
    for k in sorted(res):
        r = res[k]["H_fld"]
        D = np.asarray(r["test_d_profile"], float)
        S = np.asarray(r["test_panel_disp_fields"], float)
        d = np.asarray(r["test_disagreement"], float)
        Dp, Sp = pct(D), pct(S)
        for i in np.flatnonzero(d > d0):
            rho = spearmanr(Sp[i], Dp[i]).correlation
            rows.append((rho, r["test_files"][i], Dp[i], Sp[i]))
    rows.sort(key=lambda t: t[0])
    rho, f, obs, pred = rows[(len(rows) - 1) // 2]
    return f, obs, pred, rho, len(rows)


fig = plt.figure(figsize=(6.84, 5.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.3], height_ratios=[7, 10],
                      wspace=0.42, hspace=0.45)
for c, coh in enumerate(("LIDC", "Chaksu")):
    f, obs, pred, rho, n = pick(coh)
    z = np.load(ROOT / MS.PROC[coh] / f, allow_pickle=True)
    img = z["image"]
    ax = fig.add_subplot(gs[c, 0])
    if coh == "LIDC":
        ax.imshow(img[1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        ax.imshow(np.clip(np.transpose(img, (1, 2, 0)), 0, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(TITLE[coh], fontsize=8.5, loc="left")
    ax = fig.add_subplot(gs[c, 1])
    names = FIELDS[coh]
    y = np.arange(len(names))[::-1]
    ax.barh(y + 0.19, obs, height=0.36, color=GREY, label="Recorded (panel)")
    ax.barh(y - 0.19, pred, height=0.36, color=BLUE, label="Predicted (Field-disp)")
    top = np.argsort(-pred)[:2]
    for j in top:
        ax.text(1.02, y[j] - 0.19, "review", va="center", fontsize=7, color=BLUE,
                fontweight="bold")
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.5, 1.0])
    if c == 1:
        ax.set_xlabel("Disagreement on the field, within-fold percentile")
    ax.grid(axis="x", color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    print(f"{coh}: {f}  profile Spearman {rho:.2f} (median of {n} contested test items); "
          f"review fields: {', '.join(names[j] for j in top)}")
    for j in range(len(names)):
        print(f"    {names[j]:12s} recorded {obs[j]:.2f}  predicted {pred[j]:.2f}")
h_, l_ = ax.get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc="lower center", ncol=2, fontsize=8)
fig.subplots_adjust(left=0.03, right=0.95, top=0.95, bottom=0.13)
# 600 dpi with nearest-neighbour display: every stored pixel is kept as it is (no
# smoothing), at well above the 300 dpi the journal asks for in photographs.
fig.savefig(HERE / "fig_cases.pdf", dpi=600)
