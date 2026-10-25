"""Figures for the field-level disagreement paper, drawn from the final result tables.

Every number is the mean over three seeds (42, 43, 44) and five folds of the final
nine-arm runs, as printed in the manuscript's tables. Run from this folder:
python make_figures.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#7a5cc4"

# ---------------------------------------------------------------- Figure 2
lidc = {"fields": ["Subtlety", "Sphericity", "Margin", "Lobulation", "Spiculation",
                   "Texture", "Malignancy"],
        "H_fld": [.488, .150, .528, .475, .465, .554, .449],
        "P_emd": [.428, .055, .477, .411, .439, .510, .364],
        "D_reg": [-.176, .025, .088, .211, .270, -.006, .418]}
chaksu = {"fields": ["ACDR", "VCDR", "HCDR", "Disc area", "Cup area", "Rim area",
                     "Disc height", "Disc width", "Cup height", "Cup width"],
          "H_fld": [.218, .284, .324, .260, .532, .221, .226, .227, .477, .510],
          "P_emd": [.095, .174, .220, .219, .521, .198, .218, .189, .463, .502],
          "D_reg": [.138, .232, .250, .188, .410, -.021, .178, .148, .369, .372]}

fig, axes = plt.subplots(1, 2, figsize=(6.84, 3.2),
                         gridspec_kw={"width_ratios": [7, 10]})
for ax, d, title in ((axes[0], lidc, "LIDC-IDRI (7 fields)"),
                     (axes[1], chaksu, "Chákṣu (10 fields)")):
    y = np.arange(len(d["fields"]))[::-1]
    h = 0.26
    for k, (arm, col, lab) in enumerate((("H_fld", BLUE, "Field-disp (ours)"),
                                         ("P_emd", AQUA, "Field-soft"),
                                         ("D_reg", ORANGE, "DUP"))):
        ax.barh(y + (1 - k) * h, d[arm], height=h - 0.03, color=col, label=lab,
                edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#52514e", lw=0.6)
    ax.set_yticks(y, d["fields"])
    ax.set_xlim(-0.2, 0.6)
    ax.set_xlabel("Spearman ρ with the panel's disagreement on that field")
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.grid(axis="x", color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc="lower center", ncol=3, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("fig_perfield.pdf")

# ---------------------------------------------------------------- Figure 3
rules = ["Uncertainty threshold, 5%", "Certainty, same automation",
         "Rule with Field-soft signal (strongest baseline)", "Rule with Field-disp signal (ours)"]
cols = [GREY, PURPLE, AQUA, BLUE]
# missed positives per 1000 cases; % of highly disputed cases cleared as negative
data = {"LIDC-IDRI": {"missed": [30.9, 14.1, 8.6, 6.4],
                      "disputed": [24.0, 14.1, 11.6, 9.4]},
        "Chákṣu": {"missed": [9.3, 3.3, 5.4, 5.0],
                   "disputed": [58.3, 39.0, 20.7, 20.4]}}
fig, axes = plt.subplots(1, 4, figsize=(6.84, 2.45))
for j, (coh, d) in enumerate(data.items()):
    for i, (key, lab) in enumerate((("missed", "Missed positives per 1000"),
                                    ("disputed", "Highly disputed cleared (%)"))):
        ax = axes[2 * j + i]
        x = np.arange(4)
        ax.bar(x, d[key], color=cols, width=0.7, edgecolor="white", linewidth=0.6)
        for xi, v in zip(x, d[key]):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7,
                    color="#0b0b0b")
        ax.set_xticks([])
        ax.set_title(f"{coh}\n{lab}", fontsize=8, loc="left")
        ax.set_ylim(0, max(d[key]) * 1.18)
        ax.grid(axis="y", color="#e4e3df", lw=0.5)
        ax.set_axisbelow(True)
fig.legend([Patch(color=c) for c in cols], rules, frameon=False, loc="lower center",
           ncol=2, fontsize=8)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig("fig_served.pdf")
print("wrote fig_perfield.pdf, fig_served.pdf")

# ---------------------------------------------------------------- Figure 3
# Matched reading budgets: share of the Confidence classifier's negative calls sent
# to a reader, chosen by each signal; label-free cuts (reanalysis of saved test
# predictions). Means over three seeds and five folds.
budgets = [10, 20, 30, 40, 50, 60, 70]
curves = {
    "LIDC-IDRI": {
        "Uncertainty":  ([35.2, 29.6, 24.1, 16.7, 10.4, 7.0, 4.4], [54.7, 38.3, 26.8, 17.6, 9.2, 6.5, 3.8]),
        "MC dropout":   ([37.8, 32.6, 24.1, 18.5, 13.0, 6.7, 4.4], [56.2, 39.8, 28.7, 18.0, 10.3, 6.9, 3.8]),
        "DUP":          ([40.7, 34.8, 28.5, 20.4, 13.7, 8.1, 5.6], [55.1, 42.5, 32.5, 22.2, 14.9, 8.0, 4.2]),
        "Field-soft":   ([32.2, 23.3, 16.3, 11.1, 7.4, 3.7, 1.9], [60.1, 44.8, 31.0, 22.2, 13.4, 6.9, 3.4]),
        "Field-disp (ours)": ([30.4, 19.3, 12.6, 7.8, 5.6, 3.3, 2.6], [60.8, 47.1, 36.0, 24.9, 18.0, 10.7, 4.6]),
        "Random":       ([39.3, 35.0, 30.6, 26.2, 21.9, 17.5, 13.1], [73.0, 64.9, 56.8, 48.7, 40.6, 32.4, 24.3])},
    "Chákṣu": {
        "Uncertainty":  ([82.0, 76.8, 70.4, 60.5, 52.1, 44.2, 36.0], [26.4, 15.7, 10.0, 6.5, 4.0, 3.0, 3.0]),
        "MC dropout":   ([82.7, 76.8, 68.6, 61.7, 53.3, 45.9, 36.5], [27.9, 16.9, 11.0, 6.7, 5.5, 3.2, 2.7]),
        "DUP":          ([66.2, 50.4, 37.0, 26.7, 18.5, 11.9, 7.9], [38.1, 37.6, 35.4, 33.4, 29.9, 25.9, 23.7]),
        "Field-soft":   ([63.5, 45.4, 33.1, 24.4, 18.8, 13.3, 10.4], [37.9, 37.6, 36.9, 35.9, 33.9, 31.1, 26.9]),
        "Field-disp (ours)": ([61.5, 46.9, 35.3, 23.7, 16.5, 10.9, 7.4], [38.1, 35.9, 35.1, 34.4, 31.4, 28.4, 24.2]),
        "Random":       ([79.1, 70.3, 61.5, 52.7, 44.0, 35.2, 26.4], [35.0, 31.1, 27.2, 23.3, 19.4, 15.5, 11.7])}}
style = {"Uncertainty": (GREY, "-"), "MC dropout": (GREY, "--"), "DUP": (ORANGE, "-"),
         "Field-soft": (AQUA, "-"), "Field-disp (ours)": (BLUE, "-"), "Random": ("#b9b8b3", ":")}
fig, axes = plt.subplots(1, 4, figsize=(6.84, 2.5))
for j, (coh, d) in enumerate(curves.items()):
    for i, lab in enumerate(("Highly disputed cleared (%)", "Missed positives per 1000")):
        ax = axes[2 * j + i]
        for name, vals in d.items():
            c, ls = style[name]
            ax.plot(budgets, vals[i], color=c, ls=ls, lw=1.4 if "ours" in name else 1.0,
                    marker="o", ms=2.2, label=name)
        ax.set_title(f"{coh}\n{lab}", fontsize=8, loc="left")
        ax.set_xlabel("Negative calls read (%)", fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(color="#e4e3df", lw=0.5)
        ax.set_axisbelow(True)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc="lower center", ncol=6, fontsize=7)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("fig_budget.pdf")
print("wrote fig_budget.pdf")
