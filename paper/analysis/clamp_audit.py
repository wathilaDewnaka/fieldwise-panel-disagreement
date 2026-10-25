"""How often the mask-based translation clamp binds, and whether how much it binds
tracks the panel's disagreement (the route by which it could leak the target)."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import os, glob
os.chdir(ROOT)
import numpy as np
from scipy.stats import spearmanr
for coh in ("lidc_report", "chaksu_report"):
    ex, dd, bind, dprof = [], [], 0, []
    for f in sorted(glob.glob(f"data/processed/{coh}/*.npz")):
        z = np.load(f, allow_pickle=True)
        if int(z["label"]) < 0:
            continue
        mask = z["binary_mask"] > 0
        H, W = mask.shape
        m = int(round(0.2 * W))
        ys, xs = np.nonzero(mask)
        if not len(ys):
            continue
        lo_y, hi_y = -min(m, ys.min()), min(m, H - 1 - ys.max())
        lo_x, hi_x = -min(m, xs.min()), min(m, W - 1 - xs.max())
        frac = (max(hi_y - lo_y + 1, 1) * max(hi_x - lo_x + 1, 1)) / (2 * m + 1) ** 2
        ex.append(1 - frac); dd.append(float(z["disagreement"])); dprof.append(np.asarray(z["d_profile"], float))
        bind += frac < 1
    ex = np.array(ex); dprof = np.array(dprof)
    print(f"{coh}: n {len(ex)}, clamp binds for {bind / len(ex):.1%} of items; mean share of offsets excluded {ex.mean():.1%}")
    print(f"  Spearman(excluded share, report-level d) {spearmanr(ex, dprof.mean(1))[0]:+.3f}; "
          f"per field range {min(spearmanr(ex, dprof[:, j])[0] for j in range(dprof.shape[1])):+.3f}.."
          f"{max(spearmanr(ex, dprof[:, j])[0] for j in range(dprof.shape[1])):+.3f}")
