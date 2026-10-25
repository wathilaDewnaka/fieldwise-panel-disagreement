"""in plain words: one page that reads the scored results.

Nothing is re-scored here. It takes what `study_results.report` (and, with several
training seeds, `study_results.report_seeds`) already returned and writes it as a
short Markdown page, every method under its plain name:

    1. main result     how well each method predicts the panel's disagreement
    2. field by field  ours against the best other method on each report field
    3. accuracy        diagnostic AUC, and whether ours costs any
    4. clinical counts what the triage rule does to missed cases and reader load
    5. across seeds    the main result at each training seed (only with several)
    6. external test   the LIDC-trained methods, unchanged, on LNDb (when scored)

    from backend.notebooks.shared import study_summary
    study_summary.write(REPORT, cohort="lidc", out_dir=cfg.out)

A rival is beaten when the pooled 95% CI of ours minus it lies above 0."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from field_disagreement.triage.ladder import CLAIM, CONTROL

from .study_results import triage_contrasts

Q1_NAMES = {
    "H_fld":  ("**Ours**: predicts disagreement on each report field, with each "
               "field's rating histogram as a second target", "proposed"),
    "H_fldp": ("Ours trained on shuffled panels", "control"),
    "A_cls":  ("Classifier confidence", "standard practice"),
    "B_kl":   ("Opinion head, KL (label-distribution learning)", "prior art"),
    "C_emd":  ("Opinion head, EMD", "prior art"),
    "D_reg":  ("DUP: predicts one overall disagreement (Raghu 2019)", "prior art"),
    "R_rr":   ("Random-reader training", "prior art"),
    "U_edl":  ("Evidential deep learning", "prior art"),
    "P_emd":  ("Opinion head per report field, EMD", "prior art"),
    "A_cls_mc": ("MC-dropout uncertainty (Gal & Ghahramani 2016)", "prior art"),
}
SHORT = {"H_fld": "Ours", "H_fldp": "Shuffled control", "A_cls": "Confidence",
         "B_kl": "Opinion KL", "C_emd": "Opinion EMD", "D_reg": "DUP", "R_rr": "Random reader",
         "U_edl": "Evidential", "P_emd": "Opinion per field", "A_cls_mc": "MC dropout"}


def _cohort(name: str):
    """The registry entry naming `name` on a results page: its report-level
    condition where there is one (`lidc` -> `lidc_report`), else itself."""
    from backend.domains.datasets import DATASETS
    return DATASETS.get(f"{name}_report") or DATASETS.get(name)


def _title(name: str) -> str:
    spec = _cohort(name)
    return (spec.title if spec and spec.title else name)


def _positive(name: str) -> str:
    spec = _cohort(name)
    return spec.positive if spec else "positives"


def _m(x) -> float:
    a = np.asarray(x, dtype=float).ravel()
    return float(np.nanmean(a)) if a.size else float("nan")


def _mark(ok) -> str:
    return "✅" if ok else "❌"


def _name(table, key):
    return table.get(key, (key, ""))


# --------------------------------------------------------------------------- #
# 1. main result
# --------------------------------------------------------------------------- #
def q1_lines(rep, positive: str = "positives") -> tuple[list[str], dict]:
    """The main result and its counts."""
    if not rep:
        return ["_No results in this run folder yet._"], {"pass": None}
    rho = rep["primary"]["rho"]
    n_folds = len(rho[CLAIM])
    claim_rho = float(np.nanmean(rho[CLAIM]))
    rows, beaten, rivals = [], 0, 0
    for arm in rho:
        name, role = _name(Q1_NAMES, arm)
        r = float(np.nanmean(rho[arm]))
        if arm == CLAIM:
            rows.append(f"| {name} | {role} | **{r:.3f}** | | | |")
            continue
        p = rep["pooled"].get(arm)
        w = rep["won"].get(arm)
        if p is None:
            rows.append(f"| {name} | {role} | {r:.3f} | n/a | | |")
            continue
        ok = p["lo"] > 0
        rivals += 1
        beaten += ok
        rows.append(f"| {name} | {role} | {r:.3f} | {p['delta']:+.3f} "
                    f"[{p['lo']:+.3f}, {p['hi']:+.3f}] | {w}/{n_folds} | {_mark(ok)} |")
    par = rep.get("parity", {})
    below = [a for a, t in par.items() if t["ci95_hi"] < 0]
    lines = [
        "## 1. Main result: which report fields will the panel argue about?",
        "",
        "The model sees only the image and predicts, for each field of the report, how much "
        "the reading panel would disagree. **Score** = Spearman correlation between that "
        "prediction and the panel's real disagreement, averaged over the fields "
        "(0 = no information; higher is better).",
        "",
        "| Method | Role | Score | Ours ahead by [95% CI] | Folds won | Ours better? |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        f"Ours beats {beaten} of {rivals} methods "
        f"({_mark(beaten == rivals)} {'on every comparison' if beaten == rivals else 'not on every comparison'}); "
        "a method counts as beaten when the whole 95% interval is above 0.",
    ]
    lines += field_lines(rep)
    lines += accuracy_lines(rep, below)
    lines += served_lines(rep.get("served_v5"), positive)
    lines += two_sided_lines(rep.get("two_sided"), positive)
    return lines, {"pass": beaten == rivals, "beaten": beaten, "rivals": rivals,
                   "score": claim_rho, "parity_ok": not below}


# --------------------------------------------------------------------------- #
# 2. field by field
# --------------------------------------------------------------------------- #
def field_lines(rep) -> list[str]:
    pf = rep.get("per_field")
    if not pf or CLAIM not in pf:
        return []
    from .study_results import FIELD_NAMES
    ours = np.asarray(pf[CLAIM], dtype=float)
    names = FIELD_NAMES.get(len(ours), [f"field {j + 1}" for j in range(len(ours))])
    others = {a: np.asarray(v, dtype=float) for a, v in pf.items()
              if a not in (CLAIM, CONTROL)}
    rows, wins = [], 0
    for j, f in enumerate(names):
        best = max(others, key=lambda a: others[a][j]) if others else None
        b = others[best][j] if best else float("nan")
        wins += ours[j] > b
        rows.append(f"| {f} | **{ours[j]:.3f}** | {SHORT.get(best, best)} | {b:.3f} | "
                    f"{ours[j] - b:+.3f} |")
    return ["", "## 2. Field by field",
            "",
            f"Ours is ahead of the best other method on {wins} of {len(names)} fields.",
            "",
            "| Field | Ours | Best other method | Its score | Ours ahead by |",
            "|---|---|---|---|---|", *rows]


# --------------------------------------------------------------------------- #
# 3. accuracy
# --------------------------------------------------------------------------- #
def accuracy_lines(rep, below) -> list[str]:
    par = rep.get("parity", {})
    rows_auc = (rep.get("arms") or {}).get("rows", {})
    out = ["", "## 3. Diagnostic accuracy", ""]
    if below:
        out.append("Ours is significantly below "
                   + ", ".join(f"{_name(Q1_NAMES, a)[0]} ({par[a]['delta']:+.4f} AUC)"
                               for a in below) + ".")
    else:
        out.append("Predicting disagreement costs no diagnostic accuracy: ours is not "
                   "significantly below any method.")
    if rows_auc:
        out += ["", "| Method | AUC | Ours minus it [95% CI] |", "|---|---|---|"]
        for a, row in rows_auc.items():
            t = par.get(a)
            diff = (f"{t['delta']:+.4f} [{t['ci95_lo']:+.4f}, {t['ci95_hi']:+.4f}]"
                    if t else "")
            out.append(f"| {_name(Q1_NAMES, a)[0]} | {_m(row['auc']):.3f} | {diff} |")
    return out


# --------------------------------------------------------------------------- #
# 4b. two-sided triage
# --------------------------------------------------------------------------- #
def two_sided_lines(t, positive: str = "positives") -> list[str]:
    """The declared two-sided rule: the AI clears the cases it is surest are
    negative AND flags the ones it is surest are positive, holding back the cases
    it expects the panel to dispute. Each method with its own classifier."""
    if not t or not t.get("rows"):
        return []
    sota = t["sota"]
    rows = []
    for a, r in t["rows"].items():
        name = ("Standard practice: classifier certainty alone" if a == sota
                else _name(Q1_NAMES, a)[0])
        rows.append(f"| {name} | {100 * _m(r['automated']):.0f}% | "
                    f"{_m(r['missed_per_1000']):.1f} | {_m(r['false_flags_per_1000']):.1f} | "
                    f"{100 * _m(r['contested_automated']):.1f}% |")
    wins = t.get("claim_wins") or []
    return [
        "", "## 4b. Clinical counts: the AI decides more than half the cases",
        "",
        f"The AI clears the {t['clear']:.0%} of cases it is surest are negative and flags "
        f"the {t['flag']:.0%} it is surest are positive; a reader takes the rest. Each case "
        f"is ranked by {1 - t['w']:.1f} x its certainty rank + {t['w']:.1f} x its predicted "
        "disagreement rank, so a case the panel would dispute is held back from both sides. "
        "These settings were fixed before the run.",
        "",
        f"| Method | Decided by AI | Missed {positive} per 1000 (lower) | False flags per "
        "1000 (lower) | Disputed cases among AI decisions (lower) |",
        "|---|---|---|---|---|",
        *rows,
        "",
        (f"Ours misses no more {positive} AND leaves fewer disputed cases to the AI than "
         f"standard practice on {sum(wins)} of {len(wins)} folds. Each fold has only a few "
         "misses, so one case can flip a fold." if wins else ""),
    ]


# --------------------------------------------------------------------------- #
# 4. clinical counts
# --------------------------------------------------------------------------- #
def served_lines(sp, positive: str = "positives") -> list[str]:
    """The triage rule in clinical counts, and the declared criteria. One classifier
    makes every diagnosis; each method then sends the same share of negative calls
    to a reader by its own signal, plus the same safety net by its own opinion
    spread, so only the method's own outputs differ between rows."""
    if not sp:
        return []

    def row(name, r):
        return (f"| {name} | {100 * _m(r['auto']):.1f}% | {_m(r['missed_per_1000']):.1f} | "
                f"{100 * _m(r['high_neg']):.1f}% | {100 * _m(r['error']):.1f}% |")
    c, v4 = sp["claim"], sp["claim_v4"]
    rows = [row("Standard practice: AI clears every case it is sure of", c["sota"]),
            row("Same classifier, stricter certainty only (same automation)", c["matched"]),
            row("Random referral of the same size", c["randveto"]),
            row("**Ours**", v4["served"])]
    for a, v in sp["signal"].items():
        rows.append(row(_name(Q1_NAMES, a)[0], v["served"]))
    net = sp.get("net") or 0

    def ci(t, scale, unit):
        return f"{scale * t[0]:+.1f} {unit} [{scale * t[1]:+.1f}, {scale * t[2]:+.1f}]"
    crit = triage_contrasts(sp)
    crows = [f"| {_name(Q1_NAMES, a)[0]} | {ci(r['disputed'], 100, 'pts')} "
             f"(lower in {r['disputed'][4]}/{r['n']} folds) | "
             f"{ci(r['missed'], 1, '')} (not higher in {r['missed'][4]}/{r['n']} folds) | "
             f"{ci(r['auto'], 100, 'pts')} |"
             for a, r in crit.items()]
    return [
        "", "## 4. Clinical counts: the triage rule",
        "",
        f"One classifier makes every diagnosis. The AI decides alone the cases it is sure "
        f"of, negative or positive; a radiologist reads the rest. Of its negative calls, "
        f"the {sp['veto']:.0%} the method rates most disputed go to a radiologist"
        + (f", and so do the {net:.0%} of all negative calls with the widest predicted "
           "opinion spread (a safety net)" if net else "")
        + ". Every method does this with its own outputs and the same shares. Measured "
        "on held-out halves of each test fold.",
        "",
        "| Rule | Decided by AI alone (higher) | Missed "
        f"{positive} per 1000 (lower) | Strongly disputed cases the AI clears alone "
        "(lower) | Errors among AI decisions (lower) |",
        "|---|---|---|---|---|",
        *rows,
        "", "### Ours against each prior-art method",
        "",
        "Ours minus the method, paired over folds, with 95% intervals. Below zero means "
        f"ours clears fewer disputed cases, or misses fewer {positive}.",
        "",
        "| Method | Disputed cases cleared | Missed per 1000 | Decided by AI alone |",
        "|---|---|---|---|",
        *crows,
    ]


# --------------------------------------------------------------------------- #
# 5. across seeds
# --------------------------------------------------------------------------- #
def seeds_lines(srep) -> list[str]:
    """The main result at every training seed, and averaged across them."""
    if not srep or len(srep.get("seeds", [])) < 2:
        return []
    seeds = srep["seeds"]
    rows = []
    for r, t in srep["pooled"].items():
        name, role = _name(Q1_NAMES, r)
        cells = " | ".join(f"{t['per_seed'][s]:+.3f}" if s in t["per_seed"] else "n/a"
                           for s in seeds)
        rows.append(f"| {name} | {cells} | {t['delta']:+.3f} "
                    f"[{t['lo']:+.3f}, {t['hi']:+.3f}] | {t['won']}/{t['n_folds']} | "
                    f"{t['seeds_ci_above_0']}/{t['n_seeds']} |")
    return ["", f"## 5. Across training seeds ({', '.join(map(str, seeds))})",
            "",
            "Ours ahead of each method on the main score, at each seed and averaged over "
            "the seeds. *Seeds with CI above 0* counts the seeds whose own 95% interval is "
            "above 0.",
            "", "| Method | " + " | ".join(f"Seed {s}" for s in seeds)
            + " | Mean [95% CI] | Folds won | Seeds with CI above 0 |",
            "|---|" + "---|" * len(seeds) + "---|---|---|", *rows]


# --------------------------------------------------------------------------- #
# 6. external test
# --------------------------------------------------------------------------- #
def external_lines(ext) -> list[str]:
    """The LIDC-trained methods, unchanged, on an external cohort (study_external)."""
    if not ext:
        return []
    rows = []
    for a, r in ext["rho"].items():
        name, role = _name(Q1_NAMES, a)
        c = ext["contrast"].get(a)
        if c is None:
            rows.append(f"| {name} | {role} | **{r:.3f}** | | | {ext['auc'][a]:.3f} |")
            continue
        rows.append(f"| {name} | {role} | {r:.3f} | {c['delta']:+.3f} [{c['lo']:+.3f}, "
                    f"{c['hi']:+.3f}] | {c['won']}/{c['n_nets']} | {ext['auc'][a]:.3f} |")
    beaten = sum(c["lo"] > 0 for c in ext["contrast"].values())
    lab = ext.get("labels", {})
    return ["", f"## 6. External test: {_title(ext['cohort'])}",
            "",
            f"Each method's five LIDC-trained networks score {ext['n_items']} nodules from "
            f"{ext['n_patients']} patients ({lab.get(1, 0)} malignant-rated, {lab.get(0, 0)} "
            "benign-rated) without retraining or recalibration. Score as in section 1, averaged "
            "over the five networks; the interval resamples patients, shared by every network.",
            "",
            "| Method | Role | Score | Ours ahead by [95% CI] | Networks won | AUC |",
            "|---|---|---|---|---|---|", *rows, "",
            f"Ours beats {beaten} of {len(ext['contrast'])} methods on the external cohort "
            f"({_mark(beaten == len(ext['contrast']))})."]


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 7. sensitivity: the other selection rule
# --------------------------------------------------------------------------- #
def sensitivity_lines(rep, sens) -> list[str]:
    """Ours minus each prior-art method under the study's selection rule and
    under val AUC alone (the same training run, the other kept epoch)."""
    if not rep or not sens:
        return []

    def score(r, a):
        p = (r.get("pooled") or {}).get(a)
        return (f"{p['delta']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]" if p else "n/a")

    def disputed(r, a):
        c = triage_contrasts(r.get("served_v5")).get(a)
        return f"{100 * c['disputed'][0]:+.1f} [{100 * c['disputed'][1]:+.1f}, " \
               f"{100 * c['disputed'][2]:+.1f}]" if c else "n/a"

    def missed(r, a):
        c = triage_contrasts(r.get("served_v5")).get(a)
        return f"{c['missed'][0]:+.1f} [{c['missed'][1]:+.1f}, {c['missed'][2]:+.1f}]" \
            if c else "n/a"
    arms = [a for a in rep["primary"]["rho"] if a != CLAIM and _name(Q1_NAMES, a)[1] != "control"]
    rows = [f"| {_name(Q1_NAMES, a)[0]} | {score(rep, a)} | {score(sens, a)} | "
            f"{disputed(rep, a)} | {disputed(sens, a)} | {missed(rep, a)} | {missed(sens, a)} |"
            for a in arms]
    return ["", "## 7. Does the result depend on how the epoch was chosen?", "",
            "Every method keeps, within 0.01 of its best validation AUC, the epoch where its "
            "own disagreement signal is best on validation (main tables). Below, the same "
            "networks are also scored at the epoch chosen by validation AUC alone. Ours minus "
            "each method, 95% interval.", "",
            "| Method | Score: main rule | Score: AUC only | Disputed cleared (pts): main "
            "| AUC only | Missed per 1000: main | AUC only |",
            "|---|---|---|---|---|---|---|", *rows]


def build(q1_report, cohort: str, seed: int | None = None, seeds_report=None,
          external_report=None, sensitivity_report=None) -> str:
    pos = _positive(cohort)
    l1, v1 = q1_lines(q1_report, pos)
    seeds = (seeds_report or {}).get("seeds") or ([seed] if seed is not None else [])
    title = f"# Results: {_title(cohort)}"
    if seeds:
        title += f", seed {seeds[0]}" if len(seeds) == 1 else \
                 f", seeds {', '.join(map(str, seeds))} (tables 1-4: seed {seeds[0]})"
    head = [title, ""]
    if v1["pass"] is not None:
        head += ["| At a glance | |", "|---|---|",
                 f"| Ours beats every other method | {_mark(v1['pass'])} "
                 f"{v1['beaten']}/{v1['rivals']} rivals beaten, score {v1['score']:.3f} |",
                 f"| No loss of diagnostic accuracy | {_mark(v1['parity_ok'])} |", ""]
    return "\n".join(head + l1 + seeds_lines(seeds_report)
                     + external_lines(external_report)
                     + sensitivity_lines(q1_report, sensitivity_report) + [""])


def write(q1_report, cohort: str, out_dir, seed: int | None = None,
          seeds_report=None, external_report=None, sensitivity_report=None) -> Path:
    """Write SUMMARY_<cohort>.md into `out_dir` and return the path. The page is
    saved for the write-up, not shown: the notebook prints every table itself."""
    text = build(q1_report, cohort, seed, seeds_report, external_report,
                 sensitivity_report)
    path = Path(out_dir) / f"SUMMARY_{cohort}.md"
    path.write_text(text, encoding="utf-8")
    print(f"summary page saved to {path}")
    return path
