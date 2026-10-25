# Manuscript analyses

Every number in the paper is computed by these scripts from the per-item test
predictions that training stores under `data/runs/` (written by the notebook below).
The paper's own predictions for LIDC-IDRI and Chákṣu are in `paper/predictions/`:
unzip `runs_lidc.zip` and `runs_chaksu.zip` into `data/` (each holds a `runs/` folder).
The LNDb predictions are not included, because LNDb's licence (CC BY-NC-ND 4.0) does
not allow redistributing material derived from it; the LNDb steps need LNDb itself
and a run of the notebook's external-test cell. The scripts also read the processed
report-level cohorts (`data/processed/*_report`), built as in the top-level README. The scripts run on CPU and do not train or
rescore a network. Run them from this folder, or from anywhere: each script finds
the repository root from its own location.

The training runs are produced by `backend/notebooks/train_evaluate.ipynb`
(config `configs/notebooks/study.yaml`, panel weight 0.3, seeds 42/43/44,
folds `configs/folds/folds_{lidc,chaksu}.json`, split seed 42).

## Inputs

| Path under `data/runs/` | Contents |
|---|---|
| `lidc_report_seed{42,43,44}` | LIDC-IDRI, all arms, one folder per seed, plus `external_lndb/` |
| `chaksu_report_seed42` | Chákṣu, seed 42 |
| `chaksu_report_seeds43_44` | Chákṣu, seeds 43 and 44 |
| `lidc_report_weight03`, `chaksu_report_weight03` | DUP, Field-soft and Disp-only at weight 0.3 (read first; they take priority) |

The reference-diagnosis analysis also reads
`data/raw/lidc-idri/others/tcia-diagnosis-data-2012-04-20.xls` (TCIA), and needs `xlrd`.

## Running

```
python run_all.py              # all steps, in order (about 8 h on a laptop CPU)
python run_all.py reanalysis   # one step
```

`compute` takes about 2 h (about 20 min per cohort and seed) and `bootstrap` about
45 min per cohort; every other step takes minutes. `ms_compute.py` saves after each
cohort and seed, so an interrupted run resumes where it stopped, and
`python ms_compute.py LIDC 43` runs a single pair.

Each script writes a `.pkl` next to itself, and a `log_*.txt` with its printed output.

| Step | Script(s) | Manuscript |
|---|---|---|
| compute | `ms_compute.py` | triage rows of unchanged arms, reused by `ms_w03.py` |
| main | `ms_w03.py <cohort> <seed>` | main tables: per-field ρ, S1–S4, AUC, triage |
| bootstrap | `boot_triage.py <cohort> 2000 10` | triage and AUC intervals (about 45 min per cohort) |
| lndb | `lndb_w03.py` | external LNDb table |
| subsets | `ms_subsets.py`, `ms_subsets_w.py` | panel size, camera, median-3, rounded VCDR (Robustness) |
| baselines | `rev_handcrafted.py`, `rev_fields_w.py`, `rev_sigma_w.py`, `rev_embed.py` | hand-crafted ridge, related fields, training-only scales, frozen ResNet-50 / DINOv2 embeddings (about 40 min; needs the torchvision ResNet-50 and DINOv2 hub weights) |
| reanalysis | `reanalysis.py <cohort> A/B/C` | field prioritisation table, matched-budget figure, deployment-style table |
| workload | `review_workload.py` | field prioritisation as highly disputed fields per 100 field reviews |
| glaucoma | `glaucoma_votes.py` | predicted VCDR disagreement and split glaucoma decisions (Chákṣu) |
| routing | `field_routing.py <cohort>`, `undecided_routing.py` | combining the fields for routing, event counts and the margin, seed ensemble; routing of the undecided nodules |
| external | `lndb_prio.py` | field prioritisation on LNDb |
| reference | `anchor.py` | disagreement and the reference diagnosis |
| audits | `bin_audit.py`, `clamp_audit.py` | annotation-dependent preprocessing (Robustness) |
| stability | `boot_triage.py <cohort> 500 50 0,1,2,3,4 halv50`, `halving_check.py` | triage intervals with 50 halvings per draw (about 50 min per cohort) |
| print | `ms_print_w03.py`, `extra_w03.py`, `drop_bkl.py`, `rean_print.py` | print the numbers as quoted in the text |

The figures are drawn by `../figures/make_figures.py` from the printed values, and the
illustrative cases by `../figures/fig_cases.py` from the stored predictions (case chosen by
a fixed rule, stated in the script).

## Conventions

* Bootstraps resample patients within each test fold; the same draws are used for
  every arm and seed, so contrasts are paired. Seeds are pooled draw by draw.
* Random generators are seeded inside each script, so a rerun reproduces every digit.
* `ROOT` (the repository root) is derived from each script's location; scripts can
  be run from any working directory.
