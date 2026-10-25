# Field-level prediction of expert-panel disagreement

[![DOI](https://zenodo.org/badge/1432964297.svg)](https://doi.org/10.5281/zenodo.23637205)

Code for the paper

> W. Karunathilake, G. Poravi, A. Aponso, M. Edirisooriya.
> *Field-level prediction of expert-panel disagreement for review prioritisation in
> lung-nodule CT and glaucoma fundus images.* Submitted to Computers in Biology and Medicine.

From an image, the method predicts how much an expert panel will disagree on **each field**
of a structured report (for a lung nodule: subtlety, sphericity, margin, lobulation,
spiculation, texture, malignancy; for a fundus image: cup-to-disc ratios and disc/cup sizes),
and uses the predicted vector to choose which cases, and which fields, a second reader
should review. The repository contains the training code for all nine compared methods,
the preprocessing of the three cohorts, and the scripts that recompute every number,
table and figure in the paper from the stored per-item test predictions.

## Repository layout

| Path | Contents |
|---|---|
| `field_disagreement/` | The framework: backbones, objectives, the arm definitions (`field_disagreement/triage/ladder.py`), metrics, bootstrap, conformal and triage rules |
| `backend/domains/` | Preprocessing of LIDC-IDRI (`lidc`), Chákṣu (`chaksu`) and LNDb (`lndb`), and the per-field reader tables (`field_opinions.py`, `report_condition.py`) |
| `backend/notebooks/train_evaluate.ipynb` | Trains and evaluates all arms (one notebook; local GPU, Colab or Kaggle) |
| `backend/notebooks/shared/` | Training loop, evaluation, external (LNDb) scoring, result I/O used by the notebook |
| `configs/` | Experiment config (`notebooks/study.yaml`, panel weight 0.3), cohort configs, preprocessing configs, **the patient-level fold assignments** (`folds/*.json`), data paths (`paths.yaml`) |
| `paper/analysis/` | Recomputes every number in the paper from the stored predictions (CPU only); see `paper/analysis/README.md` |
| `paper/predictions/` | The paper's per-item test predictions for LIDC-IDRI and Chákṣu (all arms, folds and seeds) |
| `paper/figures/` | Draws the paper's figures (`make_figures.py`, `fig_cases.py`) |
| `backend/tests/` | Unit tests (CPU) |

Every arm is defined in `field_disagreement/triage/ladder.py`. Arm names in the code map
to the paper as follows:

| Code | Paper |
|---|---|
| `A_cls` | Confidence (and MC dropout, read from the same network) |
| `B_kl` | Soft-KL |
| `C_emd` | Soft-EMD |
| `D_reg` | DUP (Direct Uncertainty Prediction) |
| `R_rr` | Random-reader |
| `U_edl` | Evidential |
| `P_emd` | Field-soft |
| `H_fld` | **Field-disp** (the proposed method) |
| `H_fldp` | Field-disp, shuffled panels (control) |
| `H_disp` | Disp-only (ablation) |

## Data

The images are not redistributed. Download them from their sources and accept their licences:

| Cohort | Source | Licence |
|---|---|---|
| LIDC-IDRI | The Cancer Imaging Archive, https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX (images, XML annotations, and `tcia-diagnosis-data-2012-04-20.xls` for the reference-diagnosis analysis) | CC BY 3.0 |
| Chákṣu | Kumar et al., Scientific Data 2023, https://doi.org/10.1038/s41597-023-01943-4 | see the data descriptor |
| LNDb | Pedrosa et al., https://lndb.grand-challenge.org | CC BY-NC-ND 4.0 |

Point `configs/paths.yaml` at the downloads, then build the report-level cohorts the
paper trains and tests on (`lidc_report`, `chaksu_report`, `lndb_report`):

```
# LIDC-IDRI (the [lidc] environment below)
python -m backend.domains.lidc.setup_pylidc             # writes pylidc's config from configs/paths.yaml
python -m backend.domains.lidc.prepare --paper-cohort     # one .npz per nodule -> lidc-idri
python -m backend.domains.lidc.field_ratings              # per-field reader s.d. -> lidc_multi
python -m backend.domains.report_condition --cohort lidc  # report-level target -> lidc_report
python -m backend.domains.field_opinions --cohort lidc    # per-field rating histograms

# Chákṣu (streams from the downloaded Train.zip / Test.zip; never extracts them)
python -m backend.domains.chaksu.prepare                  # -> chaksu
python -m backend.domains.chaksu.vcdr_opinion             # experts' VCDRs -> chaksu_vcdr
python -m backend.domains.report_condition --cohort chaksu
python -m backend.domains.field_opinions --cohort chaksu

# LNDb, external test only (needs 7-Zip and the finished lidc_report)
python -m backend.domains.lndb.prepare
python -m backend.domains.field_opinions --cohort lndb
```

**`--paper-cohort`.** Seven LIDC-IDRI patients have two CT series. The paper's cohort was
built with filenames that did not include the series, so where two of those nodules had
the same index, the second overwrote the first: the cohort has 1,377 nodules where the
annotations allow 1,386. `--paper-cohort` reproduces that cohort, which is what the fold
file and the paper's numbers refer to. Without the flag, each series gets its own token in
the filename and all 1,386 nodules are written. That cohort is not the paper's, and it
needs new folds (`python -m backend.domains.freeze_folds --force`).

The fold files in `configs/folds/` fix the patient-disjoint five-fold split used in the paper
(split seed 42). Training checks each cohort against the fingerprint stored in its fold file,
so a cohort that differs from the paper's is reported rather than silently re-split.

## Environments

* **Training, evaluation and the paper's analyses** — Python 3.12:
  `pip install -r requirements.txt`. The analyses were verified with the versions pinned there.
* **LIDC-IDRI preprocessing** — `pylidc` 0.2.3 needs Python 3.11 and NumPy < 1.24:
  `pip install -e ".[lidc]"` in a Python 3.11 environment (see the comments in `pyproject.toml`).
  Chákṣu and LNDb preprocessing run in the Python 3.12 environment above (LNDb also needs
  7-Zip on the PATH).

## Training

Open `backend/notebooks/train_evaluate.ipynb`, set the flags at the top of its first cell, and
run it top to bottom:

| Flag | Set to `True` when |
|---|---|
| `RUN_ON_KAGGLE` | running on Kaggle. Build the two upload folders with `python -m backend.notebooks.shared.kaggle bundle`, upload them as Kaggle datasets and attach both to the notebook (Add Input) |
| `RUN_ON_COLAB` | running on Colab. Upload the whole repository folder to Google Drive, with the processed cohort inside it as `data/processed/<cohort>.zip` (faster to upload) or as the `data/processed/<cohort>/` folder, and set `DRIVE_ROOT` if it is not `MyDrive/field-disagreement-prediction` |
| `RELEASE_RUNTIME_AT_END` | the last cell should disconnect the Colab runtime once every result is saved (no effect elsewhere) |

Leave both platform flags `False` for a local run. Results are written under `data/runs/` (on
Colab, to the Drive folder, so they survive a disconnect; an interrupted fold resumes from its
last saved epoch). The configuration is
`configs/notebooks/study.yaml` (ResNet-50 initialised on ImageNet, panel-loss weight 0.3
for every arm, seeds 42/43/44, the fold files above, all nine arms plus the Disp-only
ablation). The cohort is set in `configs/notebooks/base.yaml` (`extends: cohorts/lidc` or
`cohorts/chaksu`). Each run writes, per arm and fold, a result file with the per-item test
predictions under `data/runs/`.

The paper's runs were produced in two steps: the nine arms in one run per seed
(`*_report_seed42`, `*_seed43`, `*_seed44` in `paper/predictions/`), in which DUP and
Field-soft still had panel weight 1.0, and then DUP, Field-soft and the ablation retrained
at 0.3 (`*_report_weight03`), which the analyses read first. The release config trains every
arm at 0.3 in a single run (`study_*`). The analysis scripts name the paper's run folders; to
analyse a new run, change the folder names in `paper/analysis/ms_subsets.py` (`COH`) and in
`boot_triage.py`, `ms_compute.py`, `ms_w03.py`, `ms_subsets_w.py`, `rev_sigma_w.py`,
`lndb_w03.py` and `lndb_prio.py`. The paper's runs were trained on Colab and Kaggle GPUs
(`backend/notebooks/shared/kaggle.py` describes the Kaggle set-up); runs are deterministic
for a given GPU type but not across types.

## Reproducing the paper's numbers

Unzip `paper/predictions/runs_lidc.zip` and `runs_chaksu.zip` into `data/`, build the
report-level cohorts (above), then:

```
cd paper/analysis
python run_all.py            # every step, in dependency order (CPU, about 8 h on a laptop)
python run_all.py main       # one step
```

`paper/analysis/README.md` maps every step to the table, figure or paragraph it produces.
Every random generator is seeded inside the scripts, so a rerun gives the same numbers. The
LNDb steps are skipped unless the LNDb outputs exist under `data/runs/` (they are not
redistributed; producing them needs LNDb and the trained LIDC-IDRI networks).

## Tests

```
python -m pytest
```

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). The code is archived on
Zenodo: https://doi.org/10.5281/zenodo.23637205 (all versions).

## Licence

MIT (see `LICENSE`). The datasets keep their own licences.
