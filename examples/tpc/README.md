# TPC reproduction: clinical experiments pending

**Status: clinical dataset access/preparation is PENDING; clinical experiments
have NOT RUN; clinical results are PENDING.** This contribution does not yet
claim reproduction of the paper's clinical scores. No clinical dataset,
patient-level output, or trained clinical checkpoint is included in this PR.

Paper: Rocheteau et al., CHIL 2021,
[Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the
Intensive Care Unit](https://arxiv.org/abs/2007.09483).

## What is ready

- `pyhealth/models/tpc.py`: TPC architecture, five dilation schedules, four loss
  modes, configurable decay, and optional lab forward-fill.
- `pyhealth/tasks/tpc_day_one.py`: a day-one adapter using PyHealth's existing
  YAIB feature extraction and cohort rules for eICU or MIMIC-IV.
- `examples/eicu_los_tpc.py`: preparation-only mode, local-data preflight,
  patient-disjoint splits, training-only normalization, validation checkpoint
  selection, three-seed evaluation, and Markdown/JSON result generation.
- Synthetic-only unit tests cover both adapters, preprocessing, missing-data
  errors, preparation without dataset access, and a tiny checkpoint round trip.

The clinical data path has **not** been validated against actual source files.
Unit-test success does not establish clinical cohort counts, training behavior,
runtime, or clinical performance.

## Clinical results table — not measured

All cells below are intentional placeholders, not zero scores or estimates.
After data access and successful experiments, replace them with generated
mean +/- sample standard deviation over at least three seeds.

| Configuration | MAD (days) | MAPE (%) | MSLE | R2 | Bias (days) | Long-stay MAD |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline TPC | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Constant dilation | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Linear/paper dilation | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Exponential dilation | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Asymmetric MSLE | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Huber log-loss | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Decay base 0.5 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Decay base 0.9 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Lab no-forward-fill | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| MSE — Sanity Check | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

Dataset/version, eligible cohort size, split sizes, hardware, runtime, selected
epochs, clinical findings, and comparison with published paper scores:
**PENDING actual data access and execution**. All hyperparameter configurations
also remain PENDING; preparation mode generates the complete 45-row table.

## Commands: no access credentials in code or chat

Install this repository and its declared dependencies in a supported Python
environment. Run commands from the repository root.

Prepare a plan and complete placeholder table **without data access or training**:

```bash
python -m examples.eicu_los_tpc --prepare-only --all-configs \
  --output-dir output/tpc_clinical_plan
```

This writes `plan.json` and `results.md` with explicit PENDING status. Every
output directory must be new to avoid overwriting earlier work. Preparation
mode is the default even if `--prepare-only` is omitted.

**Only after obtaining authorized local eICU-CRD access**, a baseline command is:

```bash
python -m examples.eicu_los_tpc --run --dataset eicu \
  --root /AUTHORIZED/PATH/eicu-crd/2.0 --data-version 2.0 \
  --configs baseline --seeds 0 1 2 --epochs 8 --batch-size 32 \
  --output-dir output/tpc_eicu_baseline
```

For MIMIC-IV, use `--dataset mimic4`, the actual release in `--data-version`,
and its local root containing `hosp/` and `icu/`. Preflight checks the source
tables and configured join files, including `hosp/d_labitems.csv[.gz]`.
eICU requires patient, lab, vitalPeriodic, vitalAperiodic, intakeOutput,
respiratoryCharting, and nurseCharting CSVs (compressed or uncompressed).
The script does not download data or request credentials.

To run all 45 one-factor configurations later, replace `--configs baseline`
with `--all-configs` and choose a new output directory. Alternatively select
individual configurations by the exact labels in `--help`, for example
`--configs baseline loss=asymmetric_msle loss=huber_log`.
The default device is CPU; `--device cuda:0` explicitly requests an available
CUDA device. `--dev` restricts the source population and must be disclosed.
A demo/dev cohort may have empty splits or too few long-stay patients; the
runner reports an error instead of manufacturing scores.

## Prespecified protocol and departures from the paper

1. Use the existing YAIB cohort exclusions and 48 hourly dynamic features plus
   four static features. Keep ICU stays longer than 24 hours. Cohort eligibility
   is retrospective, including YAIB's observation-gap rule over the stay.
   This is not the paper's original feature set; the runner supplies neither
   diagnosis embeddings nor extra clock/time inputs.
2. Observe bins 0 through 23 only (the first 24 hours). Predict at hour 24.
   The scalar target is `(actual ICU duration in hours - 24) / 24` days,
   clipped to 30 minutes–100 days. Labels are not normalized or capped at the
   YAIB seven-day horizon. This differs from training every hour in the paper.
3. Split by patient before fitting processors: 70% training, 15% validation,
   15% test, independently for seeds 0/1/2. Multiple stays for a patient remain
   in one partition. Within each seed, all configurations share the same split.
4. Fit per-feature z-score statistics on training observations only. Preserve
   the original measurement mask, forward-fill standardized values causally,
   and fill values before their first observation with zero (training mean).
   Static missing entries also become zero after training-only normalization.
5. Use the existing compute-constrained baseline: N=3, k=4, Y=Z=4, final width
   16, main dropout=0.2, temporal dropout=0.05, official dilation, MSLE, Adam
   learning rate 0.0023, batch size 32, and eight epochs. These are **not**
   the paper's full best-performing hyperparameters.
6. Select the epoch with minimum validation MSLE for every loss variant;
   never select on test performance. Evaluate each selected checkpoint once
   on the held-out test patients. This is a fixed protocol, not clinical tuning.

One-factor sweeps cover layers 1/3/5/9, kernels 2–5, each integer temporal and
pointwise width 4–16, dropout 0/0.2/0.45, and learning rates
0.001/0.0023/0.01. The unchanged value is represented by the baseline row.
Extensions cover asymmetric MSLE (underprediction weight 2), Huber log-loss
(delta 1), constant/linear/exponential dilation, decay 0.5/0.75/0.9, and lab
no-forward-fill. Lab-only features are selected from existing extraction
definitions; mixed-source features retain forward-fill.

Linear is identical to the paper schedule and is not an independent novelty.
MSE vs. MSLE is a **Sanity Check**, not novel credit. Dataset-size comparisons
remain a **Compute-Constraint Demonstration** in the synthetic script, not a
clinical novelty claim. Standard BatchNorm uses running statistics at inference;
training statistics pool over time, so strict causality applies in evaluation.
There is no mortality head or variable-length all-hour training objective.
Official-model numerical parity has not been run. The official repository is
MIT-licensed; any future source adaptation must retain its copyright/license
notices.

## Metrics and saved artifacts

MAD and bias use days; MAPE uses percent with a four-hour denominator floor;
MSLE uses log(days), not log1p; R2 compares squared errors with target variance.
Bias is `mean(predicted - true)`, so negative values mean underprediction.
Long-stay MAD/bias use true remaining LoS >=5 days. The runner requires target
variation and at least one such held-out stay; absent subgroups are errors, not
silently assigned scores. LongN reports subgroup counts per seed.

A future run saves model checkpoints, per-seed normalization metadata, package
versions, split counts, validation selection information, per-seed aggregate
test metrics, and complete-seed mean/std tables. Incomplete rows remain PENDING.
Only a fully successful run receives `COMPLETED` status. Interrupted runs are
not complete results. The source dataset and processed cache must be retained
locally for reproduction; automatic checkpoint resume is not implemented.

Output/cache directories are under ignored `output/` by default. Do not commit
clinical datasets, processed patient records, checkpoints, or credentials.
Review aggregate outputs for appropriate disclosure before publication.

## Existing synthetic evidence is separate

`examples/synthetic_icu_los_tpc.py` already contains measured three-seed
synthetic results. They are retained as software/ablation feasibility evidence,
**not** inserted into this clinical table and **not** compared numerically with
another team's clinical scores. No new experiments are required to prepare
the clinical placeholders.
