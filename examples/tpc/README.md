# TPC reproduction: MIMIC-IV 2.2 demo results

**Status: the MIMIC-IV Clinical Database Demo 2.2 baseline and full
three-seed sweep have RUN locally. Full-dataset eICU/MIMIC-IV experiments
remain PENDING.** These demo measurements do not reproduce the paper's
clinical benchmark scores. No dataset, patient-level output, or trained
checkpoint is included in the contribution. No push has been made for this
results update.

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

The MIMIC-IV demo path has now run end to end on actual local source files.
The eICU path and full MIMIC-IV population remain unverified. Tests still use
only synthetic fixtures, never the demo dataset.

## Measured MIMIC-IV demo setup

Run date: 2026-11-11. Source:
[MIMIC-IV Clinical Database Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/),
the publicly available 100-patient subset, not the full credentialed database.
The seven source/join files used by the runner match the supplied SHA256SUMS.
The source contains 100 patients and 140 ICU stays; the existing YAIB cohort
rules plus the day-one eligibility rule retain **88 patients / 117 stays**.

The prespecified settings below were unchanged: eight epochs, batch size 32,
seeds 0/1/2, patient splits before normalization, and validation-MSLE checkpoint
selection. `--dev` was **not** used. Implementation: commit `0901bd9d`.

| Seed | Train/validation/test patients | Train/validation/test stays | Baseline selected epoch | Test long-stay count |
| --- | --- | --- | ---: | ---: |
| 0 | 61 / 13 / 14 | 85 / 14 / 18 | 3 | 3 |
| 1 | 61 / 13 / 14 | 80 / 17 / 20 | 4 | 3 |
| 2 | 61 / 13 / 14 | 75 / 21 / 21 | 3 | 4 |

Patient partitions have no overlap within any seed. Observed standardized
training-feature means were checked (maximum absolute residual <0.000003).
All 45 labelled configurations completed all three seeds (135 trainings;
paper and linear are identical aliases). A separate three-seed baseline
reproduced the sweep baseline's metrics and selected epochs exactly.

Hardware/software: Apple M4 Pro, 24 GiB RAM, macOS arm64, one PyTorch CPU
thread; Python 3.13.11, PyHealth 2.1.0, PyTorch 2.15.0, NumPy 2.3.5.
The complete sweep took **149.51 seconds wall time** with already-built demo
preprocessing caches; that timing excludes initial dataset extraction.

## Measured demo results — all 45 configurations

Exact generated table from `output/tpc_mimic4_demo22_sweep/results.md`.
Values are mean +/- **sample** standard deviation over seeds 0/1/2.
MAD, Bias, LongMAD, and LongBias are in days; MAPE is percent.
LongN lists >=5-day remaining-LoS subgroup counts for each seed.
Different seeds change both initialization and patient partition, so this
spread is not isolated training-seed uncertainty. Test cohorts across seeds
can overlap; these are not three independent clinical validation cohorts.

| Configuration | MAD | MAPE | MSLE | R2 | Bias | LongMAD | LongBias | LongN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 2.418 +/- 0.440 | 152.585 +/- 45.901 | 1.982 +/- 0.076 | -0.309 +/- 0.145 | -1.918 +/- 0.655 | 8.565 +/- 1.620 | -8.565 +/- 1.620 | 3/3/4 |
| layers=1 | 2.438 +/- 0.494 | 167.833 +/- 52.958 | 2.186 +/- 0.474 | -0.264 +/- 0.114 | -1.870 +/- 0.781 | 8.379 +/- 1.643 | -8.379 +/- 1.643 | 3/3/4 |
| layers=5 | 2.464 +/- 0.547 | 151.200 +/- 46.072 | 2.101 +/- 0.355 | -0.309 +/- 0.129 | -1.979 +/- 0.822 | 8.581 +/- 1.751 | -8.581 +/- 1.751 | 3/3/4 |
| layers=9 | 2.428 +/- 0.476 | 142.696 +/- 19.957 | 1.978 +/- 0.135 | -0.314 +/- 0.057 | -1.992 +/- 0.608 | 8.601 +/- 1.482 | -8.601 +/- 1.482 | 3/3/4 |
| kernel=2 | 2.451 +/- 0.467 | 140.540 +/- 9.943 | 1.976 +/- 0.093 | -0.331 +/- 0.101 | -1.975 +/- 0.465 | 8.565 +/- 1.412 | -8.565 +/- 1.412 | 3/3/4 |
| kernel=3 | 2.443 +/- 0.485 | 131.859 +/- 20.821 | 1.970 +/- 0.196 | -0.333 +/- 0.122 | -2.057 +/- 0.611 | 8.562 +/- 1.580 | -8.562 +/- 1.580 | 3/3/4 |
| kernel=5 | 2.417 +/- 0.396 | 138.007 +/- 29.912 | 1.958 +/- 0.032 | -0.296 +/- 0.095 | -2.009 +/- 0.566 | 8.470 +/- 1.220 | -8.470 +/- 1.220 | 3/3/4 |
| temporal=5 | 3.242 +/- 1.723 | 185.507 +/- 67.470 | 2.182 +/- 0.262 | -2.520 +/- 3.726 | -1.220 +/- 1.030 | 8.606 +/- 1.438 | -8.606 +/- 1.438 | 3/3/4 |
| temporal=6 | 2.485 +/- 0.469 | 131.882 +/- 31.131 | 2.104 +/- 0.203 | -0.339 +/- 0.069 | -2.108 +/- 0.649 | 8.596 +/- 1.417 | -8.596 +/- 1.417 | 3/3/4 |
| temporal=7 | 3.984 +/- 3.122 | 250.651 +/- 110.261 | 2.253 +/- 0.582 | -8.524 +/- 14.334 | -0.328 +/- 2.017 | 8.463 +/- 1.663 | -8.463 +/- 1.663 | 3/3/4 |
| temporal=8 | 2.447 +/- 0.468 | 172.283 +/- 35.416 | 1.967 +/- 0.033 | -0.278 +/- 0.055 | -1.763 +/- 0.514 | 8.434 +/- 1.511 | -8.434 +/- 1.511 | 3/3/4 |
| temporal=9 | 2.491 +/- 0.575 | 139.582 +/- 45.734 | 2.199 +/- 0.484 | -0.346 +/- 0.197 | -2.076 +/- 0.831 | 8.630 +/- 1.845 | -8.630 +/- 1.845 | 3/3/4 |
| temporal=10 | 2.404 +/- 0.395 | 140.444 +/- 17.809 | 1.901 +/- 0.128 | -0.300 +/- 0.028 | -1.950 +/- 0.451 | 8.561 +/- 1.381 | -8.561 +/- 1.381 | 3/3/4 |
| temporal=11 | 2.460 +/- 0.435 | 153.314 +/- 15.072 | 2.017 +/- 0.084 | -0.320 +/- 0.060 | -1.946 +/- 0.490 | 8.567 +/- 1.437 | -8.567 +/- 1.437 | 3/3/4 |
| temporal=12 | 2.490 +/- 0.490 | 123.194 +/- 13.024 | 2.098 +/- 0.268 | -0.382 +/- 0.059 | -2.150 +/- 0.581 | 8.785 +/- 1.603 | -8.785 +/- 1.603 | 3/3/4 |
| temporal=13 | 2.335 +/- 0.584 | 129.781 +/- 22.903 | 1.834 +/- 0.434 | -0.248 +/- 0.153 | -1.938 +/- 0.730 | 8.378 +/- 1.962 | -8.378 +/- 1.962 | 3/3/4 |
| temporal=14 | 2.469 +/- 0.425 | 143.798 +/- 28.479 | 2.061 +/- 0.127 | -0.355 +/- 0.133 | -2.003 +/- 0.539 | 8.666 +/- 1.504 | -8.666 +/- 1.504 | 3/3/4 |
| temporal=15 | 2.459 +/- 0.445 | 165.022 +/- 49.188 | 2.072 +/- 0.164 | -0.300 +/- 0.107 | -1.885 +/- 0.684 | 8.515 +/- 1.600 | -8.515 +/- 1.600 | 3/3/4 |
| temporal=16 | 2.452 +/- 0.576 | 130.998 +/- 15.275 | 1.968 +/- 0.267 | -0.333 +/- 0.068 | -1.971 +/- 0.521 | 8.657 +/- 1.622 | -8.657 +/- 1.622 | 3/3/4 |
| pointwise=5 | 2.468 +/- 0.484 | 143.082 +/- 34.354 | 2.040 +/- 0.159 | -0.330 +/- 0.101 | -2.034 +/- 0.672 | 8.597 +/- 1.537 | -8.597 +/- 1.537 | 3/3/4 |
| pointwise=6 | 2.858 +/- 1.304 | 172.642 +/- 31.620 | 1.939 +/- 0.427 | -1.062 +/- 1.490 | -1.399 +/- 0.400 | 8.287 +/- 1.712 | -8.287 +/- 1.712 | 3/3/4 |
| pointwise=7 | 2.384 +/- 0.585 | 134.170 +/- 21.247 | 1.879 +/- 0.428 | -0.258 +/- 0.222 | -1.940 +/- 0.749 | 8.356 +/- 2.053 | -8.356 +/- 2.053 | 3/3/4 |
| pointwise=8 | 2.645 +/- 0.747 | 150.267 +/- 34.170 | 2.140 +/- 0.283 | -0.456 +/- 0.254 | -1.860 +/- 0.475 | 8.636 +/- 1.661 | -8.636 +/- 1.661 | 3/3/4 |
| pointwise=9 | 2.417 +/- 0.520 | 143.465 +/- 32.848 | 2.078 +/- 0.593 | -0.267 +/- 0.130 | -1.934 +/- 0.728 | 8.379 +/- 1.788 | -8.379 +/- 1.788 | 3/3/4 |
| pointwise=10 | 3.303 +/- 1.923 | 220.223 +/- 53.345 | 2.183 +/- 0.410 | -2.840 +/- 4.499 | -0.965 +/- 0.893 | 8.394 +/- 1.630 | -8.394 +/- 1.630 | 3/3/4 |
| pointwise=11 | 2.482 +/- 0.475 | 155.336 +/- 56.310 | 2.014 +/- 0.044 | -0.312 +/- 0.152 | -1.914 +/- 0.670 | 8.507 +/- 1.554 | -8.507 +/- 1.554 | 3/3/4 |
| pointwise=12 | 2.476 +/- 0.470 | 139.955 +/- 32.476 | 2.045 +/- 0.220 | -0.315 +/- 0.132 | -2.014 +/- 0.735 | 8.507 +/- 1.462 | -8.507 +/- 1.462 | 3/3/4 |
| pointwise=13 | 2.500 +/- 0.530 | 126.361 +/- 3.834 | 2.002 +/- 0.211 | -0.364 +/- 0.120 | -2.044 +/- 0.385 | 8.611 +/- 1.483 | -8.611 +/- 1.483 | 3/3/4 |
| pointwise=14 | 2.419 +/- 0.418 | 141.313 +/- 3.359 | 1.954 +/- 0.117 | -0.324 +/- 0.107 | -1.956 +/- 0.397 | 8.498 +/- 1.458 | -8.498 +/- 1.458 | 3/3/4 |
| pointwise=15 | 2.803 +/- 1.135 | 165.440 +/- 25.445 | 2.060 +/- 0.395 | -0.806 +/- 0.928 | -1.592 +/- 0.331 | 8.494 +/- 1.677 | -8.494 +/- 1.677 | 3/3/4 |
| pointwise=16 | 2.387 +/- 0.549 | 144.251 +/- 30.082 | 1.854 +/- 0.326 | -0.240 +/- 0.244 | -1.905 +/- 0.770 | 8.243 +/- 1.911 | -8.243 +/- 1.911 | 3/3/4 |
| dropout=0.0 | 2.414 +/- 0.481 | 146.606 +/- 28.320 | 1.948 +/- 0.250 | -0.291 +/- 0.192 | -1.919 +/- 0.669 | 8.485 +/- 1.757 | -8.485 +/- 1.757 | 3/3/4 |
| dropout=0.45 | 2.445 +/- 0.470 | 149.664 +/- 35.193 | 2.018 +/- 0.156 | -0.317 +/- 0.147 | -1.971 +/- 0.649 | 8.571 +/- 1.578 | -8.571 +/- 1.578 | 3/3/4 |
| lr=0.001 | 2.432 +/- 0.469 | 153.752 +/- 46.710 | 2.040 +/- 0.152 | -0.315 +/- 0.165 | -1.926 +/- 0.694 | 8.592 +/- 1.692 | -8.592 +/- 1.692 | 3/3/4 |
| lr=0.01 | 2.412 +/- 0.434 | 157.043 +/- 54.772 | 1.958 +/- 0.060 | -0.289 +/- 0.123 | -1.893 +/- 0.692 | 8.485 +/- 1.519 | -8.485 +/- 1.519 | 3/3/4 |
| dilation=paper | 2.414 +/- 0.475 | 149.732 +/- 32.894 | 1.964 +/- 0.182 | -0.298 +/- 0.163 | -1.917 +/- 0.654 | 8.512 +/- 1.712 | -8.512 +/- 1.712 | 3/3/4 |
| dilation=constant | 2.420 +/- 0.475 | 152.774 +/- 40.920 | 1.944 +/- 0.173 | -0.286 +/- 0.138 | -1.814 +/- 0.685 | 8.482 +/- 1.712 | -8.482 +/- 1.712 | 3/3/4 |
| dilation=linear | 2.414 +/- 0.475 | 149.732 +/- 32.894 | 1.964 +/- 0.182 | -0.298 +/- 0.163 | -1.917 +/- 0.654 | 8.512 +/- 1.712 | -8.512 +/- 1.712 | 3/3/4 |
| dilation=exponential | 2.428 +/- 0.470 | 151.983 +/- 38.404 | 2.009 +/- 0.163 | -0.315 +/- 0.163 | -1.939 +/- 0.651 | 8.579 +/- 1.663 | -8.579 +/- 1.663 | 3/3/4 |
| loss=asymmetric_msle | 2.429 +/- 0.426 | 159.959 +/- 51.447 | 2.003 +/- 0.078 | -0.303 +/- 0.167 | -1.880 +/- 0.683 | 8.515 +/- 1.653 | -8.515 +/- 1.653 | 3/3/4 |
| loss=huber_log | 2.421 +/- 0.432 | 157.031 +/- 50.108 | 1.994 +/- 0.061 | -0.306 +/- 0.153 | -1.890 +/- 0.671 | 8.556 +/- 1.626 | -8.556 +/- 1.626 | 3/3/4 |
| Sanity Check: MSE | 2.433 +/- 0.424 | 167.999 +/- 66.739 | 2.145 +/- 0.258 | -0.272 +/- 0.156 | -1.848 +/- 0.743 | 8.420 +/- 1.552 | -8.420 +/- 1.552 | 3/3/4 |
| decay=0.5 | 2.433 +/- 0.447 | 150.689 +/- 45.305 | 1.995 +/- 0.091 | -0.315 +/- 0.146 | -1.948 +/- 0.669 | 8.585 +/- 1.627 | -8.585 +/- 1.627 | 3/3/4 |
| decay=0.9 | 2.416 +/- 0.481 | 148.019 +/- 31.425 | 1.930 +/- 0.248 | -0.282 +/- 0.209 | -1.910 +/- 0.689 | 8.436 +/- 1.806 | -8.436 +/- 1.806 | 3/3/4 |
| lab no-forward-fill | 2.455 +/- 0.464 | 142.007 +/- 23.042 | 1.982 +/- 0.264 | -0.320 +/- 0.181 | -1.995 +/- 0.627 | 8.529 +/- 1.648 | -8.529 +/- 1.648 | 3/3/4 |

### Findings and limits

Baseline MAD is **2.418 +/- 0.440 days**, MSLE **1.982 +/- 0.076**, and R2
**-0.309 +/- 0.145**. Negative R2 means worse squared error than the reference
constant equal to each held-out cohort's true mean; that is the R2 definition,
not a separately fitted deployable baseline. Bias **-1.918 +/- 0.655 days**
shows underprediction. Long-stay MAD **8.565 +/- 1.620 days**, based on only
3/3/4 long stays, demonstrates poor performance on that tiny subgroup.
MAPE can exceed 100% because even moderate errors on short remaining stays
have large relative magnitude, despite the four-hour denominator floor.

Asymmetric MSLE slightly reduces negative bias (-1.880 vs. -1.918) but does
not improve MAD or MSLE here; its synthetic improvement did not clearly
transfer. Huber is also similar to baseline. Decay 0.9 and constant dilation
have somewhat lower mean MSLE, but differences are small relative to seed
spread. No-forward-fill slightly worsens MAD. Wider variants occasionally
have very large errors (notably temporal=7 and pointwise=10); those runs were
retained, not discarded.

Temporal width 13 has the lowest **observed** mean test MSLE (1.834 +/- 0.434)
and MAD (2.335 +/- 0.584) in this sweep. This is a descriptive comparison
across 45 test-evaluated rows, **not** independently validated model selection
or evidence of significant improvement. Every row still has negative mean R2.
No hyperparameters were retuned after inspecting test performance.

This is a small-demo, eight-epoch, day-one adaptation with different features,
cohort rules, and training objective from the paper. These limitations preclude
a like-for-like comparison with the paper's full-cohort/all-hour scores or
another team's full-eICU results. No claim is made that the demo reproduces
the reported clinical performance. Full clinical replication remains PENDING.

## Reproduce the measured demo sweep

With the already-downloaded demo folder in the repository root:

```bash
python -m examples.eicu_los_tpc --run --dataset mimic4 \
  --root mimic-iv-clinical-database-demo-2.2 --data-version demo-2.2 \
  --all-configs --seeds 0 1 2 --epochs 8 --batch-size 32 \
  --output-dir output/tpc_mimic4_demo22_sweep \
  --cache-dir output/tpc_mimic4_demo22_cache
```

The separate baseline used the same command with `--configs baseline` instead
of `--all-configs`, and `--output-dir output/tpc_mimic4_demo22_baseline`.
The original directories already exist; use a **new output directory** for a
repeat run. Data checksums/provenance and split-check summaries are local at
`output/tpc_mimic4_demo22_provenance.json` and
`output/tpc_mimic4_demo22_split_checks.json`. No source data or checkpoints
should be committed. The source demo remains governed by its supplied ODbL
license; this table is aggregate model-evaluation output with source attribution.

## Full-dataset clinical work — still pending

| Deliverable | eICU full dataset | MIMIC-IV full dataset |
| --- | --- | --- |
| Authorized local data / cohort extraction | PENDING | PENDING |
| Baseline and extension sweeps | PENDING | PENDING |
| Benchmark-comparable results and analysis | PENDING | PENDING |

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

Each run saves model checkpoints, per-seed normalization metadata, package
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
**not** inserted into the measured demo table and **not** compared numerically
with another team's clinical scores.
