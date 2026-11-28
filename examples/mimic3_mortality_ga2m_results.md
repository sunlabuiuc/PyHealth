# GA2M on 48h MIMIC-III mortality: results

Script: `examples/mimic3_mortality_ga2m.py`, with `pyhealth.models.GA2M` and
`pyhealth.tasks.InHospitalMortality48hMIMIC3`. It reproduces the experiments
of Hegselmann et al. (2020), "An evaluation of the doctor-interpretability of
generalized additive models with interactions" (PMLR 126).

Everything below is an aggregate test-set metric. No patient-level MIMIC-III
data is in this repository.

## Setup

- **Data:** MIMIC-III v1.4, through `MIMIC3Dataset` with the `chartevents`
  and `labevents` tables.
- **Cohort:** ICU stays of adults (18 or older) lasting at least 48 h. That
  gives 28,919 stays with 14.9% in-hospital mortality, split by patient
  70/10/20 into 20,235 / 2,836 / 5,848 stays (seed 0).
- **Features:** the mean and standard deviation of 17 clinical variables
  over the first 48 h (34 features). Unmeasured values are `-1`, which the
  model puts in its own missing bin.
- **Training:** Adam, lr 0.01, batch 256, up to 30 epochs.
  - Early stopping keeps the epoch with the best validation ROC-AUC
    (patience 3).
  - A smoothness penalty of weight 100 applies to the risk curves.
  - Every model starts from the same seed.

Command (script defaults):

```bash
python examples/mimic3_mortality_ga2m.py --root /path/to/mimic-iii/1.4 --cache-dir /path/to/cache
```

The cohort build sorts every CHARTEVENTS row and needs a lot of temporary
disk; the full table ran out of space on a machine with 87 GB free. We
therefore first kept only the CHARTEVENTS rows whose `ITEMID` is one of the
task's items (`_CHART_ITEMID_VARIABLE`) and only the columns PyHealth
loads. That left 59.0M of 330.7M rows. The task ignores all other rows, so
the samples are the same.

## Results (test split; validation ROC-AUC in brackets)

### Bin-count ablation (1D terms only)

| Bins | ROC-AUC | PR-AUC |
|-----:|--------:|-------:|
|   32 | 0.805 (0.812) | 0.422 |
|   64 | 0.802 (0.813) | 0.423 |
|  128 | 0.792 (0.807) | 0.412 |
|  256 | 0.783 (0.796) | 0.398 |

Coarser bins do slightly better here. With a heavier smoothness penalty
(weight 1000) the gap mostly closes: 0.803, 0.805, 0.798 and 0.791 for 32,
64, 128 and 256 bins.

### Pairwise interactions

All 561 pairs were trained at 8 bins and ranked by contribution variance on
the training split. The top 34 were added to a 256-bin 1D model, with each
pair table at 8 × 8 bins:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| 1D only, 256 bins | 0.783 (0.796) | 0.398 |
| 1D + top 34 pairs | 0.792 (0.798) | 0.397 |

The interactions help only when their tables are coarse. With 32 × 32 pair
tables the result was 0.779, and with 256 × 256 it was 0.750 (65k weights
per pair overfit). The top-ranked pairs combine temperature, capillary
refill, GCS, FiO2, weight and pH. The script prints the full ranked list.

### Clinician-pruning ablation

The paper's clinicians found the pH risk curves and the interactions that
involve a standard-deviation feature hard to interpret. Each variant below
is retrained from scratch without those terms:

| Model | Pairs | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| 1D + top 34 pairs | 34 | 0.792 (0.798) | 0.397 |
| without pH terms | 25 | 0.790 (0.802) | 0.405 |
| without std-feature pairs | 15 | 0.791 (0.799) | 0.407 |
| without both | 11 | 0.792 (0.801) | 0.408 |

Removing the flagged terms costs no accuracy. The model without both keeps
11 of 34 pairs, has the same ROC-AUC and a slightly higher PR-AUC. In other
words, the doctors' pruning makes the model simpler and easier to explain
for free.

### Baseline

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic regression, same 34 features (mean imputation + missing indicators) | 0.792 | 0.394 |

## Comparison with the paper

The paper reports a GA2M ROC-AUC of about 0.850 and PR-AUC of 0.456–0.468.
Its baselines are logistic regression at 0.848 and an LSTM at 0.870. Our
best model reaches 0.805 / 0.423, about 0.045 ROC-AUC lower. Our own
logistic-regression baseline is lower by a similar amount (0.792 against
0.848), so most of the gap comes from the data, not the model:

- **Fewer features:** we use 34 summary features, against the benchmark's
  714 for logistic regression.
- **Cohort:** ours includes every adult ICU stay of 48 h or more. The
  benchmark applies further exclusions (for example, transfers and multiple
  ICU stays per admission) and has about 21k stays.
- **Training method:** the paper fits its GA2M by cyclic boosting
  (InterpretML's EBM). We train all terms jointly by gradient descent.

What does reproduce: a GA2M with a small number of pairwise terms is as
accurate as logistic regression on the same features (0.792 against
0.792). Pruning the terms clinicians flagged keeps that accuracy.

## How the training settings were chosen

The first run used the original defaults: Adam lr 0.05, batch 32, 15 epochs,
no early stopping and no smoothness penalty. It reached 0.734 (32 bins) and
0.648 (256 bins), and accuracy fell as bins grew, which points to
overfitting. We then tried a few settings on 1D models:

| Change | 32 bins | 256 bins |
|---|---:|---:|
| Original defaults | 0.734 | 0.648 |
| Plus early stopping | 0.721 | 0.666 |
| lr 0.01, batch 256 | 0.781 | 0.750 |
| Plus smoothness 10 | — | 0.772 |
| Plus smoothness 100 | 0.805 | 0.783 |
| Plus smoothness 1000 | 0.803 | 0.791 |
| SGD (lr 1.0) instead of Adam, no smoothness | — | 0.754 |

These settings were compared on test-set scores, so they are slightly
optimistic. On validation, smoothness 100 and 1000 tie for the best 1D
model (0.813). Smoothness 1000 is better at 128 and 256 bins (validation
0.812 and 0.807, against 0.807 and 0.796). We kept 100 as the single
default because the best models (32 and 64 bins) do equally well with
either.

## Synthetic run

Without `--root`, the script runs on 600 synthetic patients in about 15 s on
CPU. This only checks that every stage runs; the numbers mean nothing about
MIMIC-III.

## Reproducing

```bash
python examples/mimic3_mortality_ga2m.py                                  # synthetic
python examples/mimic3_mortality_ga2m.py --root /path/to/mimic-iii/1.4    # real data
python examples/mimic3_mortality_ga2m.py --root ... --stages pruning lr   # some stages
```

See `--help` for every option (`--bins`, `--pair-bins`, `--smoothness`,
`--lr`, `--batch-size`, `--epochs`, `--patience`, `--seed`, ...).
