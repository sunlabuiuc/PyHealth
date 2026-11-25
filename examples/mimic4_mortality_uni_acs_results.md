# Uni-ACS mortality scoring results

Run `python examples/mimic4_mortality_uni_acs.py` from the repository root for the
synthetic ablation. Only the optional real-data runs require MIMIC-IV access.
All results below are aggregate metrics from local runs. The reported novel
binning results used `csaps`. The SciPy fallback can give different breakpoints.

## Synthetic ablation

The seeded example (seed 42) uses 1,500 synthetic admissions with 15 features,
split into 900 score-fitting, 300 calibration, and 300 test samples. AUROC and
AUPRC use raw integer scores. The last two columns compare Brier scores before
and after held-out isotonic calibration.

| Method | Top K | Score categories | AUROC | AUPRC | Brier, approximate | Brier, calibrated |
|---|---:|---:|---:|---:|---:|---:|
| Quantile | 3 | 12 | 0.7872 | 0.5399 | 0.1617 | 0.1661 |
| Quantile | 5 | 20 | 0.8240 | 0.6074 | 0.1504 | 0.1499 |
| Quantile | 10 | 40 | 0.8450 | 0.7067 | 0.1367 | 0.1395 |
| Quantile | 15 | 60 | 0.8450 | 0.7067 | 0.1367 | 0.1395 |
| Novel | 3 | 6 | 0.7394 | 0.4603 | 0.1725 | 0.1725 |
| Novel | 5 | 10 | 0.7545 | 0.5181 | 0.1682 | 0.1706 |
| Novel | 10 | 20 | 0.7783 | 0.5919 | 0.1614 | 0.1685 |
| Novel | 15 | 25 | 0.7783 | 0.5919 | 0.1614 | 0.1685 |

Top 10 and Top 15 had identical discrimination within each method on this
synthetic test set. Novel binning used fewer categories and had lower AUROC
than quantile binning at each feature count.

## Reproduction on the authors' preprocessed cohort

[`uni_acs_mimic_reproduction.py`](uni_acs_mimic_reproduction.py) used 98 features,
5,383 training admissions, and 2,307 test admissions. These are the results
when the full training split was used for score fitting.

| Model | Score categories | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|
| Calibrated RandomForest | | 0.9009 | 0.4405 | 0.0484 |
| Uni-ACS Novel Top 5 | 9 | 0.8134 | 0.2023 | 0.0612 |
| Uni-ACS Novel Top 10 | 21 | 0.8406 | 0.2785 | 0.0597 |
| Uni-ACS Novel Top 15 | 29 | 0.8552 | 0.3210 | 0.0591 |

Uni-ACS Brier scores above use approximate probabilities. In a separate run
with 4,306 score-fitting and 1,077 calibration admissions, Top 15 Brier improved
from 0.0588 to 0.0529 after held-out isotonic calibration on the same test set.

## MIMIC-IV 3.1 sepsis extension

The ICD-defined sepsis cohort uses age, sex, and 12 lab features. Top 14 is the
all-features configuration. The [stratified random-split experiment](uni_acs_mimic31_model.py)
used 12,202 training and 5,230 test admissions. This baseline filters labs by
`charttime` and may include results released after the cutoff. Calibrated
RandomForest achieved
AUROC 0.7235. Novel Uni-ACS AUROC was 0.6862, 0.6958, and 0.6975 at Top 5, 10, and 14.

The [temporal experiment](uni_acs_mimic31_temporal_validation.py) used labs available
by 24 hours after admission, using both `charttime` and `storetime`.
It trained the random forest on 8,593 historical admissions and used a sample
of 2,000 from that training set for SHAP and score fitting. It evaluated 3,683
historical and 2,724 recent admissions. The estimated admission ranges
were 2008–2019 and 2020–2022, respectively. Another 2,432 admissions were excluded
because their estimated ranges crossed a boundary or extended outside the study
years. Test mortality was 18.49% historically and 21.66% in the recent group.

| Model | Historical AUROC / AUPRC / Brier | Recent AUROC / AUPRC / Brier |
|---|---|---|
| Calibrated RandomForest | 0.7391 / 0.4104 / 0.1327 | 0.7326 / 0.4549 / 0.1491 |
| Uni-ACS Novel Top 10 | 0.7021 / 0.3389 / 0.1399 | 0.7067 / 0.4004 / 0.1571 |
| Uni-ACS Quantile Top 10 | 0.7074 / 0.3406 / 0.1395 | 0.7176 / 0.4117 / 0.1557 |

Uni-ACS Brier scores in the temporal experiment use approximate probabilities.
The two real-data cohorts differ in eligibility, features, and evaluation splits.
