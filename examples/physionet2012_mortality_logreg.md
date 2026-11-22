# PhysioNet 2012 in-hospital mortality: task-variation ablation

Results of `physionet2012_mortality_logreg.py`. It uses the existing
`PhysioNet2012Dataset` and the new `PhysioNet2012InHospitalMortality` task (input layout of
[TransEHR](https://proceedings.mlr.press/v225/xu23a.html), Xu et al., 2023), and varies
only the task options. The model is a class-weighted logistic regression on per-stay
summaries (per-variable mean, last value, observed fraction, static features), so
differences come from the task, not from model capacity.

## Setup

- One seeded (0), outcome-stratified, patient-level split, 64 / 16 / 20 %; the 20 % test
  part is scored (2,398 samples; 357 for the ICU-type-1 cohort). Trained with the PyHealth `Trainer` (Adam, lr 1e-3, batch 128, 20 epochs, seed 0); the\n  epoch is selected on validation AUROC.
- Normalization `(x - median) / (p95 - p5)`, fitted on training patients only.
- Single run, single split: no spread is available, so small differences are not evidence.

## Results (test)

| Variant (task options) | AUROC | AUPRC |
|---|---|---|
| all 36 variables, 48 h | 83.2 | 51.1 |
| `feature_groups=["vitals"]` | 74.4 | 31.8 |
| `feature_groups=["labs"]` | 82.6 | 50.2 |
| `observation_hours=24` | 80.8 | 43.6 |
| `clip=10` | 83.3 | 51.4 |
| `icu_types=[1]` (smaller cohort) | 73.9 | 39.8 |

## Reading

- Labs carry most of the signal; vitals alone are much weaker.
- Halving the window costs about 2.4 AUROC and 7.5 AUPRC points, so the late hours matter.
- Clipping at 10 scale units changes nothing measurable.
- The ICU-type-1 row is a different, smaller test cohort and is not comparable with the
  others.
