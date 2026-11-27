# PhysioNet 2012 in-hospital mortality: task-variation ablation

Results of `physionet2012_mortality_logreg.py`. It uses the existing
`PhysioNet2012Dataset` and the new `PhysioNet2012InHospitalMortality` task (input layout of
[TransEHR](https://proceedings.mlr.press/v225/xu23a.html), Xu et al., 2023), and varies
only the task options. The model is PyHealth's `LogisticRegression` with no class weights. It
takes the hourly grid directly: the `measurements`, `observed` and `static` inputs are
each embedded (`--embedding-dim`, default 32) and the pooled embeddings feed one linear layer. It is
a small linear-on-embeddings model, so differences come mostly from the task, not from
model capacity.

## Setup

- One seeded (0), outcome-stratified, patient-level split, 64 / 16 / 20 %; the 20 % test
  part is scored (2,398 samples; 357 for the ICU-type-1 cohort). Trained with the PyHealth `Trainer` (Adam, lr 1e-3, batch 128, 20 epochs, seed 0, plain binary
  cross-entropy); the epoch is selected on validation AUROC.
- Normalization `(x - median) / (p95 - p5)`, fitted on training patients only.
- The 00:00 admission `Weight` is a patient descriptor: `PhysioNet2012Dataset` stores it in
  the patients table, so it is in neither the grid nor the static features. The `Weight`
  grid column holds only the weights measured during the stay. In set-a (4,000 stays) 68 %
  of stays have one in the first 48 hours (18 % in hour 0), and 51 % of the grid cells of
  the column are observed (HR: 90 %).
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
