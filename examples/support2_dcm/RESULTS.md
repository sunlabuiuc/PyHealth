# SUPPORT2 reproduction results

Five outer folds on all 9,105 SUPPORT2 records. Values are mean ± sample SD across folds (descriptive, not a confidence interval); the three mask seeds are averaged inside each fold. Lower Brier is better.

## Held-out test results (clean data)

| Model | Clean mean IPCW Brier | Mean IPCW concordance |
|---|---:|---:|
| cox | 0.182307 ± 0.003291 | 0.676488 ± 0.005022 |
| dcm | 0.173291 ± 0.002101 | 0.720084 ± 0.001426 |
| dcm_indicators | 0.170509 ± 0.002010 | 0.728432 ± 0.003017 |
| rsf | 0.171476 ± 0.002459 | 0.726054 ± 0.006345 |

`dcm_indicators` is the selected DCM architecture refit with eight laboratory missing-value indicators. The indicators and the masking experiment below are an extension; they are not in the paper.

## Validation search (hyperparameter ablation)

Each fold selects one configuration per model family by mean IPCW Brier on a 20% inner validation split of its training data; the held-out fold is never used for selection. Mean ± SD over the five folds; DCM epochs are the early-stopped epoch counts.

| Model | Configuration | Validation mean IPCW Brier | Mean epochs | Folds selected |
|---|---|---:|---:|---:|
| dcm | k=3, width=50 | 0.177488 ± 0.004335 | 43.6 | 0 |
| dcm | k=3, width=100 | 0.177332 ± 0.004680 | 34.0 | 2 |
| dcm | k=4, width=50 | 0.177347 ± 0.004730 | 45.2 | 1 |
| dcm | k=4, width=100 | 0.177377 ± 0.006342 | 39.4 | 0 |
| dcm | k=6, width=50 | 0.176150 ± 0.004966 | 42.6 | 0 |
| dcm | k=6, width=100 | 0.175692 ± 0.004294 | 31.6 | 2 |
| cox | alpha=0.001 | 0.183819 ± 0.002990 |  | 3 |
| cox | alpha=0.01 | 0.183819 ± 0.002990 |  | 0 |
| cox | alpha=0.1 | 0.183819 ± 0.002989 |  | 0 |
| cox | alpha=1.0 | 0.183820 ± 0.002985 |  | 0 |
| cox | alpha=10.0 | 0.183844 ± 0.002946 |  | 2 |
| cox | alpha=100.0 | 0.184449 ± 0.002735 |  | 0 |
| rsf | min_samples_leaf=5, max_features=sqrt | 0.172844 ± 0.003379 |  | 5 |
| rsf | min_samples_leaf=5, max_features=1.0 | 0.175862 ± 0.003224 |  | 0 |
| rsf | min_samples_leaf=10, max_features=sqrt | 0.173820 ± 0.003251 |  | 0 |
| rsf | min_samples_leaf=10, max_features=1.0 | 0.175165 ± 0.003359 |  | 0 |
| rsf | min_samples_leaf=20, max_features=sqrt | 0.175068 ± 0.003451 |  | 0 |
| rsf | min_samples_leaf=20, max_features=1.0 | 0.175825 ± 0.003633 |  | 0 |

## Paired masking extension

Additional MCAR masking of originally observed laboratory values in the test fold, without retraining. Primary comparison: indicator deterioration minus plain DCM deterioration (negative favors indicators). Descriptive over five fixed folds, not proof of benefit.

| Additional mask | Plain deterioration | Indicator deterioration | Paired difference ± fold SD |
|---|---:|---:|---:|
| 10% | 0.000172 | 0.001585 | 0.001414 ± 0.000933 |
| 30% | 0.000613 | 0.005274 | 0.004661 ± 0.002664 |

## Audit and limits

110 completed protocol fits; 0 failed search configurations; 30 DCM search fits emitted warnings (including upstream numerical and dependency warnings). Held-out evaluations had 0 out-of-bounds cells before the documented tiny projection. Each fold JSON records the exact warnings, epochs and settings; only these aggregates are included here.

The authors' spline/posterior numerical handling is part of this reproduction, so warnings limit claims of exact agreement with the paper. Only SUPPORT2 is reproduced, with Cox/RSF controls; FLCHAIN, SEER, subgroup fairness and the full published tables are not.

Calibration is a separate diagnostic: five equal-count prediction bins on held-out fold 0, Kaplan–Meier survival at its training-derived horizons. No calibration metric is substituted for Brier.
