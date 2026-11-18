# Measured synthetic example results

Run: `python -m examples.eicu_hourly_los_lstm --output /tmp/tpc_ablation.json`.
Five invented patients; one training stay, two held out; three optimizer
steps; fixed seed 42; identical test masks across trained models. These are
wiring checks, not clinical evidence or an estimate of generalization.

| Train removal | Test removal | MAD (days) | log1p-MSE (days) |
|---|---|---|---|
| 0% | 0% | 9.460001 | 5.254549 |
| 0% | 10% | 9.459250 | 5.251213 |
| 0% | 20% | 9.459991 | 5.254490 |
| 10% | 0% | 9.459687 | 5.253154 |
| 10% | 10% | 9.459093 | 5.250510 |
| 10% | 20% | 9.459677 | 5.253099 |

All six conditions produced 336 predictions and finite metrics. The errors
are large because the test set includes a 400.5-hour stay while the only
training stay lasts 12.5 hours, and training lasts only three updates. Tiny
differences here do not support the augmentation hypothesis. The long stay
dominates hourly aggregation; a real experiment should specify weighting,
patient-disjoint splits, repeated seeds, and paper-compatible metrics.

log1p-MSE is explicitly different from the original paper's log-MSE.
Predictions are converted from hours to days; negative predictions, if any,
are clipped only for log1p and counted in the JSON. MAD uses raw predictions.
A must replace these illustrative results with the team's real experiment
before drawing conclusions or comparing numerical scores to the paper.
