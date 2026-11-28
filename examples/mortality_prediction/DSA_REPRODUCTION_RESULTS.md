# DSA Reproduction Results — MIMIC-III

## Evaluation Setup

- Task: 24-hour ICU decompensation prediction
- Dataset: MIMIC-III
- Test timesteps: 523,208
- ICU stays: 6,237
- Event stays: 490
- All four DSA variants evaluated on identical records
- DSA evaluation score: risk_h23
- Metric: Area Under the Precision-Recall Curve (AuPRC)

## Full-Cohort Test Results

| Model | Test AuPRC |
|---|---:|
| Vanilla DSA | 0.018762 |
| Bias Initialization | 0.215951 |
| Truncated Likelihood | 0.230058 |
| survTLS | 0.230893 |
| EEP Baseline | 0.233280 |

EEP uses its corresponding fixed-horizon prediction score.

These full-test-cohort metrics were recalculated from stored predictions.
They are distinct from the compute-constrained subset training results
reported in `mimic3_decompensation_dsa.py`. The synthetic PyHealth examples
validate the contributed implementation and do not independently reproduce
these MIMIC-III numbers.

## Reproduction Findings

- Bias initialization produced the largest improvement.
- Truncated likelihood further improved AuPRC.
- survTLS performed similarly to Truncated.
- EEP slightly outperformed survTLS at the fixed 24-hour horizon.
- The DSA-only ablation ordering is consistent with our earlier subset
  experiments; the relative ordering of EEP and survTLS differs.

## Evaluation Validation

- All four DSA prediction files contain 523,208 timesteps.
- All four DSA prediction files contain 6,237 ICU stays.
- Stay, time, and label fields match across DSA variants.
- All DSA AuPRC values were recalculated using risk_h23.

## Additional Clinical Evaluation

Validation-selected operating points were evaluated on the
independent test cohort.

At the 70% validation precision target:

| Metric | EEP | survTLS |
|---|---:|---:|
| Test alarm precision | 0.8029 | 0.7452 |
| Test event recall | 0.1633 | 0.2408 |
| Median lead time (hourly steps) | 5 | 4 |
| False alarms per 100 stays | 1.09 | 2.12 |

The event-level and clinical-extension comparisons above
were performed for EEP and survTLS, not all DSA variants.

## Limitations

- These results come from a specific reproduction setup
  and do not establish statistical significance.
- Direct numerical comparison with the original paper
  requires matching its evaluation protocol and cohort.
- MIMIC-III patient-level data and predictions are not
  included in this public-facing documentation.

## Evaluation Utility Consistency Check

The reusable early-event evaluation utility was compared against
the original event-level evaluation implementation using the
same MIMIC-III test predictions.

Six frozen validation-selected operating points were checked:
60%, 70%, and 80% validation precision targets for EEP and survTLS.

All six operating points matched across the compared event-level
metrics, including event recall, alarm precision, alarm counts,
detected events, and mean lead time.

This establishes consistency at the tested operating points,
not universal equivalence across all thresholds or datasets.
