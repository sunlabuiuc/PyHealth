# Add an hourly remaining-ICU-LOS task with missingness experiments

**Contributors:**
- Person C: Xiao Luo — xiaoluo6
- Person A: ____________________
- Person B: ____________________

Fill the A/B fields before posting; their identities were not supplied.

**Contribution type:** Standalone task for the existing eICUDataset (Option 3).

**Paper:** Rocheteau, Liò and Hyland, *Temporal Pointwise Convolutional Networks
for Length of Stay Prediction in the Intensive Care Unit* (CHIL 2021).
https://arxiv.org/abs/2007.09483

**Description:** Add hourly remaining-LOS labels from hour 5 through the first
14 days of an adult ICU stay. Reuse YAIB measurement/static extraction with
project-specific eligibility, reproducible removal of hourly observations,
and optional recency-decay inputs. Completed-bin timestamps prevent
within-hour look-ahead. Synthetic examples train PyHealth's existing LSTM;
this contribution does not implement the TPC model.

**Files to review:**
- pyhealth/tasks/tpc_eicu.py: task, cohort, label generation, masking and decay.
- pyhealth/tasks/__init__.py: public import.
- docs/api/tasks/pyhealth.tasks.tpc_eicu.rst and docs/api/tasks.rst: API/index.
- tests/core/test_tpc_eicu.py and tpc_fixtures.py: fast synthetic coverage.
- tests/integration/test_tpc_checkpoint.py: opt-in actual CSV-to-LSTM check.
- examples/eicu_hourly_los_lstm.py: six-condition synthetic example.
- project/person_c/generate_synthetic.py and checkpoint.py: shared example
  support; curate this directory with the team before upstream submission.

**Validation:** See validation.txt and checkpoint_result.json. Run:
```
python -m pytest tests/core/test_tpc_eicu.py -q --durations=5
PYHEALTH_RUN_TPC_INTEGRATION=1 python -m pytest tests/integration/test_tpc_checkpoint.py -q
python -m examples.eicu_hourly_los_lstm --output /tmp/tpc_ablation.json
```

**Results:** Synthetic example results are recorded in synthetic_ablation.json
and SYNTHETIC_RESULTS.md, solely to demonstrate executable comparisons.
Insert A's actual eICU/demo experiment setup and findings before final submission.
Do not claim synthetic metrics reproduce the paper or establish robustness.

**Reproduction differences:** Existing LSTM rather than TPC; YAIB feature set;
median/floor bins with completion-time timestamps; hourly-cell removal; training
z-score imputation/normalization; default MSE rather than paper MSLE. Targets
and activated predictions are in hours; MAD conversion uses days. Example
log1p-MSE is clearly labeled and is not the paper's logarithmic metric.

**Before posting:** fill contributors; include final evidence; verify upstream
main/master discrepancy; rebase; enable maintainer edits; ensure one team PR;
record PR link in Gradescope with all group members. No PR has been posted.
