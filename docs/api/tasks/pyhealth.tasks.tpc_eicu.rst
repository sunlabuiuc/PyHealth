Hourly remaining ICU length of stay
==================================

.. automodule:: pyhealth.tasks.tpc_eicu
   :members:
   :show-inheritance:

Use ``TPCLengthOfStayEICU(missing_rate=0.1, seed=42)`` with ``eICUDataset``.
Load lab, vitalperiodic, vitalaperiodic, intakeoutput, respiratorycharting,
and nursecharting. Use ``set_task(task, split=...)`` to fit processors on
training patients only. Use RNN_V2 for sequence_v2 and hourly regression.

The cohort includes known adults (age >= 18) with ICU stay >= 5 hours.
The privacy-preserving age value ``> 89`` is encoded as 90. Missing age is
excluded. Unknown sex and invalid admission height/weight remain missing.

The label at hour h is total ICU stay in hours minus h. Prediction times
are 5 through min(floor(total stay hours), 336), inclusive. This explicitly
includes a zero label at exact discharge. The observation horizon is capped;
remaining LOS is not. Convert predictions and targets to days for comparison
with the paper. Never silently mix hourly and daily metrics.

This task reuses YAIB's 48 dynamic and 4 static features. It is a partial
reproduction of the task in https://arxiv.org/abs/2007.09483, not the full TPC
architecture or exact original feature set. Observation removal occurs after
hourly aggregation, before processor imputation. Forward filling and training
normalization are inherited; optional paper-formula decay is described below.

Run ``pytest tests/core/test_tpc_eicu.py`` for fast synthetic regression tests;
run ``pytest tests/integration/test_tpc_checkpoint.py`` for CSV-to-LSTM coverage.

Optional decay input
--------------------

Set ``include_decay=True`` to add a separate ``decay`` sequence with 48 values
per hour. It is zero before any observation, one when observed, and
``1 / (gap_hours * 4/3)`` afterwards, following the authors' decay formula.
It is calculated after observation removal and is not standardized. Add the
``decay`` field to the RNN_V2 embedding configuration. The default remains
False for compatibility with the team prototype.

The runnable synthetic checkpoint is
``python -m project.person_c.checkpoint``. It exercises CSV loading, task
processing, train-only processor fitting, an LSTM update, and held-out
inference. This is an integration check, not a clinical performance study.

Timing correction (November 17 audit): array row b represents the interval
[b,b+1) and is timestamped b+1, when that bin is complete. A prediction at
hour 5 uses only bins 0–4. The 336-hour cap yields 336 rows, not 337.
The oracle's hour/bin numbers identify array rows, not availability timestamps.
Exact-boundary observations at minute 300 enter bin 5, available at hour 6.
This conservative floor-bin convention differs from the paper's right-closed
resampling, but prevents sub-hour look-ahead.

Missingness example
-------------------

``python -m examples.eicu_hourly_los_lstm --output /tmp/tpc_ablation.json``
trains two synthetic LSTM configurations and evaluates each at three removal
rates. It reports MAD and explicitly named log1p-MSE in days. The latter is
not the original paper's log-MSE. These synthetic results demonstrate execution
only. Set ``PYHEALTH_RUN_TPC_INTEGRATION=1`` to enable the slow integration test.
