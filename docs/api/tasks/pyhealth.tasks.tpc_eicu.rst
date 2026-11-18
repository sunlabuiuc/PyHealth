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
run the opt-in command below for CSV-to-LSTM coverage.

Optional decay input
--------------------

Set ``include_decay=True`` to add a separate ``decay`` sequence with 48 values
per hour. It is zero before any observation, one when observed, and
``1 / (gap_hours * 4/3)`` afterwards, following the authors' decay formula.
It is calculated after observation removal and is not standardized. Add the
``decay`` field to the RNN_V2 embedding configuration. The default remains
False for compatibility with the team prototype.

The runnable synthetic checkpoint is
``python -m examples.tpc_support.checkpoint``. It exercises CSV loading, task
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

Reproducing and extending the example
------------------------------------

Use a repository checkout and Python 3.12 or 3.13. Install the package and
pytest in an isolated environment, then run these commands from the repository
root (no credentialed data is needed)::

    python -m pip install -e . pytest
    python -m pytest tests/core/test_tpc_eicu.py -q --durations=5
    PYHEALTH_RUN_TPC_INTEGRATION=1 python -m pytest tests/integration/test_tpc_checkpoint.py -q
    python -m examples.tpc_support.checkpoint --output /tmp/tpc_checkpoint.json
    python -m examples.eicu_hourly_los_lstm --steps 3 --output /tmp/tpc_ablation.json
    python examples/tpc_support/generate_synthetic.py /tmp/tpc_synthetic

The synthetic generator, numerical expectations, model update checks, and
sample experiment use ``examples/tpc_support``. The rich CSV integration check
starts a local processing server and is intentionally opt-in. The regular core
checks use in-memory synthetic events. Import time is separate from the small
per-test execution time; the entire process is not guaranteed below one second.

Input contract for another training script:

* One sample per stay. Split on ``patient_id`` (unique person), not ``stay_id``.
* ``dynamic`` has ``time=1..H`` and ``values`` of shape ``(H, 48)``;
  ``H=min(floor(total stay hours), 336)``. Values are raw with NaN gaps.
* ``static`` has one record at time zero, ordered as age, sex, height, weight.
* ``los`` has times ``5..H`` and values of shape ``(H-4, 1)`` in hours.
  Exact-discharge labels may be zero. Remaining LOS is not capped at 336.
* ``decay`` is optional, with the same times and dimensions as dynamic;
  enable it with ``include_decay=True`` and include it in model embeddings.
* Use ``RNN_V2(..., rnn_type="LSTM")`` explicitly: its default is GRU.
* Fit processors on training patients and reuse them on validation/test data.
  ``set_task(..., split=...)`` is appropriate for the full dataset path.
* The inherited label processor min-max scales targets during training.
  Its activated predictions are returned in original hours; do not unscale
  ``y_prob`` a second time. Convert both predictions and targets to days when
  comparing daily errors. Define a zero-target policy before using log(y),
  MAPE or other metrics that cannot accept zero targets.

The example's ``run`` function implements the two training-removal conditions
and three test-removal conditions. To run a real experiment, replace its
synthetic generation and hard-coded patient IDs with an authorized dataset
and patient-disjoint train/validation/test split. Keep initialization, test
masks and preprocessing policies controlled across conditions. Add repeated
seeds, final training settings and paper-compatible metrics before making
performance claims. The three-step synthetic example is not a final study.

This task implements the hourly prediction problem rather than the TPC model.
Differences include the YAIB feature set, completed floor bins with median
aggregation, training-fitted z-score preprocessing and default scaled-MSE
training. Missingness is applied to aggregated hourly cells. The proposed
extension is the training/test missingness grid; merely removing decay repeats
an ablation already present in the original paper.

Cached task samples from versions with start-of-bin timestamps must be
regenerated. Rerun the core and integration checks after changing extraction,
labels, missingness, processor schemas or the model interface.
