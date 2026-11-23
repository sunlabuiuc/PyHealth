.. Contributors: Manling Tan, Yining Jin, Xiao Luo
.. NetIDs: manling4, yining34, xiaoluo6
..
.. Paper Title:
.. Temporal Pointwise Convolutional Networks for
.. Length of Stay Prediction in the Intensive Care Unit
..
.. Paper Authors:
.. Emma Rocheteau, Pietro Liò, and Stephanie Hyland (2021)
..
.. Paper URL:
.. https://arxiv.org/abs/2007.09483


Hourly remaining ICU length of stay
==================================

.. automodule:: pyhealth.tasks.tpc_eicu
   :members:
   :show-inheritance:

Implementation status
---------------------

The task implements adult cohort selection, hourly dynamic feature
extraction, admission static features, and hourly remaining-LOS labels.
The observation horizon is capped at 336 hours.

Observation removal is controlled by ``missing_rate`` (0.0 to 1.0).
Removal is deterministic for a given seed and stay identifier.
A rate of 0.0 preserves all observed values, while a rate of 1.0
removes all observed feature cells.

When ``include_decay=True``, decay features are calculated after
observation removal. Observed values reset to 1.0, and unobserved
values decay by a factor of 0.75 per hour.

Missing dynamic values remain NaN in the task output. Forward filling
and normalization are handled by the configured YAIB preprocessing
pipeline.


Extension points
----------------

* ``pyhealth/tasks/tpc_eicu.py`` implements hourly remaining-LOS
  preprocessing, deterministic observation removal, and optional
  decay features.
* ``tests/core/test_tpc_eicu.py`` covers cohort selection, static
  features, LOS labels, observation removal, decay, and temporal
  consistency. Some hourly aggregation, forward-fill, and combined
  preprocessing tests remain skipped pending verification.
* ``examples/eicu_hourly_los_lstm.py`` is the model-training and
  missingness-ablation integration point. Complete and validate
  model training, patient-disjoint splits, evaluation metrics,
  and the two-training by three-test missingness study.
* The integration checkpoint currently validates synthetic
  CSV-to-task outputs. It does not replace actual model training
  or final experimental evaluation.

Runnable validation
-------------------

From a repository checkout in Python 3.12 or 3.13::

    python -m pip install -e . pytest
    python examples/tpc_support/generate_synthetic.py /tmp/tpc_synthetic
    python -m examples.tpc_support.checkpoint --output /tmp/tpc_checkpoint.json
    python -m pytest tests/core/test_tpc_eicu.py -q --durations=5
    PYHEALTH_RUN_TPC_INTEGRATION=1 python -m pytest tests/integration/test_tpc_checkpoint.py -q

All seven generated CSVs use invented records across five patients. The
checkpoint verifies cohort selection, static values, aggregated measurements
and remaining-LOS labels against independent numerical expectations. It does
not train a model. The real loader check starts a local processing server and
is opt-in. Core assertion execution is fast; imports still add startup time.

Input/output contract to review
------------------------------

Load lab, vitalperiodic, vitalaperiodic, intakeoutput, respiratorycharting and
nursecharting using eICUDataset. One sample is produced per eligible stay:

* Adults have known age >=18 and total ICU stay >=5 hours; ``> 89`` maps to 90.
  Unknown age excludes the stay. Unknown sex or invalid height/weight is NaN.
* Static features are age, sex (male=1, female=0), admission height and weight.
* Dynamic uses 48 YAIB features. For H=min(floor(total LOS hours),336), its
  values have shape (H,48), with missing values retained as NaN.
* Bin [b,b+1) becomes available at timestamp b+1. At hour 5 the model may
  consume bins 0–4 only. Regenerate older cached samples with start-bin times.
* Labels at hours 5..H have shape (H-4,1) and equal total LOS hours minus h.
  Exact-discharge labels may be zero. The observation horizon is capped;
  remaining LOS is not capped.
* ``missing_rate`` controls removal of observed hourly feature cells.
  A rate of 0.0 preserves observations; 1.0 removes all observed cells.
* ``include_decay=True`` adds decay features matching the dynamic
  feature dimensions. Decay resets to 1.0 upon observation and
  decreases by a factor of 0.75 per unobserved hour.
* Split by patient_id rather than stay_id. Fit learned processors on training
  patients only. Existing YAIB label processors min-max scale training targets;
  activated predictions return original hours. Avoid double unscaling.

This reproduces the prediction problem, not the TPC architecture. Existing
median/floor aggregation and YAIB features differ from the paper. Confirm
preprocessing, loss and metric conventions before comparing paper results.
The original paper already ablates decay removal; a decay toggle alone is
not a novel extension for the assignment.
