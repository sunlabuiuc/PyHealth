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

The task implements adult/five-hour eligibility, admission static features,
hourly remaining-LOS labels, configurable observation removal, optional
decay features, and synthetic CSV validation. Dynamic extraction uses
existing YAIB helpers provisionally. Model training and ablation
experiments are documented separately.

``TPCLengthOfStayEICU()`` supports observation removal through
``missing_rate`` values from 0.0 to 1.0. Missing observations
are represented by NaN, with deterministic masking based on
the random seed and stay identifier.

When ``include_decay=True``, decay features are calculated
after observation removal. Observed values reset decay to
1.0; unobserved values decay by a factor of 0.75 per hour.


Extension points
----------------

* ``pyhealth/tasks/tpc_eicu.py`` implements hourly remaining-LOS
  preprocessing, configurable observation removal, and optional
  decay features. Observation removal is deterministic and
  applied before decay calculation.
* ``TODO(A)`` in ``examples/eicu_hourly_los_lstm.py``: integrate existing training
  work where available and complete explicit LSTM
  training, patient-disjoint splits, agreed losses/metrics and the two-training
  by three-test missingness study. This file is a scaffold and raises until
  implemented. The example filename is present but the rubric's working
  ablation requirement is still outstanding.
* Preprocessing contract tests in ``tests/core/test_tpc_eicu.py``
  cover observation removal and decay. Some hourly aggregation,
  forward-fill, and combined preprocessing tests remain skipped
  pending verification.
* Extend the integration check with actual model training when the training
  implementation is available. Its current scope ends at CSV-to-task outputs.

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
