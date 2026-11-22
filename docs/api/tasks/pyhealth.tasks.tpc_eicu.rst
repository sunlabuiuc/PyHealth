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

The baseline implements adult/five-hour eligibility, admission static features,
hourly remaining-LOS labels and synthetic CSV validation. Dynamic extraction
uses existing YAIB helpers provisionally. This is a partial contribution under
active integration, not a final training/ablation implementation.

``TPCLengthOfStayEICU()`` works with zero observation removal and no decay.
Nonzero ``missing_rate`` and ``include_decay=True`` deliberately raise
``NotImplementedError`` when samples are processed, rather than silently
producing unimplemented results.

Extension points
----------------

* ``TODO(B)`` in ``pyhealth/tasks/tpc_eicu.py``: review dynamic feature coverage
  and hourly aggregation, confirm training-only normalization/forward fill,
  and integrate decay after removal. For masking, A/B should reuse the existing
  ``_apply_missingness`` in the repository-root ``tpc_eicu.py`` prototype after
  checking compatibility. It already implements seeded removal of hourly cells;
  this placeholder is an integration boundary, not a request to write it again.
  Confirm raw-row versus aggregated-cell removal and rerun the pending tests.
* ``TODO(A)`` in ``examples/eicu_hourly_los_lstm.py``: integrate existing training
  work where available and complete explicit LSTM
  training, patient-disjoint splits, agreed losses/metrics and the two-training
  by three-test missingness study. This file is a scaffold and raises until
  implemented. The example filename is present but the rubric's working
  ablation requirement is still outstanding.
* Disabled preprocessing contract tests in ``tests/core/test_tpc_eicu.py``
  retain numerical expectations. Enable and revise these after the shared
  interface is confirmed. Skipped tests are pending work, not passing coverage.
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
* Planned decay mirrors dynamic dimensions/times. Existing schema wiring is
  provisional until preprocessing review.
* Split by patient_id rather than stay_id. Fit learned processors on training
  patients only. Existing YAIB label processors min-max scale training targets;
  activated predictions return original hours. Avoid double unscaling.

This reproduces the prediction problem, not the TPC architecture. Existing
median/floor aggregation and YAIB features differ from the paper. Confirm
preprocessing, loss and metric conventions before comparing paper results.
The original paper already ablates decay removal; a decay toggle alone is
not a novel extension for the assignment.
