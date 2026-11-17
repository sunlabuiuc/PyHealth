pyhealth.tasks.NHANESMortalityPrediction
========================================

5-year all-cause mortality prediction from NHANES accelerometer data,
following MotionAge.

The task cuts each participant's 5-minute epochs into windows and keeps the
ones with enough worn epochs. Each window is one sample, labeled 1 if the
participant died within ``followup_months`` of the NHANES interview.
Survivors with shorter follow-up are skipped.

All windows from a participant share one label. Split by participant with
``set_task(task, split=PatientSplit(...))``, and average each participant's
window probabilities to get one prediction per person.

The label is ``binary``, so the trainer reports binary metrics such as
``pr_auc`` and ``roc_auc``.

Task parameters:

- seq_len: int, default=1008 (3.5 days; 288 is one day)
- stride: int or None, default=None (same as seq_len)
- min_coverage: float, default=0.3
- followup_months: int, default=60
- covariate_set: ``"clinical"`` (age, BMI, waist, blood pressure, sex,
  income), ``"age_sex"``, or None for activity only; default="clinical"
- wear_aware: bool, default=True

.. autoclass:: pyhealth.tasks.NHANESMortalityPrediction
    :members:
    :undoc-members:
    :show-inheritance:
