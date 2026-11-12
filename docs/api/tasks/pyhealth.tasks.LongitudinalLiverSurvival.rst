pyhealth.tasks.LongitudinalLiverSurvival
========================================

Time-to-event samples from repeated liver-lab visits, for reproducing the Model
for Allograft Survival (MAS; Gao et al., MLHC 2024). ``mode="counting_process"``
produces ``[start, stop, event]`` intervals for time-varying Cox models;
``mode="landmark"`` produces ``[duration, event]`` labels from each visit.
Features only use labs measured at or before the visit.

.. autoclass:: pyhealth.tasks.LongitudinalLiverSurvival
    :members:
    :undoc-members:
    :show-inheritance:

.. currentmodule:: pyhealth.tasks.liver_survival

.. autofunction:: build_counting_process_samples

.. autofunction:: build_landmark_samples

.. autofunction:: build_binary_horizon_samples

.. autofunction:: counting_process_samples_from_frame

.. autodata:: FEATURE_SETS
