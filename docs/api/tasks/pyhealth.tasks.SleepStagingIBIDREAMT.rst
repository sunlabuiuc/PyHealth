pyhealth.tasks.SleepStagingIBIDREAMT
=====================================

Per-recording inter-beat-interval sleep staging for DREAMT. Reads the
``data_100Hz`` BVP signal by default, recovers the 30 s epoch grid from stage
changes, and emits ``(T, 750)`` IBI tensors with three-class labels.

.. autoclass:: pyhealth.tasks.sleep_staging_ibi.SleepStagingIBIDREAMT
    :members:
    :undoc-members:
    :show-inheritance:
