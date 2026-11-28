pyhealth.tasks.SleepStagingIBISleepAccel
========================================

Heart-rate-derived inter-beat-interval sleep staging for Sleep-Accel. The task
interpolates ``60 / bpm`` onto a 25 Hz grid and emits ``(T, 750)`` tensors with
three-class labels. It maps R&K stages to Wake ``0``, NREM ``1``, REM ``2`` and
unknown ``-1``, and caps each recording at 1100 epochs by default.

.. autoclass:: pyhealth.tasks.sleep_staging_ibi.SleepStagingIBISleepAccel
    :members:
    :undoc-members:
    :show-inheritance:
