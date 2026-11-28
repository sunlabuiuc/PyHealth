pyhealth.tasks.SleepStagingIBIMESA
===================================

Per-recording inter-beat-interval sleep staging for NSRR MESA. Resamples the
Pleth channel from 256 Hz to 100 Hz, strides IBI by four, and emits
``(T, 750)`` tensors with three-class labels.

.. autoclass:: pyhealth.tasks.sleep_staging_ibi.SleepStagingIBIMESA
    :members:
    :undoc-members:
    :show-inheritance:
