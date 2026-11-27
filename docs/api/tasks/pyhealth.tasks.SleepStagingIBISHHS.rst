pyhealth.tasks.SleepStagingIBISHHS
===================================

Per-recording inter-beat-interval sleep staging for NSRR SHHS. Reads the ECG
channel at 125 Hz, strides IBI by five, and emits ``(T, 750)`` tensors with
three-class labels.

.. autoclass:: pyhealth.tasks.sleep_staging_ibi.SleepStagingIBISHHS
    :members:
    :undoc-members:
    :show-inheritance:
