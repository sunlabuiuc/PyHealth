pyhealth.tasks.SleepStagingIBIISRUC
===================================

ECG-derived inter-beat-interval sleep staging for ISRUC-Sleep. By default the
task reads the ``X2`` ECG channel at its native 200 Hz rate, strides the IBI
signal by eight, and emits ``(T, 750)`` tensors with three-class labels. Choose
``scorer="expert2"`` to use the second expert's hypnogram instead of the
first.

.. autoclass:: pyhealth.tasks.sleep_staging_ibi.SleepStagingIBIISRUC
    :members:
    :undoc-members:
    :show-inheritance:
