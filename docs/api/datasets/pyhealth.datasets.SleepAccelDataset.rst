pyhealth.datasets.SleepAccelDataset
===================================

PhysioNet Sleep-Accel contains wrist heart rate, acceleration, step counts and
30 s sleep-stage labels from a night of polysomnography. The PyHealth loader
writes ``sleep_accel-pyhealth.csv`` with absolute paths for each complete
subject and exposes the four modalities as one event per recording.

Because the dataset has no raw PPG or ECG waveform, IBI staging derives
``60 / bpm`` samples from the heart-rate stream. Use
:class:`~pyhealth.tasks.SleepStagingIBISleepAccel` to build the shared
``(T, 750)`` IBI tensors and three-class labels.

.. autoclass:: pyhealth.datasets.SleepAccelDataset
    :members:
    :undoc-members:
    :show-inheritance:
