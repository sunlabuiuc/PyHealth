pyhealth.tasks.SeizureDetectionHelsinki
=======================================

Seizure detection on the Helsinki neonatal EEG dataset.

For each recording the task combines the per-second annotations of the three
experts, builds an 18-channel bipolar montage from the referential channels,
band-pass filters and resamples the signal, and cuts overlapping windows.
The label is ``binary`` (1 seizure, 0 non-seizure), so the trainer reports
binary metrics such as ``pr_auc`` and ``roc_auc``.

With the default ``label_mode="consensus"``, a second counts as a seizure
when the three experts mark it, as a non-seizure when none does, and windows
that contain a second on which the experts disagree are dropped. A window is
a seizure when at least ``min_seizure_fraction`` of its seconds are seizure
seconds.

Task parameters:

- label_mode: ``"consensus"`` (default), ``"majority"``, ``"any"``, ``"A"``,
  ``"B"`` or ``"C"``
- window_seconds: int, default=16
- stride_seconds: int, default=4
- resample_rate: float or None, default=32
- bandpass: ``(low, high)`` in Hz or None, default=(0.5, 16)
- min_seizure_fraction: float, default=0.5
- montage: electrode pairs, default ``HELSINKI_BIPOLAR_MONTAGE``

.. autoclass:: pyhealth.tasks.SeizureDetectionHelsinki
    :members:
    :undoc-members:
    :show-inheritance:
