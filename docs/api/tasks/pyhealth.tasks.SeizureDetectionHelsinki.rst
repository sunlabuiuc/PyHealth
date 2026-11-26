pyhealth.tasks.SeizureDetectionHelsinki
=======================================

Seizure detection on the Helsinki neonatal EEG dataset.

For each recording the task combines the per-second annotations of the three
experts, builds an 18-channel bipolar montage from the referential channels,
band-pass filters and resamples the signal, and cuts overlapping windows.
The label is ``binary`` (1 seizure, 0 non-seizure), so the trainer reports
binary metrics such as ``pr_auc`` and ``roc_auc``.

The defaults follow Borovac et al. (2022) and their published code. With
``label_mode="consensus"``, a second counts as a seizure when the three
experts mark it, as a non-seizure when none does, and windows that contain a
second on which the experts disagree are dropped. A window is a seizure when
all its seconds are seizure seconds and a non-seizure when none is; windows
that are partly seizure are dropped, unless ``min_seizure_fraction`` is set.

Task parameters:

- label_mode: ``"consensus"`` (default), ``"majority"``, ``"any"``, ``"A"``,
  ``"B"`` or ``"C"``
- window_seconds: int, default=16
- stride_seconds: int, default=4
- resample_rate: float or None, default=32
- bandpass: stopband edges ``(low, high)`` in Hz of the Chebyshev type II
  band-pass filter, or None, default=(0.5, 32)
- min_seizure_fraction: float or None, default=None
- montage: electrode pairs, default ``HELSINKI_BIPOLAR_MONTAGE``

.. autoclass:: pyhealth.tasks.SeizureDetectionHelsinki
    :members:
    :undoc-members:
    :show-inheritance:
