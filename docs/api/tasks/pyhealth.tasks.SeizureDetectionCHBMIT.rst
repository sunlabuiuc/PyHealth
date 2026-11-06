pyhealth.tasks.SeizureDetectionCHBMIT
=========================================

Seizure detection on the CHB-MIT Scalp EEG Database.

For each recording the task keeps 16 bipolar channels, cuts non-overlapping
10-second windows, and labels them as seizure or background. The windowing
follows CBraMod and TFM-Tokenizer. Extra seizure windows are tagged with
``is_augmented`` so they can be dropped from validation and test.

The label is ``binary`` (1 seizure, 0 background), so the trainer reports
binary metrics such as ``pr_auc`` and ``roc_auc``.

Resampling is done per window, as in CBraMod: windows are cut at the native
256 Hz and each one is resampled to ``resample_rate``. TUAB and TUEV instead
resample the whole recording before windowing, so values near window edges
differ slightly if you compare the two pipelines.

Task parameters:

- resample_rate: float, default=200
- window_sec: float, default=10
- normalization: None, ``"div_by_100"`` or ``"95th_percentile"``
- bandpass_filter: tuple or None, default=None
- notch_filter: float or None, default=None
- compute_stft: bool, default=True
- oversample_seizures: bool, default=True
- oversample_stride_sec: float, default=5
- label_mode: ``"overlap"`` or ``"cbramod"``, default=``"overlap"``
- subjects: list of subject ids, or None to process every subject

.. autoclass:: pyhealth.tasks.SeizureDetectionCHBMIT
    :members:
    :undoc-members:
    :show-inheritance:
