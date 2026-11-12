pyhealth.tasks.SleepStagingEarEEG
====================================

Sleep staging on EESM23 ear-EEG (OpenNeuro ds005178).

For each scored night the task reads the EEGLAB ``.set`` file, keeps the four
in-ear channels RB, RT, LB and LT (the flat ELE channel is never used), and
labels each 30-second epoch from the manual scoring TSV. Preprocessing
follows TFM-Tokenizer: per-epoch band-pass (0.1-100 Hz) and 50 Hz notch
filtering (applied per epoch because NaN gaps are common), optional
resampling, and 95th-percentile normalization.

The default label map has six classes (Wake, REM, N1, N2, N3, Artefact),
matching ``scoring_idx - 1`` in TFM-Tokenizer. Set ``include_artefact=False``
to drop Artefact epochs and use five AASM stages. Only ``ses-001`` and
``ses-002`` nights carry manual scoring.

The label is ``multiclass``, so the trainer reports metrics such as
``accuracy``, ``f1_macro``, and ``cohen_kappa``.

Task parameters:

- resample_rate: float, default=200
- epoch_sec: float, default=30
- normalization: None, ``"div_by_100"`` or ``"95th_percentile"``
- bandpass_filter: tuple or None, default=(0.1, 100.0)
- notch_filter: float or None, default=50.0
- compute_stft: bool, default=True
- include_artefact: bool, default=True
- subjects: list of subject ids, or None to process every subject

.. autoclass:: pyhealth.tasks.SleepStagingEarEEG
    :members:
    :undoc-members:
    :show-inheritance:
