pyhealth.tasks.mitbih
=====================

Overview
--------

ECG anomaly detection on the MIT-BIH Arrhythmia Database, following MedTsLLM
(`Chan et al., MLHC 2024 <https://arxiv.org/abs/2408.07773>`_). Each recording
is resampled from 360 Hz to 125 Hz and cut into 128-sample windows of the
two-lead signal. The label is a per-time-step binary mask in which every
abnormal beat annotation (any beat other than ``N``) is expanded to 150 ms
before and after the beat.

Use with :meth:`pyhealth.datasets.MITBIHDataset.anomaly_ratio_split` to obtain
the paper's train/test split.

API Reference
-------------

.. automodule:: pyhealth.tasks.mitbih
    :members:
    :undoc-members:
    :show-inheritance:
