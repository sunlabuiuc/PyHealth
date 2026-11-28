pyhealth.datasets.MITBIHDataset
===============================

Overview
--------

The `MIT-BIH Arrhythmia Database <https://physionet.org/content/mitdb/1.0.0/>`_
contains 48 half-hour, two-lead ambulatory ECG recordings from 47 patients,
sampled at 360 Hz, with every heartbeat annotated by cardiologists. Each
recording is loaded as one patient; waveforms are read lazily with ``wfdb``
(``pip install wfdb``).

:class:`~pyhealth.datasets.MITBIHAnomalyRatioSplit` provides a deterministic
record-level split by abnormal-beat ratio. Its default ``"paper"`` mode
reproduces the anomaly-detection split of MedTsLLM
(`Chan et al., MLHC 2024 <https://arxiv.org/abs/2408.07773>`_): the 6 least
and 4 most abnormal of the 33 MLII+V1 recordings that are not entirely
abnormal, which matches the paper's reported 1,354,170 training and 902,780
test samples at 125 Hz.

API Reference
-------------

.. autoclass:: pyhealth.datasets.MITBIHDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.datasets.MITBIHAnomalyRatioSplit
    :members:
    :undoc-members:
    :show-inheritance:
