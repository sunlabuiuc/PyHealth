pyhealth.datasets.EESM23Dataset
===================================

Dataset is available at https://openneuro.org/datasets/ds005178

Ear-EEG Sleep Monitoring 2023 (EESM23) is a BIDS-formatted collection of
in-ear dry EEG from ten healthy subjects. Each subject has up to twelve home
nights; manual sleep scoring (30 s epochs) is available for the first two
sessions (partial PSG nights). Ear electrodes are RB, RT, LB and LT, not a
10-20 scalp montage.

The dataset writes one metadata row per ear-EEG night. The default task,
``SleepStagingEarEEG``, extracts scored 30-second epochs on the four ear
channels, following the TFM-Tokenizer preprocessing for EESM23. Choose the
train, validation and test subjects with ``SubjectSplit`` or
``EESM23Dataset.default_subject_split()`` (``sub-001``-``sub-006`` /
``sub-007``-``sub-008`` / ``sub-009``-``sub-010``).

.. autoclass:: pyhealth.datasets.EESM23Dataset
    :members:
    :undoc-members:
    :show-inheritance:
