pyhealth.datasets.MEDSDataset
===================================

Dataset class for data in the `Medical Event Data Standard (MEDS) <https://github.com/Medical-Event-Data-Standard/meds>`_, a minimal event-based schema for machine learning over EHR data (MEDS Working Group / Arnrich et al., ICLR 2024 Workshop on Learning from Time Series For Health; `openreview:IsHy2ebjIG <https://openreview.net/forum?id=IsHy2ebjIG>`_). Sharded Parquet event files are read with their native types, and standard MEDS splits (train / tuning / held_out) can be selected directly via the ``subset`` argument. The canonical subject-to-split mapping is defined in ``metadata/subject_splits.parquet``; see the `MEDS schema documentation <https://medical-event-data-standard.github.io/>`_.

.. autoclass:: pyhealth.datasets.MEDSDataset
    :members:
    :undoc-members:
    :show-inheritance:
