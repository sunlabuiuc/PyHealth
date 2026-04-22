pyhealth.tasks.DrugSensitivityPredictionGDSC
=============================================

Task definition for multi-drug binary sensitivity prediction on the GDSC dataset.
Each sample represents one cancer cell line; the label vector encodes whether the
cell line is sensitive (1) or resistant (0) to each of the 260 screened drugs.
Missing (untested) drug/cell-line pairs are indicated by a companion mask vector.

This task is used with :class:`~pyhealth.datasets.GDSCDataset` and the
:class:`~pyhealth.models.CADRE` model to reproduce the results from:

    Tao, Y. et al. (2020). *Predicting Drug Sensitivity of Cancer Cell Lines via
    Collaborative Filtering with Contextual Attention.*
    Proceedings of Machine Learning Research, 126, 456-477.  PMLR (MLHC 2020).

.. autoclass:: pyhealth.tasks.DrugSensitivityPredictionGDSC
    :members:
    :undoc-members:
    :show-inheritance:
