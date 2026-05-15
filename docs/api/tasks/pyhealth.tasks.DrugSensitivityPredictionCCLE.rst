pyhealth.tasks.DrugSensitivityPredictionCCLE
=============================================

Task definition for multi-drug binary sensitivity prediction on the CCLE dataset.
Follows the same interface as
:class:`~pyhealth.tasks.DrugSensitivityPredictionGDSC`, enabling cross-dataset
evaluation: train on GDSC, evaluate on CCLE using overlapping drugs identified
via :meth:`~pyhealth.datasets.GDSCDataset.get_overlap_drugs`.

Each sample represents one cancer cell line; the label vector encodes whether the
cell line is sensitive (1) or resistant (0) to each of the tested drugs.
Missing (untested) drug/cell-line pairs are indicated by a companion mask vector.

Reference dataset:

    Barretina, J. et al. (2012). *The Cancer Cell Line Encyclopedia enables
    predictive modelling of anticancer drug sensitivity.*
    Nature, 483(7391), 603-607.

.. autoclass:: pyhealth.tasks.DrugSensitivityPredictionCCLE
    :members:
    :undoc-members:
    :show-inheritance:
