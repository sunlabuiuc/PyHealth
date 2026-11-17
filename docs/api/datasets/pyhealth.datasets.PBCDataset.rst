PBCDataset
==========

.. autoclass:: pyhealth.datasets.PBCDataset
    :members:
    :show-inheritance:

The longitudinal ``pbc2.csv`` distributed in
`Jeanselme's DeepSurvivalMachines fork
<https://github.com/Jeanselme/DeepSurvivalMachines/tree/e4b07b3f497f2266eaa71d0e182195e95663d367>`_
provides the 312-patient cohort used by Neural Fine-Gray.
This loader requires a local file and a controlled cache.
It does not download or redistribute records. R's 418-person ``pbc`` object
has different labels and is not interchangeable.

Use :class:`pyhealth.tasks.CompetingRisksPBC` to create baseline samples.
``dataset_name`` can name a separate cohort/cache; its default is ``"pbc"``.
