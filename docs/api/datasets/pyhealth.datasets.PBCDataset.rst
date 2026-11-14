PBCDataset
==========

.. autoclass:: pyhealth.datasets.PBCDataset
    :members:
    :show-inheritance:

The author's longitudinal ``pbc2.csv`` provides the 312-patient cohort used by
Neural Fine-Gray. This loader requires a local file and a controlled cache.
It does not download or redistribute records. R's 418-person ``pbc`` object
has different labels and is not interchangeable.

Use :class:`pyhealth.tasks.CompetingRisksPBC` to create baseline samples.
``dataset_name`` can name a separate cohort/cache; its default is ``"pbc"``.
