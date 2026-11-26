pyhealth.tasks.MetabricSurvivalPrediction
=========================================

Time-to-event survival prediction on :class:`~pyhealth.datasets.MetabricDataset`:
one sample per patient with a flat ``features`` vector and ``duration`` and
``event`` labels, the format :class:`~pyhealth.models.SurvivalMDN` expects.
Options select the feature set (all, clinical only, genes only), the time unit
(months, years, or the months / 365.25 scale of the SODEN and Survival MDN
files) and an administrative censoring horizon, for task-level ablations.

.. autoclass:: pyhealth.tasks.MetabricSurvivalPrediction
    :members:
    :undoc-members:
    :show-inheritance:
