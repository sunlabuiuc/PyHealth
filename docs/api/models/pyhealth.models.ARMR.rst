pyhealth.models.ARMR
===================================

ARMR (Wu et al., "ARMR: Adaptively Responsive Network for Medication
Recommendation", IJCAI 2025) for the drug-recommendation tasks
(:class:`~pyhealth.tasks.DrugRecommendationMIMIC3`,
:class:`~pyhealth.tasks.DrugRecommendationMIMIC4`). It needs ``conditions``,
``procedures`` and ``drugs_hist`` in the input schema and predicts ``drugs``.

Example notebook: ``examples/armr_mimic3_drug_recommendation.ipynb``.

.. autoclass:: pyhealth.models.ARMR
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.armr.PiecewiseTemporalLearning
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.models.armr.SelectiveSSM
    :members:
    :show-inheritance:
