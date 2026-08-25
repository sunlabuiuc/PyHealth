pyhealth.tasks.drug_recommendation
===================================

Task Classes
------------

.. autoclass:: pyhealth.tasks.drug_recommendation.DrugRecommendationMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.drug_recommendation.DrugRecommendationMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.drug_recommendation.DrugRecommendationEICU
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.drug_recommendation.DrugRecommendationOMOP
    :members:
    :undoc-members:
    :show-inheritance:

Task Functions (Legacy)
------------------------

.. note::

   These functions predate the current dataset API: they expect an
   indexable, ``len()``-able ``Patient`` with ``Visit.get_code_list(table)``,
   neither of which the current ``pyhealth.data.Patient``/``Visit`` classes
   provide (``Visit`` is now a deprecated no-op stub). As a result they
   cannot currently be run through ``BaseDataset.set_task()``. Prefer the
   task classes above (``DrugRecommendationMIMIC3``/``MIMIC4``/``EICU``),
   which use the current API and are actively maintained.

.. autofunction:: pyhealth.tasks.drug_recommendation.drug_recommendation_mimic3_fn
.. autofunction:: pyhealth.tasks.drug_recommendation.drug_recommendation_mimic4_fn
.. autofunction:: pyhealth.tasks.drug_recommendation.drug_recommendation_omop_fn