pyhealth.tasks.eol_mistrust
==============================

End-of-life cohort tasks from Boag et al. 2018, *"Racial Disparities and
Mistrust in End-of-Life Care."* Three binary prediction targets are
defined on top of the :class:`~pyhealth.datasets.EOLMistrustDataset`:
Left-AMA, code-status change (DNR/DNI/CMO), and in-hospital mortality.
All three share the same input schema and differ only in the extracted
label.

.. autoclass:: pyhealth.tasks.EOLMistrustDownstreamMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.EOLMistrustLeftAMAPredictionMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.EOLMistrustCodeStatusPredictionMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.EOLMistrustMortalityPredictionMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:
