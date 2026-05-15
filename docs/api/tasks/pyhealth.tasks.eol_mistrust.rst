pyhealth.tasks.eol_mistrust
==============================

End-of-life cohort tasks from Boag et al. 2018, *"Racial Disparities and
Mistrust in End-of-Life Care"*
(`paper <https://proceedings.mlr.press/v85/boag18a.html>`_). Three binary
prediction targets are defined on top of the
:class:`~pyhealth.datasets.EOLMistrustDataset`:
Left-AMA, code-status change (DNR/DNI/CMO), and in-hospital mortality.
All three share the same input schema (demographics, diagnoses, procedures,
medications, and optionally clinical notes) and differ only in the
extracted label.

Supports both a corrected ``"default"`` and a notebook-faithful
``"paper_like"`` label-extraction strategy via the
``dataset_prepare_mode`` parameter.

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
