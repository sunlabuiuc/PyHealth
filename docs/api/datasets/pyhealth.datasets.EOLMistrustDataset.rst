pyhealth.datasets.EOLMistrustDataset
======================================

MIMIC-III dataset wrapper used to replicate Boag et al. 2018,
*"Racial Disparities and Mistrust in End-of-Life Care"*
(`paper <https://proceedings.mlr.press/v85/boag18a.html>`_). It loads the
admissions, ICU stays, and (optionally) note events tables, and exposes
the proxy-mistrust and end-of-life cohort definitions used by the three
downstream tasks in :doc:`../tasks/pyhealth.tasks.eol_mistrust`.

Supports both a corrected ``"default"`` pipeline and a notebook-faithful
``"paper_like"`` reproduction mode via the ``dataset_prepare_mode``
parameter.

.. autoclass:: pyhealth.datasets.EOLMistrustDataset
    :members:
    :undoc-members:
    :show-inheritance:
