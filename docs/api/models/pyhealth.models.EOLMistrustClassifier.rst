pyhealth.models.EOLMistrustClassifier
=======================================

Multimodal classifier for the end-of-life prediction targets from
Boag et al. 2018, *"Racial Disparities and Mistrust in End-of-Life Care"*
(`paper <https://proceedings.mlr.press/v85/boag18a.html>`_).

The model handles three modality types from the
:class:`~pyhealth.datasets.EOLMistrustDataset`:

- **Coded EHR sequences** (diagnoses, procedures, drugs) — learned
  embeddings with mean pooling.
- **Scalar numeric features** (age, length of stay) — linear projections.
- **Text / categorical fields** (demographics, clinical notes) — stable
  hash-based token embeddings with mean pooling.

It predicts a binary target such as Left-AMA, code-status change, or
in-hospital mortality.

.. autoclass:: pyhealth.models.EOLMistrustClassifier
    :members:
    :undoc-members:
    :show-inheritance:
