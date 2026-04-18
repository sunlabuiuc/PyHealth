pyhealth.models.EOLMistrustClassifier
=======================================

Multimodal classifier that mirrors the end-of-life prediction head from
Boag et al. 2018. It consumes sequence features (diagnoses, procedures,
drugs), tensor features (age, length of stay), and text features
(demographics and free-text clinical notes) from the
:class:`~pyhealth.datasets.EOLMistrustDataset` and predicts a binary
target such as Left-AMA, code-status change, or in-hospital mortality.

.. autoclass:: pyhealth.models.EOLMistrustClassifier
    :members:
    :undoc-members:
    :show-inheritance:
