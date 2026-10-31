pyhealth.models.Unified_LogisticRegression
=========================================

A linear prediction head over bottom-up means of learned record embeddings
and flattened static tensors. Unlike :class:`~pyhealth.models.Unified_MLP`,
it introduces no MLPs at hierarchy levels or on static fields.

.. autoclass:: pyhealth.models.Unified_LogisticRegression
    :members:
    :show-inheritance:

See :doc:`pyhealth.models.Unified_MLP` for a shared input example and the
spatial, empty-group, and per-step prediction conventions.

The existing :class:`~pyhealth.models.LogisticRegression` remains available
for legacy schemas. Both versions use learned embeddings rather than
classical logistic regression directly on raw features.
