pyhealth.models.ehrmamba_embedding
===================================

EHRMamba §2.2 embedding components for MIMIC-IV. Implements the full 7-component token embedding scheme (Eq. 1) fusing code, time-delta, age, token-type, visit-order, and visit-segment embeddings.

.. autodata:: pyhealth.models.ehrmamba_embedding.MIMIC4_TOKEN_TYPES

.. autodata:: pyhealth.models.ehrmamba_embedding.NUM_TOKEN_TYPES

.. autodata:: pyhealth.models.ehrmamba_embedding.SPECIAL_TYPE_MAX

.. autodata:: pyhealth.models.ehrmamba_embedding.NUM_VISIT_SEGMENTS

.. autodata:: pyhealth.models.ehrmamba_embedding.MAX_NUM_VISITS

.. autoclass:: pyhealth.models.ehrmamba_embedding.TimeEmbeddingLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ehrmamba_embedding.VisitEmbedding
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ehrmamba_embedding.EHRMambaEmbedding
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ehrmamba_embedding.EHRMambaEmbeddingAdapter
    :members:
    :undoc-members:
    :show-inheritance:
