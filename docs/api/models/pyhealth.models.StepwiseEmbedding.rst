pyhealth.models.StepwiseEmbedding
====================================

Step-wise Embedding model for heterogeneous clinical time-series.
Groups features by clinical modality, embeds per-group using Linear,
MLP, or FT-Transformer layers, aggregates group embeddings, and
processes the result with a Transformer backbone.

Reference: Kuznetsova et al., "On the Importance of Step-wise Embeddings
for Heterogeneous Clinical Time-Series", JMLR 2023.

.. autoclass:: pyhealth.models.stepwise_embedding.StepwiseEmbeddingLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.StepwiseEmbedding
    :members:
    :undoc-members:
    :show-inheritance:
