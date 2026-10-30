pyhealth.models.SequenceEmbeddingModel
======================================

Embeddings for the packed records of ``sequence_v2``
(:class:`~pyhealth.processors.SequenceProcessorV2`) fields. Each level's
records get one embedding per token namespace and a projection of their
values, concatenated; each level keeps its offsets and times.
:class:`~pyhealth.models.Unified_RNN` builds on it.

.. autoclass:: pyhealth.models.SequenceEmbeddingModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.LevelEmbedding
    :members:

.. autoclass:: pyhealth.models.PretrainedEmbedding
    :members:
