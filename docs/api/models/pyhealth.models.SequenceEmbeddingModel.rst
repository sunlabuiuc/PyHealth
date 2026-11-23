pyhealth.models.SequenceEmbeddingModel
======================================

Embeddings for the packed records of ``jagged``
(:class:`~pyhealth.processors.JaggedFeatureProcessor`) fields. Each level's
records get one embedding per token namespace and a projection of their
values, concatenated; each level keeps its offsets and times.
:class:`~pyhealth.models.RNN_V2` builds on it. Embedding sizes default to
128 for every token namespace and values projection; pass one int, or an int
per field, to change them.

.. autoclass:: pyhealth.models.SequenceEmbeddingModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.level_embeddings

.. autoclass:: pyhealth.models.LevelEmbedding
    :members:

.. autoclass:: pyhealth.models.PretrainedEmbedding
    :members:
