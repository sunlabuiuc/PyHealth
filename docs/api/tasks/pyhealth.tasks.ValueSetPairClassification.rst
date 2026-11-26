pyhealth.tasks.ValueSetPairClassification
=========================================

The text-pair form of retrieval-augmented set completion (RASC) for clinical value set authoring: each candidate code retrieved for a value set becomes one (value set title, candidate display name) pair with a binary label, the input of :class:`~pyhealth.models.RASCCrossEncoder`. It reads an index built by :func:`~pyhealth.tasks.build_retrieval_index` (title embeddings, as in the paper) or by :func:`~pyhealth.tasks.build_hybrid_retrieval_index`, which fuses the title-embedding search with BM25 over titles by reciprocal rank fusion. :class:`~pyhealth.tasks.ValueSetCompletion` is the embedding form of the same task, for the paper's MLP.

.. note::

    A hybrid index is for :class:`~pyhealth.tasks.ValueSetPairClassification` only. It stores neighbors in fused-rank order, not score order. :class:`~pyhealth.tasks.ValueSetCompletion` assumes score order, so on a hybrid index it can give a code shared by several neighbors a ``score`` that is not the highest one. Build its index with :func:`~pyhealth.tasks.build_retrieval_index`.

.. autofunction:: pyhealth.tasks.build_hybrid_retrieval_index

.. autoclass:: pyhealth.tasks.ValueSetPairClassification
    :members:
    :undoc-members:
    :show-inheritance:
