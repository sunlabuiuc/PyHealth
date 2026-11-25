pyhealth.models.RASCCrossEncoder
===================================

The cross-encoder of Retrieval-Augmented Set Completion (RASC) for clinical value set authoring: a value set title and a retrieved candidate code's display name are read as one sentence pair by a BERT encoder (SapBERT in the paper), and a linear head scores whether the code belongs to the value set. Pair it with :class:`~pyhealth.tasks.ValueSetPairClassification`. Other inputs of the dataset, such as the retrieval score, can be added to the logit by late fusion.

Paper: `Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value Set Authoring <https://arxiv.org/abs/2604.14616>`_ (Mukherjee et al., 2026).

.. autoclass:: pyhealth.models.RASCCrossEncoder
    :members:
    :undoc-members:
    :show-inheritance:
