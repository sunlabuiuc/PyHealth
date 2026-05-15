pyhealth.models.CADRE
=====================

CADRE (Contextual Attention-based Drug REsponse prediction) is a collaborative
filtering model for multi-drug binary sensitivity prediction.  It encodes cancer
cell-line genomic profiles using frozen Gene2Vec embeddings conditioned on
drug target pathway context (contextual attention), then decodes per-drug
predictions via a dot-product collaborative filter.

Implementation of:

    Tao, Y. et al. (2020). *Predicting Drug Sensitivity of Cancer Cell Lines via
    Collaborative Filtering with Contextual Attention.*
    Proceedings of Machine Learning Research, 126, 456-477.  PMLR (MLHC 2020).

Original code: https://github.com/yifengtao/CADRE

See also :class:`~pyhealth.models.CADREDotAttn` for the Transformer-style
scaled dot-product attention extension.

.. autoclass:: pyhealth.models.ExpEncoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.DrugDecoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.CADRE
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.cadre_collate_fn
