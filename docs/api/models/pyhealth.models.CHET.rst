pyhealth.models.CHET
===================================

The CHET model for health event prediction via transition functions
on dynamic disease graphs.

Paper: Chang Lu, Tian Han, Yue Ning. "Context-aware Health Event
Prediction via Transition Functions on Dynamic Disease Graphs."
Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2022).
Paper link: https://arxiv.org/abs/2112.05195

Dynamic Disease Graphs:
- CHET models medical code transitions across longitudinal visits
  by constructing and updating dynamic disease graphs.
- Co-occurrence adjacency can be precomputed from the training dataset via
  ``compute_cooccurrence_adj(dataset, feature_key)`` or automatically
  computed from the dataset during initialization when ``adj=None``.
  Only training data should be used to avoid data leakage.
- Note that per-visit dense N x N adjacency multiplications scale
  quadratically with vocabulary size N.
- Dual graph convolutions propagate structural information across center
  and neighbor embeddings.
- Transition dynamics decompose diseases into persistent conditions (GRU),
  emerging neighbor conditions (attention), and novel unrelated
  conditions (attention).

.. autofunction:: pyhealth.models.compute_cooccurrence_adj

.. autoclass:: pyhealth.models.SingleHeadAttentionLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.DotProductAttention
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.CHET
    :members:
    :undoc-members:
    :show-inheritance:

