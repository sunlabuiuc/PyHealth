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
  Only training data should be used to avoid data leakage; when ``adj=None``,
  ensure only the training split is passed. Unlike ``ehr_cooccurrence_adjacency``
  in ``pyhealth.models.utils`` (which computes unweighted 0/1 matrices over
  drug label sets for recommendation models), ``compute_cooccurrence_adj``
  operates on input code features, supports self-loops, and row-normalizes
  transition probabilities for disease graph convolutions. When user-supplied,
  ``adj`` is expected to be row-normalized with the padding row and column
  (index 0) zeroed.
- Note that per-visit dense N x N adjacency multiplications and the
  patient-by-visit Python loop scale with vocabulary size and sequence length,
  which can impact training speed for large clinical vocabularies.
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

