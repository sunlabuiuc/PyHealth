pyhealth.models.BEHRT
===================================

The BEHRTEmbedding, BEHRTLayer, and the complete BEHRT model for electronic health records.

Paper: Li, Yikuan, et al. "BEHRT: Transformer for Electronic Health Records."
Scientific Reports 10.1 (2020): 7155.
Paper link: https://www.nature.com/articles/s41598-020-62922-y

Temporal Features:
- ``age``: Patient chronological age (integer or float tensor of shape ``[batch_size, seq_len]`` or ``[batch_size]`` with values in ``[0, max_age - 1]``).
  Users can pass integer age tensors directly in batch kwargs (e.g. ``kwargs["age"]``). Note that standard categorical sequence processors map unique tokens to arbitrary vocabulary indices rather than biological ages; to preserve true chronological ages, provide pre-binned integer tensors. When omitted, age embeddings are skipped.
- ``segment``: Visit segment indicators (integer tensor of shape ``[batch_size, seq_len]`` or ``[batch_size]``) distinguishing
  consecutive hospital visits. When omitted, segment embeddings are skipped.
- If neither is provided, BEHRT operates as a standard position-aware Transformer baseline.

Sequence Pooling:
- ``"cls"``: Uses the first code token representation (index 0) in the sequence as the patient summary embedding.
- ``"mean"``: Computes the average representation across all valid (unpadded) tokens according to the token mask.

.. autoclass:: pyhealth.models.BEHRTEmbedding
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.BEHRTLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.BEHRT
    :members:
    :undoc-members:
    :show-inheritance:
