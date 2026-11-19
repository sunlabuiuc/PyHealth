pyhealth.models.BiteNet
===================================

The AttentionPooling, BiteNetTemporalEncoding, BiteNetEncoder, and BiteNet model for electronic health records.

Paper: Chen, Xueping, et al. "BiteNet: Bidirectional Temporal Encoder Network to Predict Medical Outcomes."
IEEE International Conference on Data Mining (ICDM 2020): 412-421.
Paper link: https://ieeexplore.ieee.org/document/9378278

Temporal Interval Features:
- ``interval`` or ``time``: Inter-visit time intervals in days or integer timestamps (tensor of shape ``[batch_size, seq_len]`` or ``[batch_size]``).
  When supplied in the input batch, it maps each visit to a temporal interval encoding.
- When omitted from inputs, BiteNet operates as a position-aware bidirectional Transformer encoder using sequence positional encodings.

Sequence Pooling:
- ``"attention"`` (default): Uses attention pooling over the forward and backward directional hidden representations.
- ``"mean"``: Computes the average representation across all valid (unpadded) tokens according to the token mask.
- ``"cls"``: Uses the last valid token representation for forward (which sees all preceding visits) and index 0 for backward (which sees all succeeding visits).

.. autoclass:: pyhealth.models.AttentionPooling
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.BiteNetTemporalEncoding
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.BiteNetEncoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.BiteNet
    :members:
    :undoc-members:
    :show-inheritance:
