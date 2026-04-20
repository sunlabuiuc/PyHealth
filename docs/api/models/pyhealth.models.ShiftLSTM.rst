pyhealth.models.ShiftLSTM
========================================


The segment-wise recurrent layer and the complete ShiftLSTM model.

``ShiftLSTM`` relaxes parameter sharing over time by dividing the sequence
into ``K`` temporal segments. Each segment uses its own ``LSTMCell`` while the
hidden and cell states continue flowing through the full sequence. When
``num_segments=1``, the model reduces to the shared-parameter baseline.

This implementation is inspired by:

  Oh, J., Wang, J., Wiens, J. (2019).
  "Relaxed Parameter Sharing: Effectively Modeling Time-Varying Relationships
  in Clinical Time-Series."

.. autoclass:: pyhealth.models.ShiftLSTMLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ShiftLSTM
    :members:
    :undoc-members:
    :show-inheritance:
