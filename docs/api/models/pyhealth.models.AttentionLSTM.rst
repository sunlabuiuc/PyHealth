pyhealth.models.AttentionLSTM
==============================

Overview
--------

``AttentionLSTM`` implements temporal attention over LSTM hidden states for
clinical time-series classification. The model learns to weight time steps by
importance, producing an interpretable context vector for prediction.

Based on the attention-LSTM architecture evaluated in Yadav & Subbian (CHIL
2025), *When Attention Fails: Pitfalls of Attention-based Model
Interpretability for High-dimensional Clinical Time-Series*, which studies
attention reliability across random initializations.

API Reference
-------------

.. autoclass:: pyhealth.models.AttentionLSTM
    :members:
    :undoc-members:
    :show-inheritance:
