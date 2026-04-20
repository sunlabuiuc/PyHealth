pyhealth.models.TransEHR
==========================

.. note::

   This implementation provides a simplified supervised TransEHR-style backbone
   for PyHealth. It includes dual Transformer encoders for multivariate time
   series and clinical events, followed by pooled fusion and a prediction head.

   It does not implement the full training pipeline from the original paper,
   such as self-supervised pretraining heads or Transformer Hawkes Process
   objectives.

   The constructor flag ``use_event_stream`` can be used for a simple ablation.
   When set to ``False``, the event branch is skipped and the model uses only
   the multivariate stream and optional static features.

.. autoclass:: pyhealth.models.TransEHR
    :members:
    :undoc-members:
    :show-inheritance:
