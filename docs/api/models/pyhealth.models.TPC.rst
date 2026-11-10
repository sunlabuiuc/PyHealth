pyhealth.models.TPC
===================================

The separate callable TPCBlock and the complete TPC model.

Overview
--------
Temporal Pointwise Convolutional (TPC) networks are designed for remaining length of
stay prediction and in-hospital mortality prediction in intensive care units, adapted
from:

    Emma Rocheteau, Pietro Liò, and Stephanie Hyland.
    "Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the
    Intensive Care Unit."
    ACM Conference on Health, Inference, and Learning (CHIL), 2021.
    DOI: https://doi.org/10.1145/3450439.3451860

The model consumes multi-modal clinical inputs:

- ``vitals``: Continuous dynamic time-series features (shape: ``(B, T, F)``).
- ``vitals_decay``: Exponential decay indicators ``0.75^j`` (shape: ``(B, T, F)``).
- ``static``: Baseline flat demographics and admission features (shape: ``(B, S)``).
- ``time_features``: Non-convolved temporal context features (e.g. elapsed ICU days
  :math:`(t+1)/24` and cyclic hour-of-day in :math:`[0, 1]`) (shape: ``(B, T, 2)``).
- ``mask``: Observation presence indicators (shape: ``(B, T)`` or ``(B, T, F)``).

Batch Normalization Variants:
The model supports configurable 1D batch normalization across temporal and pointwise
convolution branches via the ``batchnorm`` argument:

- ``"mybatchnorm"``: Custom BatchNorm1d that flattens sequence dimensions matching
  Rocheteau et al. (official eICU configuration).
- ``"default"``: Standard PyTorch ``nn.BatchNorm1d``.
- ``"pointonly"``: BatchNorm applied only on pointwise convolution layers.
- ``"temponly"``: BatchNorm applied only on temporal convolution layers.
- ``"low_momentum"``: Standard BatchNorm1d with momentum=0.01.
- ``"none"``: Identity mapping (no normalization).

Architectural Ablations & Prediction Modes:
The model supports the full suite of architectural variants from Rocheteau et al.
(Table 3 & Table 4):

- ``share_weights`` (bool): When True, shares temporal convolution weights across
  features (``groups=1``), ablating per-feature independent temporal processing
  (corresponds to 'Temp. only (WS)' in Table 3).
- ``use_skip`` (bool): When False, disables residual skip connections from raw
  features and prior pointwise layers, reproducing 'TPC (no skip)'.
- ``mode``: Supports both ``"regression"`` (LoS prediction) and ``"binary"``
  (single-task in-hospital mortality prediction matching Table 4 Row 1).
- ``decay_rate`` (float): Exponential decay base rate (default: 0.75).
- ``seq_to_seq`` (bool): When True (default), emits hourly predictions of shape
  ``(B, T, 1)``. When False, pools to the final valid time step of shape ``(B, 1)``
  for standard PyHealth visit-level tasks and benchmarks.

Outputs:

- ``y_prob``: Predicted remaining length of stay in days or mortality probabilities
  (shape: ``(B, T, 1)`` if ``seq_to_seq=True`` else ``(B, 1)``).
- ``logit``: Raw linear projection (shape: ``(B, T, 1)`` if ``seq_to_seq=True`` else
  ``(B, 1)``).
- ``loss``: MSLE, MSE, or BCE loss across valid stay hours / stays.
- ``mask``: Evaluated boolean mask tensor (shape: ``(B, T, 1)``)
  when ``seq_to_seq=True`` and mask is provided.
- ``aux_prob``: Auxiliary classification probabilities when ``aux_label_key`` is
  configured.
- ``aux_logit``: Raw auxiliary logits when ``aux_label_key`` is configured.
- ``aux_loss``: Auxiliary binary cross-entropy loss when auxiliary labels are provided.

.. autoclass:: pyhealth.models.TPCBlock
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TPC
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.MyBatchNorm1d
    :members:
    :undoc-members:
    :show-inheritance:

