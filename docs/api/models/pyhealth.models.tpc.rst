pyhealth.models.tpc
===================

Temporal Pointwise Convolution (TPC) network for ICU length-of-stay prediction
(Rocheteau et al., ACM CHIL 2021).

`Original paper <https://arxiv.org/abs/2007.09483>`_.

The model combines per-feature causal grouped temporal convolutions and
pointwise feature mixing, with growing skip connections, optional static and
diagnosis inputs, and bounded positive remaining-LoS predictions.
``forward`` predicts the final hour; ``forward_sequence`` predicts all hours.
Fixed-length, unnormalized day-valued regression labels are required.

Extensions
----------

``dilation_schedule`` supports ``official`` (released-code default), ``paper``,
``constant``, ``linear`` and ``exponential``. Linear is identical to paper
(``d=n``); constant uses 1 and exponential uses ``2**(n-1)``.

``loss`` supports ``msle``, ``mse``, ``asymmetric_msle`` and ``huber_log``.
Asymmetric MSLE weights negative log errors by ``underprediction_weight``;
Huber uses ``huber_delta`` in log units. All log losses use log(days), not
log1p(days), to match the paper.

Supply ``observed_key`` and ``decay_key=None`` for internal missingness
preparation, or call ``prepare_time_series`` before creating a dataset.
``decay_base`` controls decay from the original observation mask;
``forward_fill_labs=False`` zeros missing lab entries selected by ``lab_indices``
while still forward-filling vitals. Never compute decay from filled values.

``examples/synthetic_icu_los_tpc.py`` contains a reproducible, matched three-seed
one-factor sweep and its measured results. MSE vs. MSLE is a **Sanity Check**;
dataset sizes are a **Compute-Constraint Demonstration**, not novel ablations.
Synthetic results do not reproduce the paper's clinical benchmark performance.
Standard BatchNorm uses running statistics at inference, unlike the released
code's batch-statistics normalization. Only evaluation-mode predictions are
strictly causal: training BatchNorm pools statistics over the time axis.

API
---

.. autoclass:: pyhealth.models.tpc.TPCLayer
    :members:
    :undoc-members:
    :show-inheritance:
    :no-inherited-members:

.. autoclass:: pyhealth.models.tpc.TPC
    :members:
    :undoc-members:
    :show-inheritance:
    :no-inherited-members:

.. autofunction:: pyhealth.models.tpc.compute_decay_indicators

.. autofunction:: pyhealth.models.tpc.msle_loss

.. autofunction:: pyhealth.models.tpc.asymmetric_msle_loss

.. autofunction:: pyhealth.models.tpc.huber_log_loss

.. autofunction:: pyhealth.models.tpc.prepare_time_series
