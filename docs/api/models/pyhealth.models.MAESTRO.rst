pyhealth.models.MAESTRO
=======================

Overview
--------

MAESTRO is a multimodal time-series model. Each input feature is one modality,
encoded with ProbSparse attention and distilling convolutions. The encoded
modalities are concatenated along time and fused by a stack with a sparse
mixture-of-experts feed-forward layer. Training randomly drops whole modalities
so the model is robust to missing sensors.

Reference: Mohapatra et al., "MAESTRO: Adaptive Sparse Attention and Robust
Learning for Multimodal Dynamic Time Series." NeurIPS 2025.
https://arxiv.org/abs/2509.25278

API Reference
-------------

.. autoclass:: pyhealth.models.MAESTRO
   :members:
   :undoc-members:
   :show-inheritance:
