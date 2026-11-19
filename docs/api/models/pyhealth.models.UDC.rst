pyhealth.models.UDC
===================

Overview
--------

UDC (Zhao et al., `Unveiling Discrete Clues: Superior Healthcare Predictions
for Rare Diseases <https://arxiv.org/abs/2501.16373>`_, The Web Conference
/ WWW 2025) predicts clinical outcomes for rare diseases from longitudinal
Electronic Health Records (EHR).

Predicting rare diseases in EHR is notoriously difficult due to extreme
sample scarcity, long-tail code distributions, and weak co-occurrence patterns
with other medical codes. Standard continuous deep learning architectures are
dominated by frequent conditions, drowning out subtle clinical clues.

UDC addresses this challenge by establishing condition-aware discrete clues
via Residual Vector Quantization (RVQ):

1. **Feature Embedding & Intra-Visit Pooling**: Categorical and numeric
   EHR codes across modalities (conditions, procedures, drugs) are embedded
   into continuous representations via :class:`~pyhealth.models.EmbeddingModel`.
   Nested visits are pooled into visit representations using configurable
   intra-visit pooling (``sum``, ``mean``, ``max``, or ``attention``).
2. **Discrete Clue Encoding**: Continuous visit representations are quantized
   into discrete prototypes across hierarchical codebook stages using
   :class:`~pyhealth.models.udc.ResidualVectorQuantizer`. Continuous and
   discrete clue representations are fused via residual LayerNorm in
   :class:`~pyhealth.models.udc.DiscreteClueEncoder`.
3. **Longitudinal Visit Modeling**: A temporal sequence model (GRU, LSTM,
   or Transformer) aggregates patient trajectories across visits.
4. **Outcome Prediction & Joint Loss**: Patient representations from the last
   valid visit are mapped to outcome logits. The objective function jointly
   optimizes the task loss and the discrete codebook commitment loss.

Inputs and Outputs
------------------

UDC accepts standard PyHealth :class:`~pyhealth.datasets.SampleDataset`
batches. Both 2D sequences (``SequenceProcessor``) and 3D nested sequences
(``NestedSequenceProcessor``) are supported. Binary, multiclass, and
multilabel classification tasks are fully supported. The ``forward`` method
always returns ``y_prob``, ``logit``, ``embed``, ``vq_loss``, and the
discrete clue prototype indices ``vq_indices``. When labels are provided,
``loss`` and ``y_true`` are also returned.

API
---

.. autoclass:: pyhealth.models.UDC
   :members:
   :show-inheritance:

.. autoclass:: pyhealth.models.udc.ResidualVectorQuantizer
   :members:
   :show-inheritance:

.. autoclass:: pyhealth.models.udc.DiscreteClueEncoder
   :members:
   :show-inheritance:
