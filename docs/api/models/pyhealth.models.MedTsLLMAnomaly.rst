pyhealth.models.MedTsLLMAnomaly
===================================

MedTsLLM (`Chan et al., MLHC 2024 <https://arxiv.org/abs/2408.07773>`_) for
reconstruction-based anomaly detection in multichannel signals such as the
two-lead MIT-BIH ECG. Windows are normalized per instance (RevIN), cut into
overlapping patches, and each patch is mapped into a frozen LLM's input
embedding space by cross-attention onto text prototypes built from the LLM's
word embeddings. A text prompt (dataset and task description, optionally a
patient description) is prepended, and a linear head reconstructs the window
from the LLM's outputs at the patch positions. Only these adapter layers are
trained.

This is a port of the authors' implementation restricted to the paper's
anomaly detection configuration; with the same weights and inputs it computes
the same outputs. The model reports ``mode == "regression"`` (reconstruction
against the standardized input), so it trains with the standard ``Trainer``.
Score the reconstructions with :mod:`pyhealth.metrics.anomaly_detection`.
See ``examples/mitbih_anomaly_detection_medtsllm.py`` for the full pipeline
with :class:`~pyhealth.datasets.MITBIHDataset` and
:class:`~pyhealth.tasks.MITBIHAnomalyDetection`.

.. autoclass:: pyhealth.models.MedTsLLMAnomaly
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.medtsllm_anomaly.RevIN
    :members:

.. autoclass:: pyhealth.models.medtsllm_anomaly.PatchEmbedding
    :members:

.. autoclass:: pyhealth.models.medtsllm_anomaly.ReprogrammingLayer
    :members:
