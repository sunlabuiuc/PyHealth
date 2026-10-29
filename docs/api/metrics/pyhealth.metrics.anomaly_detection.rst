pyhealth.metrics.anomaly_detection
===================================

Point-wise evaluation of reconstruction-based time-series anomaly detection,
following MedTsLLM (`Chan et al., MLHC 2024 <https://arxiv.org/abs/2408.07773>`_):
per-step scores from reconstruction errors, a quantile threshold, optional
point adjustment, and the resulting classification metrics. The threshold-free
AUROC of the raw scores is reported as well.

.. currentmodule:: pyhealth.metrics.anomaly_detection

.. autofunction:: reconstruction_anomaly_scores

.. autofunction:: point_adjust

.. autofunction:: anomaly_detection_metrics
