pyhealth.models.CAMELOTModule
===================================

CAMELOT: Cluster-based Feature Importance for Electronic Health Record Time-series.

Paper: Yingzhen Li, Yingtao Tian, Yuexian Zou, Chaoqi Yang, and Jimeng Sun.
CAMELOT: Cluster-based Feature Importance for Electronic Health Record Time-series.
In International Conference on Machine Learning, pages 11817-11827. PMLR, 2022.

This model learns cluster-based feature importance for EHR time-series data.
It uses an attention-based RNN encoder to extract temporal patterns and
identifies patient clusters with distinct clinical phenotypes.

The model consists of three main components:
- Encoder: Attention-based RNN encoder for temporal feature extraction
- Identifier: MLP that assigns patients to clusters
- Predictor: MLP that predicts clinical outcomes for each cluster

This implementation is adapted from the CTPD repository:
https://github.com/HKU-MedAI/CTPD

.. autoclass:: pyhealth.models.CAMELOTModule
    :members:
    :undoc-members:
    :show-inheritance:

