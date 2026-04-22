pyhealth.tasks.ecg_classification
==================================

Overview
--------
Binary ECG classification task for the PTB-XL dataset, implementing the
task interface from Raghu et al. (2022) *Data Augmentation for
Electrocardiograms* (CHIL, PMLR 174). Each ECG record is loaded from disk
via WFDB, per-lead z-score normalised, and padded or truncated to a fixed
time length. Supports four diagnostic superclasses: MI, HYP, STTC, and CD.

Paper: https://proceedings.mlr.press/v174/raghu22a.html

API Reference
-------------

.. autoclass:: pyhealth.tasks.ecg_classification.ECGBinaryClassification
   :members:
   :undoc-members:
   :show-inheritance:
