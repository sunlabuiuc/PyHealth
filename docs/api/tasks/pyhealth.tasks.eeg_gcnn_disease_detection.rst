pyhealth.tasks.eeg_gcnn_nd_detection
=========================================

Neurological disease detection task from the EEG-GCNN paper (Wagh & Varatharajah, ML4H @ NeurIPS 2020).

Binary classification: patient-normal (TUAB) vs healthy-control (LEMON).

The task extracts PSD band-power features and graph adjacency matrices from EEG recordings,
supporting configurable adjacency types, frequency bands, and connectivity measures for ablation studies.

Paper: https://proceedings.mlr.press/v136/wagh20a.html

.. autoclass:: pyhealth.tasks.eeg_gcnn_nd_detection.EEGGCNNDiseaseDetection
    :members:
    :undoc-members:
    :show-inheritance:
