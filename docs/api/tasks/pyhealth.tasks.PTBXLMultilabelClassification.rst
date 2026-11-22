pyhealth.tasks.PTBXLMultilabelClassification
============================================

Multi-label classification of PTB-XL 12-lead ECGs (all 71 SCP statements, or the
diagnostic, sub/super-diagnostic, form or rhythm label sets). Supports
train-fold per-lead normalization via ``standardize="train_stats"`` and
``stats_path`` from :func:`pyhealth.datasets.compute_ptbxl_lead_stats`.

.. autoclass:: pyhealth.tasks.PTBXLMultilabelClassification
    :members:
    :undoc-members:
    :show-inheritance:
