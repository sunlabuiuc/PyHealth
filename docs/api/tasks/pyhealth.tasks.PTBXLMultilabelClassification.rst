pyhealth.tasks.PTBXLMultilabelClassification
============================================

Multi-label classification of PTB-XL 12-lead ECGs (all 71 SCP statements, or the
diagnostic, sub/super-diagnostic, form or rhythm label sets). Supports
train-fold per-lead normalization via ``standardize="train_stats"`` and
``stats_path`` from :func:`pyhealth.datasets.compute_ptbxl_lead_stats`.

The SCP statement sets are severely imbalanced. As an extension beyond the
Nonaka & Seita benchmark baseline (which uses unweighted BCE),
:func:`pyhealth.tasks.compute_multilabel_pos_weight` counts each label's
negatives/positives on the training split, for
``pyhealth.models.ResNet1D(pos_weight=...)``.

.. autoclass:: pyhealth.tasks.PTBXLMultilabelClassification
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.compute_multilabel_pos_weight
