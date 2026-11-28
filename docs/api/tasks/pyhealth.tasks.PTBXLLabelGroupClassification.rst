pyhealth.tasks.PTBXLLabelGroupClassification
============================================

Multi-label classification of PTB-XL 12-lead ECGs under a configurable
grouping of the 71 SCP-ECG statements, as used by the branches of ProtoECGNet
(`Sethi et al., MLHC 2025 <https://arxiv.org/abs/2504.08713>`_). The default
is the paper's physician grouping (16 rhythm, 52 morphology and 3 global
labels); ``group`` keeps a single group, which reproduces the per-branch
training sets. The helper functions build data-driven (spectral clustering of
the Jaccard label co-occurrence) and size-matched random groupings, so the
physician grouping can be tested against alternatives.

Use it with :doc:`../datasets/pyhealth.datasets.PTBXLDataset` (download PTB-XL
1.0.3 from `PhysioNet <https://physionet.org/content/ptb-xl/1.0.3/>`_).

.. autoclass:: pyhealth.tasks.PTBXLLabelGroupClassification
    :members:
    :undoc-members:
    :show-inheritance:

Label-grouping helpers
----------------------

.. autodata:: pyhealth.tasks.ptbxl_label_group_classification.PHYSICIAN_LABEL_GROUPS
    :no-value:

.. autofunction:: pyhealth.tasks.ptbxl_label_group_classification.validate_label_groups

.. autofunction:: pyhealth.tasks.ptbxl_label_group_classification.label_cooccurrence

.. autofunction:: pyhealth.tasks.ptbxl_label_group_classification.spectral_label_groups

.. autofunction:: pyhealth.tasks.ptbxl_label_group_classification.random_label_groups

.. autofunction:: pyhealth.tasks.ptbxl_label_group_classification.match_label_groups
