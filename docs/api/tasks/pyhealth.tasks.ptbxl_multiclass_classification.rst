pyhealth.tasks.ptbxl_multiclass_classification
==============================================

Overview
--------

3-way multi-class ECG classification on PTB-XL, following the multi-class
experiment of *In-depth Benchmarking of Deep Neural Network Architectures
for ECG Diagnosis* (Nonaka & Seita, 2021): each record is assigned to
exactly one of ``normal`` (the PTB-XL ``NORM`` statement), ``target`` (one
of AF, IAVB, LBBB, RBBB, PAC, PVC, STD), or ``other``. The target class
takes precedence over normal. Records with no SCP statements are dropped.

Scaling statistics come from the training folds only
(``standardize="train_stats"`` + ``stats_path`` from
:func:`pyhealth.datasets.compute_ptbxl_lead_stats`). The five benchmark
fold pairs are provided as ``PTBXL_PAPER_FOLD_SPLITS`` and consumed by
:func:`pyhealth.datasets.split_by_strat_fold`. Multi-class loss weights
(``[1, n_target/n_normal, n_other/n_normal]`` from the training split) are
computed with :func:`pyhealth.tasks.compute_multiclass_class_weights` and
passed to :class:`pyhealth.models.ResNet1D` as ``class_weight``.

API Reference
-------------

.. automodule:: pyhealth.tasks.ptbxl_multiclass_classification
    :members:
    :undoc-members:
    :show-inheritance:
