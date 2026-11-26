pyhealth.tasks.ptbxl_binary
===========================

Overview
--------

Binary single-diagnosis tasks for PTB-XL. Each task labels an ECG ``1`` when
it carries one diagnostic superclass (``MI``, ``HYP``, ``STTC`` or ``CD``) and
``0`` otherwise, using the superclass aggregation of
:mod:`pyhealth.tasks.ptbxl`. They are built on the records produced by
:class:`~pyhealth.datasets.PTBXLDataset`; load the dataset with
``sampling_rate=500`` to use them with :class:`~pyhealth.models.TaskAugResNet`.
Official fold splitting is provided by
:func:`pyhealth.datasets.split_by_strat_fold`.

API Reference
-------------

.. automodule:: pyhealth.tasks.ptbxl_binary
    :members:
    :undoc-members:
    :show-inheritance:
