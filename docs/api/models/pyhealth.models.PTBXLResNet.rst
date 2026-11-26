pyhealth.models.PTBXLResNet
===========================

Overview
--------

A small 1D ResNet for PTB-XL ECG classification, and ``TaskAugResNet``, the
same network preceded by the TaskAug augmentation policy of
`Raghu et al., CHIL 2022 <https://proceedings.mlr.press/v174/raghu22a.html>`_.
The augmentation code in ``pyhealth.models.taskaug`` is adapted from the
authors' repository (https://github.com/aniruddhraghu/ecg_aug, Apache License
2.0). Use the tasks in :mod:`pyhealth.tasks.ptbxl_binary` to build its inputs
and ``examples/ecg/ptbxl/ptbxl_binary_classification_taskaug_resnet.py`` to train it.

.. note::

   ``TaskAugResNet`` applies the augmentation during training but does not
   reproduce the paper's bilevel optimization of the augmentation magnitudes.

API Reference
-------------

.. autoclass:: pyhealth.models.PTBXLResNet
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TaskAugResNet
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ptbxl_resnet.ResBlock1D
    :members:
    :show-inheritance:
