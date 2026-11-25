pyhealth.models.ResNet1D
========================

1-D ResNet ECG classifier reproducing the backbone and prediction head of
*In-depth Benchmarking of Deep Neural Network Architectures for ECG
Diagnosis* (Nonaka & Seita, 2021,
`PMLR 149:414-439 <https://proceedings.mlr.press/v149/nonaka21a.html>`_).
Supports ResNet-18, -34 and -50 layouts via the ``depth`` argument and
pairs with the PTB-XL tasks ``PTBXLMultilabelClassification`` and
``PTBXLMulticlassClassification``. The benchmark's training/inference
protocol is configured on the model: ``training_crop_len=1250`` (random
2.5 s crop while training) and ``eval_window_len=1250`` /
``eval_window_stride=625`` (sliding-window evaluation with per-class max
aggregation) at 500 Hz; ``class_weight`` sets the multi-class loss weights.
See
``examples/ecg/ptbxl/ptbxl_multilabel_classification_resnet1d.py``.

.. autoclass:: pyhealth.models.ResNet1D
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.BasicBlock1D
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.Bottleneck1D
    :members:
    :undoc-members:
    :show-inheritance:
