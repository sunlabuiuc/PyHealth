pyhealth.models.ResNet1D
========================

1-D ResNet ECG classifier reproducing the backbone and prediction head of
*In-depth Benchmarking of Deep Neural Network Architectures for ECG
Diagnosis* (Nonaka & Seita, 2021,
`arXiv:2112.13779 <https://arxiv.org/abs/2112.13779>`_). Supports ResNet-18,
-34 and -50 layouts via the ``depth`` argument and pairs with the PTB-XL
tasks ``PTBXLMultilabelClassification`` and
``PTBXLMulticlassClassification``.

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
