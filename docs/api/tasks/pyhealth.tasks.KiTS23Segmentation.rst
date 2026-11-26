pyhealth.tasks.KiTS23Segmentation
=================================

The KiTS23 challenge task: 3D segmentation of kidneys, kidney tumours and
cysts in contrast-enhanced CT (labels 0 background, 1 kidney, 2 tumour,
3 cyst). The CT and mask go through the ``nifti`` processor and fit
:class:`~pyhealth.models.UNet3D`.

.. autoclass:: pyhealth.tasks.KiTS23Segmentation
    :members:
    :undoc-members:
    :show-inheritance:
