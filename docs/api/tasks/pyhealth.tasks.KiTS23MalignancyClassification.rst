pyhealth.tasks.KiTS23MalignancyClassification
=============================================

Benign vs malignant kidney tumour classification on KiTS23 CTs, from the
whole resampled volume (3D) or from the three axial slices through the
largest tumour cross-section (2.5D).

The 3D task is the dataset's ``default_task`` but has no example: PyHealth
has no basic 3D classifier yet (``CNN`` rejects ``nifti`` inputs and
``TorchvisionModel`` is 2D only). The example
``examples/ct/kits23_malignancy_classification_resnet18.py`` trains
ResNet-18 on the 2.5D task instead.

.. autoclass:: pyhealth.tasks.KiTS23MalignancyClassification
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.KiTS23MalignancySliceClassification
    :members:
    :undoc-members:
    :show-inheritance:
