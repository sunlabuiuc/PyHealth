pyhealth.datasets.BrainTumorMRIDataset
======================================

Brain Tumor Classification (MRI) dataset of Bhuvaji et al.: 3,264 brain MRI
slices in four classes (no tumour, glioma, meningioma, pituitary) with an
official Training / Testing split. The loader hashes every image, groups
byte-identical copies under one ``patient_id`` and, by default, keeps one
copy per group, so no test image is also a training image.

.. autoclass:: pyhealth.datasets.BrainTumorMRIDataset
    :members:
    :undoc-members:
    :show-inheritance:
