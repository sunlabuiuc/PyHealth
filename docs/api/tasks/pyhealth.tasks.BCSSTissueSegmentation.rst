pyhealth.tasks.BCSSTissueSegmentation
=====================================

Five-class semantic segmentation of BCSS pathology tiles, the benchmark task
of the BCSS paper: other, tumor, stroma, inflammatory infiltrate and necrosis,
with rare codes grouped as in the `BCSS baseline
<https://bcsegmentation.grand-challenge.org/Baseline/>`_ and outside-ROI
pixels ignored by the loss. Build the dataset with every class
(``class_codes=BCSS_CLASS_CODES``) and a low ``min_purity``, train a
:class:`~pyhealth.models.UNet2D`, and use ``BCSSDataset.official_split()`` for
the official site-held-out test set. See
``examples/pathology/bcss_tissue_segmentation_unet.py``.

.. autoclass:: pyhealth.tasks.BCSSTissueSegmentation
    :members:
    :undoc-members:
    :show-inheritance:

.. autodata:: pyhealth.tasks.bcss_tissue_segmentation.BCSS_SEGMENTATION_GROUPS
    :no-value:
