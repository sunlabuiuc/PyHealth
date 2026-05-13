pyhealth.datasets.ISIC2018ArtifactsDataset
======================================

A dataset class for dermoscopy images paired with per-image artifact
annotations.  The default annotation file is ``isic_bias.csv`` from
Bissoto et al. (2020), but any CSV following the same column format can be
supplied via the ``annotations_csv`` parameter.

**Default data sources (Bissoto et al. 2020)**

Using ``ISIC2018ArtifactsDataset`` with the default annotation CSV requires
**two separate downloads**:

1. **Artifact annotations** (``isic_bias.csv``):
   https://github.com/alceubissoto/debiasing-skin

   See ``artefacts-annotation/`` in that repository for the annotation files.

   Reference:
   Bissoto et al. "Debiasing Skin Lesion Datasets and Models? Not So Fast",
   ISIC Skin Image Analysis Workshop @ CVPR 2020.

2. **ISIC 2018 Task 1/2 images & masks** (~8 GB):
   https://challenge.isic-archive.com/data/#2018

   * Training images: ``ISIC2018_Task1-2_Training_Input.zip``
   * Segmentation masks: ``ISIC2018_Task1_Training_GroundTruth.zip``

Both can be fetched automatically by passing ``download=True``.

.. autoclass:: pyhealth.datasets.ISIC2018ArtifactsDataset
    :members:
    :undoc-members:
    :show-inheritance:
