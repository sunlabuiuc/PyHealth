pyhealth.datasets.BCSSDataset
=============================

Overview
--------

Breast Cancer Semantic Segmentation (BCSS): 151 annotated TCGA-BRCA regions of
interest at 0.25 MPP, one per patient, with pixel masks of 21 tissue classes
(mask value 0 is "don't care"). Open access, data under CC0 1.0. The loader
cuts each ROI into fixed-size tiles labelled by their dominant class, with a
configurable recipe (``tile_size``, ``stride``, ``min_purity``,
``class_codes``, ``image_dir``), and exposes the official site-held-out test
split through ``official_split()``. ``root`` must hold ``masks/`` and
``rgbs_colorNormalized/`` (or ``images/``). The tiling recipe is this
loader's own and does not reproduce CLIMB's numbers.

Source: `BCSS on GitHub <https://github.com/PathologyDataScience/BCSS>`_

API Reference
-------------

.. autoclass:: pyhealth.datasets.BCSSDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.bcss.tile_label

.. autofunction:: pyhealth.datasets.bcss.parse_bcss_roi_name
