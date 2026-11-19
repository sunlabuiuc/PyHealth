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
loader's own and does not reproduce CLIMB's numbers. Each tile is saved
with its mask crop, so the same tiles serve two tasks:
:class:`~pyhealth.tasks.BCSSTileClassification` (dominant tissue class,
CLIMB-style) and :class:`~pyhealth.tasks.BCSSTissueSegmentation` (the BCSS
paper's five-class pixel segmentation, with a U-Net).

Getting the data
----------------

- **Source page:** `BCSS on GitHub <https://github.com/PathologyDataScience/BCSS>`_
  (the README holds the download links).
- **Paper:** Amgad M, Elfandy H, Hussein H, et al. Structured crowdsourcing
  enables convolutional segmentation of histology images. *Bioinformatics*
  35(18):3461-3467 (2019). https://doi.org/10.1093/bioinformatics/btz083
- **Access:** open. No account, registration or data-use agreement.
- **Download, option A** (what this loader was tested on): the public
  `Google Drive folder <https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss>`_
  linked under "Download (single link - convenient)" in the README, at
  0.25 MPP. It holds ``masks/`` and ``rgbs_colorNormalized/`` (151 PNGs
  each), ``meta/`` and ``logs/``. In the browser, right-click ``masks`` and
  ``rgbs_colorNormalized`` (and optionally ``meta``), choose **Download**,
  and unzip every ``.zip`` part Drive produces into the same folder. The
  total size is not shown on the source page. ``gdown --folder`` stops at 50
  files per folder, so it does not fetch the full release.
- **Download, option B:** the README's command-line script, which pulls the
  ROIs from the public Digital Slide Archive and writes raw (not
  stain-normalised) RGBs to ``images/``:

  .. code-block:: bash

      git clone https://github.com/PathologyDataScience/BCSS
      cd BCSS
      pip install girder_client pillow numpy scikit-image imageio
      python download_crowdsource_dataset.py   # set SAVEPATH in configs.py

  Point ``root`` at ``SAVEPATH``; the loader then uses ``images/``.
- **Layout:** mask and RGB file names must match one to one; a missing
  partner raises ``FileNotFoundError``.

  .. code-block:: text

      <root>/
          masks/
              TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png
              ...                                  # 151 masks
          rgbs_colorNormalized/                    # or images/
              TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png
              ...                                  # 151 RGBs
          meta/gtruth_codes.tsv                    # optional

  Tiles are written to the PyHealth cache (or ``metadata_dir``), never into
  ``root``.
- **Licence:** data under `CC0 1.0 <https://creativecommons.org/publicdomain/zero/1.0/>`_
  (public domain, no NC or ND terms; the authors ask that you cite the
  paper). Code in the BCSS repository is MIT.

API Reference
-------------

.. autoclass:: pyhealth.datasets.BCSSDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.bcss.tile_label

.. autofunction:: pyhealth.datasets.bcss.parse_bcss_roi_name
