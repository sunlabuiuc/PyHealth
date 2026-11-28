pyhealth.datasets.PolypDataset
==============================

Overview
--------

Polyp bounding boxes on colonoscopy frames: the PraNet benchmark (Fan et al.,
MICCAI 2020), with the boxes MIU-VL (Qin et al., ICLR 2023) derived from its
masks. The test sets are CVC-300 (60 images), CVC-ClinicDB (62), CVC-ColonDB
(380), ETIS-LaribPolypDB (196) and Kvasir (100), 848 polyps in all; the
training set has 1,450 images from Kvasir-SEG and CVC-ClinicDB. Each image is
its own patient.

Getting the data
----------------

- **Boxes:** MIU-VL's ``POLYP.zip`` (50 MB), linked from the
  `MIU-VL repository <https://github.com/MembrAI/MIU-VL>`_
  (`Google Drive <https://drive.google.com/file/d/10ISx1yXxfE20nKq6UqquUAD5Egk3hyqi>`__).
  It holds ``DATA/POLYP`` with the COCO annotations and the training images,
  but empty test image folders.
- **Test images:** PraNet's ``TestDataset.zip`` (327 MB), linked from the
  `PraNet repository <https://github.com/DengPingFan/PraNet>`_
  (`Google Drive <https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao>`__).
- **Papers:**

  - Fan, D.-P., et al. PraNet: Parallel Reverse Attention Network for Polyp
    Segmentation. *MICCAI* 2020. https://arxiv.org/abs/2006.11392
  - Qin, Z., et al. Medical Image Understanding with Pretrained Vision
    Language Models: A Comprehensive Study. *ICLR* 2023.
    https://arxiv.org/abs/2209.15517
  - The source datasets: Jha et al., Kvasir-SEG (*MMM* 2020); Bernal et al.,
    CVC-ClinicDB (*CMIG* 2015) and CVC-ColonDB (*Pattern Recognition* 2012);
    Silva et al., ETIS-Larib (*IJCARS* 2014); Vázquez et al., EndoScene
    (CVC-300; *J. Healthc. Eng.* 2017).

- **Access:** open; both archives are public Google Drive links (``gdown``
  can fetch them). No account or data-use agreement.
- **Licence:** research use, as released by the source datasets' authors
  (Kvasir-SEG is CC BY 4.0). Check each source before other uses.
- **Unpack:** copy PraNet's test images into MIU-VL's folders:

  .. code-block:: bash

      unzip POLYP.zip && unzip TestDataset.zip
      for s in CVC-300 CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB Kvasir; do
          cp TestDataset/$s/images/* DATA/POLYP/val/$s/images/
      done

- **root layout** (``DATA/POLYP``):

  .. code-block:: text

      POLYP/                                  <- root
          annotations/CVC-300_val.json ... Kvasir_val.json
          annotations/Polyp_train.json
          train/images/*.png
          val/CVC-300/images/*.png ...

``root`` is only read. The derived metadata CSV goes to ``metadata_dir``.

API Reference
-------------

.. autoclass:: pyhealth.datasets.PolypDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autodata:: pyhealth.datasets.POLYP_TEST_SETS
