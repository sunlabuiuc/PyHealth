pyhealth.datasets.ISIC2019Dataset
=================================

Overview
--------

ISIC 2019 dermoscopy dataset: 25,331 training images in eight diagnostic
classes from the HAM10000, BCN_20000 and MSK collections, plus the public
8,238-image test set (which adds a ninth class, UNK). The PyHealth patient is
the lesion where the lesion id is known. It contains all of HAM10000 / ISIC
2018 Task 3, flagged per image as ``is_ham10000``.

Getting the data
----------------

- **Source:** `ISIC 2019 challenge data <https://challenge.isic-archive.com/data/#2019>`_.
- **Papers** (the data page asks for all three):

  - Tschandl, P., Rosendahl, C., Kittler, H. The HAM10000 dataset, a large
    collection of multi-source dermatoscopic images of common pigmented skin
    lesions. *Scientific Data* 5, 180161 (2018).
    https://doi.org/10.1038/sdata.2018.161
  - Codella, N., et al. Skin lesion analysis toward melanoma detection: a
    challenge at the 2017 ISBI, hosted by the ISIC. arXiv:1710.05006 (2017).
  - Hernández-Pérez, C., et al. BCN20000: Dermoscopic lesions in the wild.
    *Scientific Data* 11, 641 (2024).
    https://doi.org/10.1038/s41597-024-03387-w

- **Access:** open. No account, registration or data-use agreement; the files
  are plain HTTPS links on the ISIC S3 bucket.
- **Licence:** CC BY-NC 4.0 (attribution, non-commercial use only; derivatives
  allowed).
- **Files**, from ``https://isic-archive.s3.amazonaws.com/challenges/2019/``
  (sizes in decimal GB; the ISIC page shows GiB):

  - training: ``ISIC_2019_Training_Input.zip`` (9.8 GB),
    ``ISIC_2019_Training_GroundTruth.csv`` (1.3 MB) and
    ``ISIC_2019_Training_Metadata.csv`` (1.2 MB, required: it holds the
    lesion ids);
  - test, only for ``splits=("train", "test")``: ``ISIC_2019_Test_Input.zip``
    (3.8 GB), ``ISIC_2019_Test_GroundTruth.csv`` (465 KB) and
    ``ISIC_2019_Test_Metadata.csv`` (294 KB).

  .. code-block:: bash

      BASE=https://isic-archive.s3.amazonaws.com/challenges/2019
      mkdir isic2019 && cd isic2019
      for f in Input.zip GroundTruth.csv Metadata.csv; do
          wget "$BASE/ISIC_2019_Training_$f"
      done
      unzip -q ISIC_2019_Training_Input.zip   # -> ISIC_2019_Training_Input/

  Or pass ``download=True``, which downloads and unzips the missing files of
  the requested ``splits`` into ``root``.
- **Unpack:** each image zip extracts to a folder of the same name of
  ``ISIC_xxxxxxx.jpg`` files; the zips can then be deleted.
- **root layout** (the test files only when ``"test"`` is in ``splits``):

  .. code-block:: text

      isic2019/                          <- root
          ISIC_2019_Training_GroundTruth.csv
          ISIC_2019_Training_Metadata.csv
          ISIC_2019_Training_Input/ISIC_0000000.jpg ...
          ISIC_2019_Test_GroundTruth.csv
          ISIC_2019_Test_Metadata.csv
          ISIC_2019_Test_Input/ISIC_0034321.jpg ...

``root`` is only read, unless ``download=True``, which writes the downloaded
files into it. The derived metadata CSV goes to ``metadata_dir``.

API Reference
-------------

.. autoclass:: pyhealth.datasets.ISIC2019Dataset
    :members:
    :undoc-members:
    :show-inheritance:
