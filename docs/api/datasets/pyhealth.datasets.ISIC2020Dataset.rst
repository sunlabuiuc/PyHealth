pyhealth.datasets.ISIC2020Dataset
=================================

Overview
--------

SIIM-ISIC 2020 melanoma dermoscopy dataset: 33,126 training images from 2,056
patients, labelled benign or malignant (1.8% malignant). The PyHealth patient
is the ISIC ``patient_id``. The 425 listed duplicate images are dropped by
default. Labels of the challenge test set were never released, so only the
training set is indexed.

Getting the data
----------------

- **Source:** `ISIC 2020 challenge data <https://challenge.isic-archive.com/data/#2020>`_
  (https://challenge2020.isic-archive.com/ redirects there).
- **Citation** (both are asked for on the data page):

  - International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge
    Dataset (2020). https://doi.org/10.34970/2020-ds01
  - Rotemberg, V., Kurtansky, N., Betz-Stablein, B., et al. A patient-centric
    dataset of images and metadata for identifying melanomas using clinical
    context. *Scientific Data* 8, 34 (2021).
    https://doi.org/10.1038/s41597-021-00815-z

- **Access:** open. No account, registration or data-use agreement; the files
  are plain HTTPS links on the ISIC S3 bucket.
- **Licence:** CC BY-NC 4.0 (attribution, non-commercial use only; derivatives
  allowed).
- **Files**, from ``https://isic-archive.s3.amazonaws.com/challenges/2020/``:
  ``ISIC_2020_Training_JPEG.zip`` (24.7 GB; the page shows 23 GiB),
  ``ISIC_2020_Training_GroundTruth_v2.csv`` (2.4 MB; the older
  ``ISIC_2020_Training_GroundTruth.csv``, without lesion ids, is also accepted)
  and ``ISIC_2020_Training_Duplicates.csv`` (12 KB). The DICOM archives and
  the unlabelled test set are not used.

  .. code-block:: bash

      BASE=https://isic-archive.s3.amazonaws.com/challenges/2020
      mkdir isic2020 && cd isic2020
      wget "$BASE/ISIC_2020_Training_GroundTruth_v2.csv"
      wget "$BASE/ISIC_2020_Training_Duplicates.csv"
      wget "$BASE/ISIC_2020_Training_JPEG.zip"
      unzip -q ISIC_2020_Training_JPEG.zip   # -> train/

  Or pass ``download=True``, which downloads and unzips the missing files into
  ``root``. A ``train/`` folder with fewer JPEGs than the ground-truth CSV has
  rows (for example after an interrupted unzip) counts as missing: the zip is
  downloaded and extracted again, with a warning.
- **Unpack:** the zip extracts to a ``train/`` folder of 33,126
  ``ISIC_xxxxxxx.jpg`` files; the zip can then be deleted.
- **root layout:**

  .. code-block:: text

      isic2020/                                <- root
          ISIC_2020_Training_GroundTruth_v2.csv
          ISIC_2020_Training_Duplicates.csv
          train/ISIC_0015719.jpg ...

``root`` is only read, unless ``download=True``, which writes the CSVs and the
``train/`` folder into it. The derived metadata CSV goes to ``metadata_dir``,
and is rebuilt when ``root``, ``drop_duplicates`` or the number of JPEGs in
``train/`` changes, so images restored after a partial unzip are picked up.

API Reference
-------------

.. autoclass:: pyhealth.datasets.ISIC2020Dataset
    :members:
    :undoc-members:
    :show-inheritance:
