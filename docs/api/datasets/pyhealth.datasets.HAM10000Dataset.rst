pyhealth.datasets.HAM10000Dataset
=================================

Overview
--------

HAM10000 dermoscopy dataset as released for ISIC 2018 Task 3: 10,015 training
images of 7,470 lesions in seven diagnostic classes, plus the official
validation (193) and test (1,512) sets. The PyHealth patient is the lesion, so
patient-level splits keep all images of a lesion together. All HAM10000 images
are also in ISIC 2019, so do not train on one and test on the other.

Getting the data
----------------

- **Source:** `ISIC 2018 challenge data <https://challenge.isic-archive.com/data/#2018>`_
  (Task 3). The images are also on
  `Harvard Dataverse <https://doi.org/10.7910/DVN/DBW86T>`_, but the class
  reads the ISIC 2018 Task 3 layout below.
- **Papers:**

  - Tschandl, P., Rosendahl, C., Kittler, H. The HAM10000 dataset, a large
    collection of multi-source dermatoscopic images of common pigmented skin
    lesions. *Scientific Data* 5, 180161 (2018).
    https://doi.org/10.1038/sdata.2018.161
  - Codella, N., et al. Skin lesion analysis toward melanoma detection 2018: a
    challenge hosted by the International Skin Imaging Collaboration (ISIC).
    arXiv:1902.03368 (2019).

- **Access:** open. No account, registration or data-use agreement; the files
  are plain HTTPS links on the ISIC S3 bucket.
- **Licence:** CC BY-NC 4.0 (attribution, non-commercial use only; derivatives
  allowed), as stated on the ISIC data page, in the ``LICENSE.txt`` inside each
  image zip and on the current Dataverse version.
- **Files**, from ``https://isic-archive.s3.amazonaws.com/challenges/2018/``
  (sizes in decimal units; the ISIC page shows binary ones):

  - training: ``ISIC2018_Task3_Training_Input.zip`` (2.8 GB),
    ``ISIC2018_Task3_Training_GroundTruth.zip`` (37 KB) and
    ``ISIC2018_Task3_Training_LesionGroupings.csv`` (493 KB, required: it
    holds the lesion ids);
  - official holdout, only when ``splits`` asks for it:
    ``ISIC2018_Task3_Validation_Input.zip`` (53 MB),
    ``ISIC2018_Task3_Validation_GroundTruth.zip`` (8 KB),
    ``ISIC2018_Task3_Test_Input.zip`` (421 MB) and
    ``ISIC2018_Task3_Test_GroundTruth.zip`` (12 KB).

  .. code-block:: bash

      BASE=https://isic-archive.s3.amazonaws.com/challenges/2018
      mkdir isic2018_task3 && cd isic2018_task3
      wget "$BASE/ISIC2018_Task3_Training_LesionGroupings.csv"
      for f in Input GroundTruth; do
          wget "$BASE/ISIC2018_Task3_Training_$f.zip"
          unzip -q "ISIC2018_Task3_Training_$f.zip"
      done

  Or pass ``download=True``, which downloads and unzips the missing files of
  the requested ``splits`` into ``root``.
- **Unpack:** each zip extracts to a folder of the same name; the zips can then
  be deleted. Instead of the lesion groupings CSV, the Dataverse
  ``HAM10000_metadata`` file is accepted, saved as ``HAM10000_metadata.csv``:

  .. code-block:: bash

      wget -O HAM10000_metadata.csv \
          "https://dataverse.harvard.edu/api/access/datafile/4338392?format=original"

- **root layout** (validation and test folders only when ``splits`` asks for
  them):

  .. code-block:: text

      isic2018_task3/                                <- root
          ISIC2018_Task3_Training_LesionGroupings.csv
          ISIC2018_Task3_Training_Input/ISIC_0024306.jpg ...
          ISIC2018_Task3_Training_GroundTruth/
              ISIC2018_Task3_Training_GroundTruth.csv
          ISIC2018_Task3_Validation_Input/  ISIC2018_Task3_Validation_GroundTruth/
          ISIC2018_Task3_Test_Input/        ISIC2018_Task3_Test_GroundTruth/

``root`` is only read, unless ``download=True``, which writes the downloaded
files into it. The derived metadata CSV goes to ``metadata_dir``.

API Reference
-------------

.. autoclass:: pyhealth.datasets.HAM10000Dataset
    :members:
    :undoc-members:
    :show-inheritance:
