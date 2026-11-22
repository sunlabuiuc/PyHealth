pyhealth.datasets.KiTS23Dataset
===============================

KiTS23 kidney tumour CT dataset: 489 contrast-enhanced abdominal CTs with
kidney / tumour / cyst segmentations. The loader reads ``kits23.json`` into
one event per case with its CT and segmentation paths, the benign /
malignant label and the pathology record (histologic subtype, ISUP grade,
TNM stage).

Source: `KiTS23 challenge <https://kits-challenge.org/kits23/>`_,
`neheller/kits23 on GitHub <https://github.com/neheller/kits23>`_ and the
`KiTS-Challenge-Imaging <https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging>`_
Hugging Face dataset. Paper: Heller et al., "The KiTS21 Challenge: Automatic
segmentation of kidneys, renal tumors, and renal cysts in
corticomedullary-phase CT", arXiv:2307.01984 (2023).

Getting the data
----------------

- **Access**: open, no account, registration or agreement.
- **Files**: the GitHub repository (about 0.5 GB) holds ``kits23.json`` and
  every ``case_XXXXX/segmentation.nii.gz``. The 489 CTs
  (``images/case_XXXXX.nii.gz`` on Hugging Face, about 42 GB in total) are
  downloaded separately into ``case_XXXXX/imaging.nii.gz``.
- **Download**:

  .. code-block:: bash

      git clone https://github.com/neheller/kits23
      cd kits23
      pip3 install -e .
      kits23_download_data     # all CTs into dataset/case_XXXXX/imaging.nii.gz

  For a few cases only, fetch single CTs instead of running
  ``kits23_download_data``:

  .. code-block:: bash

      wget -O dataset/case_00000/imaging.nii.gz \
          https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_00000.nii.gz

  Nothing needs unpacking. Cases without ``imaging.nii.gz`` are kept with
  ``has_imaging = 0`` and skipped by the tasks.
- **root layout** (pass ``kits23/dataset``; the repository root also works):

  .. code-block:: text

      kits23/dataset/
          kits23.json
          case_00000/
              imaging.nii.gz          # from Hugging Face
              segmentation.nii.gz     # from the git clone
              instances/
          case_00001/
          ...

- **Licence**: images and segmentations under CC BY-NC-SA 4.0
  (non-commercial, attribution, share alike); the repository code is MIT.
  Contact the KiTS organisers for commercial use.

Metadata location
-----------------

The loader writes its metadata CSV to ``metadata_dir``, never into
``root``. By default this is ``<cache_dir>/metadata/kits23-<hash>``, where
``<hash>`` is the first 12 hex digits of the SHA-1 of the resolved
``root``, so two roots sharing one ``cache_dir`` keep separate CSVs. An
explicit ``metadata_dir`` is used as given.

API Reference
-------------

.. autoclass:: pyhealth.datasets.KiTS23Dataset
    :members:
    :undoc-members:
    :show-inheritance:
