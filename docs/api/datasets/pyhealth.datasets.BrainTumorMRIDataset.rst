pyhealth.datasets.BrainTumorMRIDataset
======================================

Brain Tumor Classification (MRI) dataset of Bhuvaji et al.: 3,264 brain MRI
slices in four classes (no tumour, glioma, meningioma, pituitary) with an
official Training / Testing split. The loader hashes every image, groups
byte-identical copies under one ``patient_id`` and, by default, keeps one
copy per group, so no test image is also a training image.

Getting the data
----------------

- **Source**: `GitHub repository
  <https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet>`_
  (archived), also on `Kaggle
  <https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri>`_
  with the same 3,264 files (93 MB). Cite Bhuvaji et al., *Brain Tumor
  Classification (MRI)*, Kaggle, 2020,
  `doi:10.34740/KAGGLE/DSV/1183165 <https://doi.org/10.34740/KAGGLE/DSV/1183165>`_.
- **Access**: open. GitHub needs no account; Kaggle needs a free account and,
  for the CLI, an API token in ``~/.kaggle/kaggle.json`` and ``pip install kaggle``.
- **Download**, either
  ``git clone --depth 1 https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet``
  (175 MB on disk with ``.git``) or
  ``kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri -p <root> --unzip``.
- **Unpacking**: none for the clone; ``--unzip`` (or ``unzip
  brain-tumor-classification-mri.zip -d <root>``) for Kaggle.
- **root layout**:

  .. code-block:: text

      root/
        Training/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}/*.jpg
        Testing/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}/*.jpg

- **Licence**: MIT, as given on the Kaggle listing. The archived GitHub
  repository carries no LICENSE file.

.. autoclass:: pyhealth.datasets.BrainTumorMRIDataset
    :members:
    :undoc-members:
    :show-inheritance:
