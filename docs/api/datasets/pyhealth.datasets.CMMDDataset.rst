pyhealth.datasets.CMMDDataset
=============================

The Chinese Mammography Database (CMMD) from The Cancer Imaging Archive. The
loader reads patient, laterality and view from each DICOM header and joins the
image to the clinical row of its own breast.

Getting the data
----------------

- **Source**: `TCIA CMMD collection page
  <https://www.cancerimagingarchive.net/collection/cmmd/>`_ (data DOI
  `10.7937/tcia.eqde-4b16 <https://doi.org/10.7937/tcia.eqde-4b16>`_). Cite
  Cui et al. (2021), *The Chinese Mammography Database (CMMD): An online
  mammography database with biopsy confirmed types for machine diagnosis of
  breast*, The Cancer Imaging Archive.
- **Access**: open. No account, registration or data-use agreement; follow
  the TCIA Data Usage Policy and cite the data set.
- **Clinical sheet** (70 KB):
  ``wget https://www.cancerimagingarchive.net/wp-content/uploads/CMMD_clinicaldata_revision.xlsx``
  into ``root``. Reading it needs ``openpyxl``; a CSV export named
  ``CMMD_clinicaldata_revision.csv`` works without it. The sheet must have
  the columns ``ID1``, ``LeftRight`` and ``classification``. ``openpyxl`` is
  not in PyHealth's test environment, so the ``.xlsx`` read test
  (``test_xlsx_clinical_sheet``) runs only where it is installed; CI covers
  the CSV path, which shares all code after the read.
- **DICOMs** (22.86 GB, 1,775 patients, 5,202 images): install the `NBIA Data
  Retriever <https://www.cancerimagingarchive.net/tcia-data-retriever/>`_,
  download the manifest
  ``https://www.cancerimagingarchive.net/wp-content/uploads/The-Chinese-Mammography-Database.tcia``,
  open it in the Data Retriever and download into ``root``. There is no
  ``download=True``.
- **Unpacking**: nothing to unpack. Reading DICOM headers needs ``pydicom``
  (``pip install pyhealth[imaging]``).
- **root layout**: every ``*.dcm`` under ``root`` is read, at any depth,
  so the folder names the Data Retriever writes do not matter. For example,
  a download may look like this (not verified against a fresh Data
  Retriever run):

  .. code-block:: text

      root/
        CMMD_clinicaldata_revision.xlsx   # or .csv, or clinical_file=
        <any>/CMMD/<ID1>/.../*.dcm         # example Data Retriever tree

- **Licence**: CC BY 4.0 for the images and the clinical data (no NC or ND
  terms).

.. autoclass:: pyhealth.datasets.CMMDDataset
    :members:
    :undoc-members:
    :show-inheritance:
