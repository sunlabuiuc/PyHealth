pyhealth.datasets.CBISDDSMDataset
=================================

CBIS-DDSM curated screening mammography dataset from The Cancer Imaging
Archive. The loader maps the four official case-description CSVs to one event
per annotated lesion, keyed to full-field mammogram DICOMs by series UID, and
keeps the official train/test split.

Getting the data
----------------

- **Source**: `TCIA CBIS-DDSM collection page
  <https://www.cancerimagingarchive.net/collection/cbis-ddsm/>`_ (data DOI
  `10.7937/K9/TCIA.2016.7O02S9CY <https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY>`_).
  Paper: Lee et al., *A curated mammography data set for use in
  computer-aided detection and diagnosis research*, Scientific Data 4, 170177
  (2017), `doi:10.1038/sdata.2017.177 <https://doi.org/10.1038/sdata.2017.177>`_.
- **Access**: open. No account, registration or data-use agreement; follow
  the TCIA Data Usage Policy and cite the data set and the paper.
- **CSVs** (about 2 MB): ``download=True`` fetches the four
  ``{mass,calc}_case_description_{train,test}_set.csv`` files, or run
  ``wget https://www.cancerimagingarchive.net/wp-content/uploads/<name>.csv``
  for each.
- **DICOMs** (163.5 GB for the whole collection): install the `NBIA Data
  Retriever <https://www.cancerimagingarchive.net/tcia-data-retriever/>`_,
  download a ``.tcia`` manifest from the collection page, open it in the Data
  Retriever and download into ``root``. The four "full_mammogram_images"
  manifests in the page's "Detailed Description" section are enough; ROI masks
  and crops are not used. ``CBISDDSMDataset.download_series(uids)`` fetches a
  few series over the NBIA REST API instead.
- **Unpacking**: nothing to unpack.
- **root layout**:

  .. code-block:: text

      root/
        mass_case_description_train_set.csv
        mass_case_description_test_set.csv
        calc_case_description_train_set.csv
        calc_case_description_test_set.csv
        # then the DICOMs, in any one of these layouts:
        dicom/<SeriesUID>/*.dcm               # download_series()
        <case>/<StudyUID>/<SeriesUID>/*.dcm   # CSV image paths
        <any>/metadata.csv                    # Data Retriever output,
        <any>/CBIS-DDSM/...                   # found via metadata.csv

- **Licence**: CC BY 3.0 (no NC or ND terms).

.. autoclass:: pyhealth.datasets.CBISDDSMDataset
    :members:
    :undoc-members:
    :show-inheritance:
