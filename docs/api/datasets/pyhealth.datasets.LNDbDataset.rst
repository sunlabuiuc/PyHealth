pyhealth.datasets.LNDbDataset
=============================

LNDb lung nodule CT dataset. The loader indexes the merged radiologist
findings of the LNDb training set, one event per finding with its world
coordinate, size, texture rating, nodule / non-nodule label and official
cross-validation fold, and reads the ``.mhd`` / ``.raw`` CTs with a
built-in numpy reader (SimpleITK is used when installed).

Source: `Zenodo record 8348419 <https://doi.org/10.5281/zenodo.8348419>`_
(challenge page: https://lndb.grand-challenge.org/). Papers: Pedrosa et al.,
"LNDb: A Lung Nodule Database on Computed Tomography", arXiv:1911.08434
(2019); Pedrosa et al., "LNDb challenge on automatic lung cancer patient
management", Medical Image Analysis 70, 102027 (2021).

Getting the data
----------------

- **Access**: open, no account, registration or agreement.
- **Files** (training set): ``trainset_csv.zip`` (78 KB; ``trainNodules_gt.csv``,
  ``trainFolds.csv``, ``trainCTs.csv``, ``trainNodules.csv``,
  ``trainFleischner.csv``) and the CTs in ``data0.rar`` to ``data5.rar``
  (3.8-4.2 GB each, 24.3 GB in all, about 40 CTs each). ``masks.rar`` and the
  ``test*`` files are not used.
- **Download**:

  .. code-block:: bash

      curl -L -o trainset_csv.zip "https://zenodo.org/records/8348419/files/trainset_csv.zip?download=1"
      for i in 0 1 2 3 4 5; do
          curl -L -o data$i.rar "https://zenodo.org/records/8348419/files/data$i.rar?download=1"
      done

  One archive is enough to try the tasks; findings whose CT is not on disk
  are skipped.
- **Unpack**: ``unzip trainset_csv.zip`` and ``unrar x dataN.rar`` (or
  ``bsdtar -xf dataN.rar``). Python cannot read RAR, so there is no
  ``download=True``.
- **root layout** (CSVs and CTs are found recursively, so folder names do
  not matter):

  .. code-block:: text

      lndb/                          # root
          trainNodules_gt.csv        # required
          trainFolds.csv             # optional: official folds
          ...
          data0/
              LNDb-0001.mhd
              LNDb-0001.raw          # uncompressed int16 CT, ~170 MB
              ...

- **Licence**: ambiguous. The Zenodo licence field says CC BY 4.0, but the
  record's description says CC BY-NC-ND 4.0 (non-commercial, no
  derivatives). Follow the stricter NC-ND terms unless the LNDb authors say
  otherwise.

API Reference
-------------

.. autoclass:: pyhealth.datasets.LNDbDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.lndb.read_mhd

.. autoclass:: pyhealth.datasets.lndb.MetaImage
    :members:
