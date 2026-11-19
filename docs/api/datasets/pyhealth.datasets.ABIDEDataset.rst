pyhealth.datasets.ABIDEDataset
==============================

Overview
--------

ABIDE I resting-state fMRI preprocessed by the Preprocessed Connectomes
Project: 1,035 scans (505 autism, 530 typical controls) from 20 site codes,
as ROI time series for a chosen pipeline, noise strategy and atlas (default
C-PAC, ``filt_noglobal``, Craddock-200).

Source: `ABIDE Preprocessed <http://preprocessed-connectomes-project.org/abide/>`_
(`download guide <http://preprocessed-connectomes-project.org/abide/download.html>`_).
Paper: Di Martino et al., "The autism brain imaging data exchange",
Molecular Psychiatry 19, 659-667 (2014), https://doi.org/10.1038/mp.2013.78 ;
Craddock et al., "The Neuro Bureau Preprocessing Initiative" (2013),
https://doi.org/10.3389/conf.fninf.2013.09.00041 .

Getting the data
----------------

- **Access**: open. The PCP derivatives sit in the public S3 bucket
  ``s3://fcp-indi/data/Projects/ABIDE_Initiative/`` (HTTPS:
  ``https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/``) and
  need no account, sign-up or agreement. (The raw ABIDE images on NITRC ask
  for a NITRC registration; they are not used here.)
- **Files**: ``Phenotypic_V1_0b_preprocessed1.csv`` (about 450 KB) and one
  ``Outputs/<pipeline>/<strategy>/rois_<atlas>/<FILE_ID>_rois_<atlas>.1D``
  per scan (about 0.1-1 MB each; the C-PAC ``filt_noglobal`` ``rois_cc200``
  folder holds 1,102 files, about 430 MB).
- **Download, option 1**: ``ABIDEDataset(root, download=True)`` fetches the
  phenotypic file and the ROI files of the chosen selection into ``root``
  (``max_subjects=N`` for the first N scans only).
- **Download, option 2** (AWS CLI, no credentials):

  .. code-block:: bash

      aws s3 cp --no-sign-request s3://fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv <root>/
      aws s3 sync --no-sign-request s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_cc200/ <root>/Outputs/cpac/filt_noglobal/rois_cc200/

  Nothing needs unpacking.
- **root layout**:

  .. code-block:: text

      <root>/
          Phenotypic_V1_0b_preprocessed1.csv
          Outputs/
              cpac/                     # pipeline
                  filt_noglobal/        # strategy
                      rois_cc200/       # rois_<atlas>
                          Pitt_0050003_rois_cc200.1D
                          ...

  The chosen selection's S3 path is used first. ROI files kept elsewhere
  under ``root`` (e.g. a flat ``rois_cc200/``) are used only for scans that
  lack it, and files of another pipeline or strategy under ``Outputs/`` are
  ignored, so several selections can share one ``root``.
- **Licence**: Creative Commons Attribution-NonCommercial-ShareAlike 3.0
  (CC BY-NC-SA 3.0): non-commercial research use only, with attribution, and
  derivatives under the same licence.
- **Missing values**: scans without ``AGE_AT_SCAN`` get ``age = -1.0``.

API Reference
-------------

.. autoclass:: pyhealth.datasets.ABIDEDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.abide.abide_roi_relpath
