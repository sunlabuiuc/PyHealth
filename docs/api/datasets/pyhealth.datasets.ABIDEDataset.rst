pyhealth.datasets.ABIDEDataset
==============================

Overview
--------

ABIDE I resting-state fMRI preprocessed by the Preprocessed Connectomes
Project: 1,035 scans (505 autism, 530 typical controls) from 20 site codes,
as ROI time series for a chosen pipeline, noise strategy and atlas (default
C-PAC, ``filt_noglobal``, Craddock-200). Open access from the ``fcp-indi`` S3
bucket under CC BY-NC-SA 3.0 (non-commercial). ``root`` holds
``Phenotypic_V1_0b_preprocessed1.csv`` and the ``*_rois_<atlas>.1D`` files;
``download=True`` fetches them.

Source: `ABIDE Preprocessed <http://preprocessed-connectomes-project.org/abide/>`_

API Reference
-------------

.. autoclass:: pyhealth.datasets.ABIDEDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.abide.abide_roi_relpath
