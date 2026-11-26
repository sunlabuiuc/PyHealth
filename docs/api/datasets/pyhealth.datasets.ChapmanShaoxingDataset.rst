pyhealth.datasets.ChapmanShaoxingDataset
========================================

Overview
--------

The Chapman University / Shaoxing and Ningbo hospitals 12-lead ECG database:
45,152 resting ECGs (10 s, 500 Hz), one per patient, each a WFDB ``.hea`` /
``.mat`` pair whose header lists SNOMED-CT diagnosis codes. Open access under
CC BY 4.0, no account needed. ``root`` is the version folder holding
``WFDBRecords/``.

Source: `PhysioNet ecg-arrhythmia 1.0.0 <https://physionet.org/content/ecg-arrhythmia/1.0.0/>`_

API Reference
-------------

.. autoclass:: pyhealth.datasets.ChapmanShaoxingDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.chapman_shaoxing.parse_chapman_header
