pyhealth.datasets.PADUFES20Dataset
==================================

Overview
--------

PAD-UFES-20: 2,298 smartphone (clinical, not dermoscopy) photos of 1,641 skin
lesions from 1,373 patients in Brazil, with six diagnoses (ACK, BCC, MEL, NEV,
SCC, SEK) and up to 22 clinical features per image. Open access under CC BY
4.0. ``root`` must hold ``metadata.csv`` and the unzipped ``images/`` folder.
``patient_id`` is the PAD patient id, so patient-level splits keep all of a
patient's lesions together.

Source: `PAD-UFES-20 on Mendeley Data <https://data.mendeley.com/datasets/zr7vgbcyr2/1>`_

API Reference
-------------

.. autoclass:: pyhealth.datasets.PADUFES20Dataset
    :members:
    :undoc-members:
    :show-inheritance:
