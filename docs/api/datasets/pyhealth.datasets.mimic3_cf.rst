pyhealth.datasets.mimic3_cf
===========================

Overview
--------

MIMIC3CirculatoryFailureDataset is a MIMIC-III based dataset for early warning
prediction of circulatory failure.

It constructs an ICU-stay-level cohort from PATIENTS, ADMISSIONS, and ICUSTAYS,
and uses CHARTEVENTS to extract Mean Arterial Pressure (MAP) measurements.

Circulatory failure is defined using a proxy event:

- MAP < 65 mmHg

For each ICU stay, the dataset identifies the first occurrence of this event and
supports building task-ready patient records for downstream prediction tasks.

API Reference
-------------

.. autoclass:: pyhealth.datasets.MIMIC3CirculatoryFailureDataset
   :members:
   :undoc-members:
   :show-inheritance: