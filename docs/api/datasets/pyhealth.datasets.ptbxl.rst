pyhealth.datasets.ptbxl
=======================

Overview
--------
PTB-XL is a large publicly available 12-lead ECG dataset containing 21 799
10-second records from 18 869 patients (Wagner et al., 2020). This PyHealth
wrapper maps each record to four binary diagnostic superclass labels —
myocardial infarction (MI), hypertrophy (HYP), ST/T-change (STTC), and
conduction disturbance (CD) — enabling direct use with
:class:`~pyhealth.tasks.ecg_classification.ECGBinaryClassification`.

Source: https://physionet.org/content/ptb-xl/1.0.3/

API Reference
-------------

.. autoclass:: pyhealth.datasets.ptbxl.PTBXLDataset
   :members:
   :undoc-members:
   :show-inheritance:
