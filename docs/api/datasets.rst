Datasets
===============

Getting Started
---------------

New to PyHealth datasets? Start here:

- **Tutorial**: `Introduction to pyhealth.datasets <https://colab.research.google.com/drive/1voSx7wEfzXfEf2sIfW6b-8p1KqMyuWxK?usp=sharing>`_ | `Video (PyHealth 1.6) <https://www.youtube.com/watch?v=c1InKqFJbsI&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=3>`_

This tutorial covers:

- How to load and work with different healthcare datasets (MIMIC-III, MIMIC-IV, eICU, etc.)
- Understanding the ``BaseDataset`` structure and patient representation
- Parsing raw EHR data into standardized PyHealth format
- Accessing patient records, visits, and clinical events
- Dataset splitting for train/validation/test sets

**Data Access**: If you're new and need help accessing MIMIC datasets, check the :doc:`../how_to_contribute` guide's "Data Access for Testing" section for information on:

- Getting MIMIC credentialing through PhysioNet
- Using openly available demo datasets (MIMIC-III Demo, MIMIC-IV Demo)
- Working with synthetic data for testing

Available Datasets
------------------

.. toctree::
    :maxdepth: 3

    datasets/pyhealth.datasets.BaseDataset
    datasets/pyhealth.datasets.SampleDataset
    datasets/pyhealth.datasets.MIMIC3Dataset
    datasets/pyhealth.datasets.MIMIC4Dataset
    datasets/pyhealth.datasets.MedicalTranscriptionsDataset
    datasets/pyhealth.datasets.CardiologyDataset
    datasets/pyhealth.datasets.eICUDataset
    datasets/pyhealth.datasets.ISRUCDataset
    datasets/pyhealth.datasets.MIMICExtractDataset
    datasets/pyhealth.datasets.OMOPDataset
    datasets/pyhealth.datasets.DREAMTDataset
    datasets/pyhealth.datasets.SHHSDataset
    datasets/pyhealth.datasets.SleepEDFDataset
    datasets/pyhealth.datasets.SIIMISICDataset
    datasets/pyhealth.datasets.EHRShotDataset
    datasets/pyhealth.datasets.Support2Dataset
    datasets/pyhealth.datasets.BMDHSDataset
    datasets/pyhealth.datasets.COVID19CXRDataset
    datasets/pyhealth.datasets.ChestXray14Dataset
    datasets/pyhealth.datasets.TUABDataset
    datasets/pyhealth.datasets.TUEVDataset
    datasets/pyhealth.datasets.ClinVarDataset
    datasets/pyhealth.datasets.COSMICDataset
    datasets/pyhealth.datasets.TCGAPRADDataset
    datasets/pyhealth.datasets.splitter
    datasets/pyhealth.datasets.utils
