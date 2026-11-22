pyhealth.datasets.MIMICExtractDataset
===================================

Reads the HDF5 output of `MIMIC-Extract <https://github.com/MLforHealth/MIMIC_Extract>`_, a preprocessing pipeline over MIMIC-III. You need credentialed access to `MIMIC-III on PhysioNet <https://physionet.org/content/mimiciii/1.4/>`_ first; the class docstring below lists the ways to build or obtain the HDF5 files.

The dataset converts ``all_hourly_data.h5`` into two event tables, ``patients`` (one event per ICU stay, with outcomes) and ``vitals_labs`` (one event per stay, hour and variable). The binary ICU tasks in :doc:`../tasks/pyhealth.tasks.mimic_extract` (mortality and length of stay, as evaluated in CaliForest) run on it.

.. autoclass:: pyhealth.datasets.MIMICExtractDataset
    :members:
    :undoc-members:
    :show-inheritance:

   

   
   
   